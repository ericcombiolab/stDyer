import omegaconf
import os
from os import path as osp
import collections
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
import numpy as np
import torch

from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR, CyclicLR, OneCycleLR
from dgl import NID
from sklearn.metrics import *
from src.models.modules.gmgat import VAE
from src.models.losses import LossFunctions
from src.utils.util import *
from sklearn.cluster import kmeans_plusplus
from scipy.spatial.distance import cdist
import time
from datetime import datetime
from sklearn.decomposition import PCA
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages
from src.utils.states import GMM_model, Dynamic_neigh_level
from torch.distributed import broadcast, gather, all_gather
import gc
import logging
logging.getLogger("anndata").setLevel(logging.WARNING)

# from torch.optim.lr_scheduler import SequentialLR, ConstantLR, ChainedScheduler
# from torch.autograd import detect_anomaly
# from statsmodels.stats.weightstats import DescrStatsW
# from src.utils.metric import recover_labels
# from memory_profiler import profile
# import gc
# from pytorch_memlab import profile

class GMVGAT(pl.LightningModule):
    """GMVGAT, inheriting the LightningModule, need to implement the training_step,
    validation_step, test_step and optimizers, different from the GMVAEModel, including
    neighbors feature reconstruction loss.

    Args:
        exp_encoder_in_channels (list): exp graph encoder input channels.
        exp_encoder_out_channels (list): exp graph encoder output channels.
        gaussian_size (int): size of latent space vector.
        num_classes (int): size of the gaussian mixture.
        lr (float): learning rate of the models.
        num_heads (int): number of heads in graph transformer block.
        exp_w_cat (float): exp graph encoder categorical loss weight.
        exp_w_gauss (float): exp graph encoder gaussian loss weight.
        exp_w_rec (float): exp graph encoder reconstruction loss weight.
        k (int): k neighbors used when plotting KNN graph.
        block_type (string): graph encoder type, support 'GCN' and 'Transformer'.
        plot_graph_size (int): size of logging graph.
        log_path (string): path to save the logging result.
        use_bias (True): whether use bias in models.
        dropout (float): dropout ratio in models.

    Attrs:
        SPGMGAT (VAE): sp graph encoder GMGAT model.
        EXPGMGAT (VAE): exp graph encoder GMGAT model.
    """
    def __init__(
        self,
        exp_encoder_channels=None,
        exp_encoder_in_channels=None,
        exp_encoder_out_channels=None,
        exp_decoder_channels="reverse",
        gaussian_size=512,
        lr=0.0001,
        lr_scheduler="cosine",
        T_max=None,
        cyclic_gamma="auto",
        attention_size=128,
        num_heads=1,
        y_block_type="Dense",
        z_block_type="Dense",
        exp_rec_type=None,
        use_bias=True,
        max_mu=10.,
        max_logvar=5.,
        min_logvar=-5.,
        use_kl_bn=False,
        kl_bn_gamma=32.,
        dropout=0.,
        weight_decay=0.0,
        prior='uniform',
        GMM_model_name="VVI",
        gaussian_kind="element-wise",
        prior_lr=0,
        allow_fewer_classes=False,
        activation='relu',
        prior_generator="fc",
        semi_ce_degree="just",
        exp_w_rec=1.,
        exp_w_gauss=1.,
        exp_w_cat=1.,
        exp_neigh_w_rec=None,
        exp_neigh_w_gauss=None,
        exp_neigh_w_cat=None,
        exp_self_w=1.,
        exp_neigh_w=1.,
        exp_sup_w_rec=1.,
        exp_sup_w_gauss=1.,
        exp_sup_w_cat=0.,
        sup_epochs=10,
        gaussian_start_epoch_pct=0,
        patience=10.,
        patience_start_epoch_pct=0.,
        patience_diff_pct=0.01,
        dynamic_neigh_quantile=0.5,
        use_pseudo_labels=False,
        detect_anomaly=False,
        clip_value=0.0,
        clip_type="norm",
        log_path="",
        plot_graph_size=200,
        print_loss=False,
        use_batch_norm=False,
        add_cat_bias=False,
        force_enough_classes=False,
        inbalance_weight=False,
        verbose_loss=False,
        legacy=False,
        debug=False,
    ):
        super().__init__()
        torch.autograd.set_detect_anomaly(detect_anomaly)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation.startswith("leaky_relu"):
            activation_splits = activation.split("_")
            if len(activation_splits) == 2:
                self.activation = nn.LeakyReLU()
            elif len(activation_splits) == 3:
                self.activation = nn.LeakyReLU(float(activation_splits[2]))
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        self.exp_encoder_channels = exp_encoder_channels
        self.exp_decoder_channels = exp_decoder_channels
        if self.exp_encoder_channels is not None:
            self.exp_encoder_in_channels = self.exp_encoder_channels[:-1]
            self.exp_encoder_out_channels = self.exp_encoder_channels[1:]
        else:
            self.exp_encoder_in_channels = exp_encoder_in_channels
            self.exp_encoder_out_channels = exp_encoder_out_channels
        if self.exp_decoder_channels == "reverse":
            self.exp_decoder_in_channels = self.exp_encoder_out_channels[::-1]
            self.exp_decoder_out_channels = self.exp_encoder_in_channels[::-1]
        elif self.exp_decoder_channels == "same":
            self.exp_decoder_in_channels = self.exp_encoder_in_channels
            self.exp_decoder_out_channels = self.exp_encoder_out_channels
        else:
            self.exp_decoder_in_channels = self.exp_decoder_channels[:-1]
            self.exp_decoder_out_channels = self.exp_decoder_channels[1:]
        if gaussian_size is None:
            self.gaussian_size = self.exp_encoder_out_channels[-1]
        else:
            self.gaussian_size = gaussian_size
        if attention_size is None:
            self.attention_size = self.gaussian_size
        else:
            self.attention_size = attention_size
        self.num_heads = num_heads
        self.y_block_type = y_block_type
        self.z_block_type = z_block_type
        self.use_bias = use_bias
        self.max_mu = max_mu
        self.max_logvar = max_logvar
        self.min_logvar = min_logvar
        self.use_kl_bn = use_kl_bn
        self.kl_bn_gamma = kl_bn_gamma
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.prior_generator = prior_generator
        self.semi_ce_degree = semi_ce_degree
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.cyclic_gamma = cyclic_gamma
        self.T_max = T_max

        self.log_base_path = log_path
        self.plot_graph_size = plot_graph_size
        self.exp_rec_type = exp_rec_type
        self.detect_anomaly = detect_anomaly
        # self.automatic_optimization = False
        self.automatic_optimization = True
        self.weight_decay = weight_decay
        self.highest_exp_ARI = -1
        self.clip_value = clip_value
        self.clip_type = clip_type
        # self.early_stopping_patience = 0
        self.last_epoch_centroid_threshold = None
        self.exp_w_cat = exp_w_cat
        self.exp_w_gauss = exp_w_gauss
        self.exp_w_rec = exp_w_rec
        self.exp_neigh_w_rec = exp_neigh_w_rec
        self.exp_neigh_w_gauss = exp_neigh_w_gauss
        self.exp_neigh_w_cat = exp_neigh_w_cat
        self.exp_sup_w_rec = exp_sup_w_rec
        self.exp_sup_w_gauss = exp_sup_w_gauss
        self.exp_sup_w_cat = exp_sup_w_cat
        self.sup_epochs = sup_epochs
        self.exp_self_w = exp_self_w
        self.exp_neigh_w = exp_neigh_w
        self.gaussian_start_epoch_pct = float(gaussian_start_epoch_pct)
        self.pred_diff_patience = 0.
        self.is_well_initialized = False
        self.patience = float(patience)
        self.patience_start_epoch_pct = float(patience_start_epoch_pct)
        self.patience_diff_pct = float(patience_diff_pct)
        self.dynamic_neigh_quantile = float(dynamic_neigh_quantile)
        self.use_pseudo_labels = use_pseudo_labels
        self.prior = prior
        self.GMM_model_name = GMM_model_name
        self.gaussian_kind = gaussian_kind
        self.prior_lr = prior_lr
        self.allow_fewer_classes = allow_fewer_classes
        self.print_loss = print_loss
        self.add_cat_bias = add_cat_bias
        self.force_enough_classes = force_enough_classes
        self.inbalance_weight = inbalance_weight

        self.last_epoch_exp_pred_df = None
        self.last_epoch_ARI = -1
        self.setup_num = 0
        self.not_enough_classes = False
        self.not_enough_train_classes = False
        self.force_add_prob_cat_domain = None
        self.stop_cluster_init = False
        self.is_mu_prior_learnable = False
        self.training_step_outputs = collections.defaultdict(list)
        self.validation_step_outputs = collections.defaultdict(list)
        self.verbose_loss = verbose_loss
        self.legacy = legacy
        self.debug = debug
        self.trainig_step_debug_outputs = collections.defaultdict(list)
        self.validation_step_debug_outputs = collections.defaultdict(list)
        self.validation_step_keep_outputs = collections.defaultdict(list)

    def setup(self, stage):
        self.setup_num += 1
        # print(f"{stage} setup start", self.setup_num)
        if self.setup_num == 1:
            self.sync_dist = True if self.trainer.world_size > 1 else False
            self.data_device = self.trainer.datamodule.device
            if self.patience_start_epoch_pct >= 1:
                self.patience_start_epoch_pct = self.patience_start_epoch_pct / self.trainer.max_epochs
                assert (self.trainer.max_epochs * self.patience_start_epoch_pct % 1) == 0
            if self.gaussian_start_epoch_pct >= 1:
                self.gaussian_start_epoch_pct = self.gaussian_start_epoch_pct / self.trainer.max_epochs
                assert (self.trainer.max_epochs * self.gaussian_start_epoch_pct % 1) == 0
            if self.T_max is None:
                self.T_max = self.trainer.max_epochs // 10
            if self.cyclic_gamma == "auto":
                self.cyclic_gamma = 1 - 1 / self.trainer.max_epochs
            self.exp_encoder_in_channels[0] = min(self.exp_encoder_in_channels[0], self.trainer.datamodule.data_val.adata.shape[1])
            self.exp_decoder_out_channels[-1] = min(self.exp_decoder_out_channels[-1], self.trainer.datamodule.data_val.adata.shape[1])
            self.weighted_neigh = self.trainer.datamodule.weighted_neigh
            self.num_classes = self.trainer.datamodule.data_val.num_classes
            self.num_cells = len(self.trainer.datamodule.data_val.adata)
            self.max_dynamic_neigh = self.trainer.datamodule.max_dynamic_neigh
            self.rec_mask_neigh_threshold = self.trainer.datamodule.rec_mask_neigh_threshold
            if self.max_dynamic_neigh:
                self.dynamic_neigh_level = self.trainer.datamodule.data_val.dynamic_neigh_level
                if self.exp_neigh_w == "auto":
                    self.exp_neigh_ws = torch.tensor(self.trainer.datamodule.data_val.dynamic_neigh_nums, dtype=torch.float32, device=self.data_device)
                else:
                    self.exp_neigh_ws = torch.tensor([self.exp_neigh_w] * len(self.trainer.datamodule.data_val.dynamic_neigh_nums), dtype=torch.float32, device=self.data_device)
            self.supervise_cat = self.trainer.datamodule.supervise_cat
            self.seed = self.trainer.datamodule.seed
            if 'is_labelled' in self.trainer.datamodule.data_val.adata.obs.keys():
                self.is_labelled = True
                self.plot_gt = True
            else:
                self.is_labelled = False
                self.plot_gt = False
            if self.plot_graph_size == "all":
                self.plot_graph_size = self.num_cells
            self.rng = np.random.default_rng(seed=self.seed)
            self.plot_node_idx = self.rng.choice(self.num_cells, size=self.plot_graph_size, replace=False)
            self.rec_neigh_num = self.trainer.datamodule.rec_neigh_num
            if self.rec_neigh_num == 'half':
                self.rec_neigh_num = self.trainer.datamodule.k // 2
            self.forward_neigh_num = self.trainer.datamodule.forward_neigh_num
            if not self.forward_neigh_num:
                assert self.y_block_type == "Dense" and self.z_block_type == "Dense"
            self.updated_prior = 1 / (self.num_classes * np.ones(self.num_classes))
            self.EXPGMGAT = VAE(
                encoder_in_channels=self.exp_encoder_in_channels,
                encoder_out_channels=self.exp_encoder_out_channels,
                decoder_in_channels=self.exp_decoder_in_channels,
                decoder_out_channels=self.exp_decoder_out_channels,
                attention_dim=self.attention_size,
                num_heads=self.num_heads,
                c_dim=self.num_cells,
                y_dim=self.num_classes,
                latent_dim=self.gaussian_size,
                dropout=self.dropout,
                use_bias=self.use_bias,
                y_block_type=self.y_block_type,
                z_block_type=self.z_block_type,
                rec_type=self.exp_rec_type,
                activation=self.activation,
                max_mu=self.max_mu,
                max_logvar=self.max_logvar,
                min_logvar=self.min_logvar,
                forward_neigh_num=self.forward_neigh_num,
                use_batch_norm=self.use_batch_norm,
                add_cat_bias=self.add_cat_bias,
                prior_generator=self.prior_generator,
                GMM_model_name=self.GMM_model_name,
                use_kl_bn=self.use_kl_bn,
                kl_bn_gamma=self.kl_bn_gamma,
                adata=self.trainer.datamodule.data_val.adata,
                device=self.data_device,
                legacy=self.legacy,
                debug=self.debug,
            )
            self.losses = LossFunctions(self.num_classes, GMM_model_name=self.GMM_model_name, nb_r=self.trainer.datamodule.data_val.adata.uns["r"], device=self.data_device)
            # self.losses = LossFunctions(self.num_classes, self.trainer.datamodule.data_train.out_log_sigma)
            # print("self.trainer.logger.version", self.trainer.logger.version)
            if self.trainer.is_global_zero:
                if isinstance(self.logger, CometLogger):
                    self.log_path = osp.join(self.log_base_path, self.trainer.logger.version)
                    with open("record_ids.txt", 'a') as f:
                        f.write(self.trainer.datamodule.sample_id + '|' + self.trainer.logger.version + "\n")
                else:
                    self.log_path = self.log_base_path
                if not osp.exists(self.log_path):
                    os.makedirs(self.log_path)
            # print("self.device", self.device) # cpu yet
            if isinstance(self.prior, str):
                pass
            elif isinstance(self.prior, omegaconf.listconfig.ListConfig):
                self.prior = torch.tensor(self.prior, dtype=torch.float32, device=self.data_device)
                self.prior = self.prior / self.prior.sum()
            else:
                print(type(self.prior))
                print(self.prior)
                raise NotImplementedError
            if self.use_pseudo_labels:
                assert self.rec_neigh_num
            self.time_dict = {"begin": time.time(), "end": time.time()}
        else:
            pass


    def forward(self):
        pass

    # @profile
    def dynamic_loss(self, x, out_net, output_nodes_idx, input_nodes, output_nodes, rec_neigh_attr, exp_rec_mask=None):
        data_index = output_nodes.cpu().numpy()
        neigh_loss_dict = {"reconstruction": torch.tensor(0.), "gaussian": torch.tensor(0.), "categorical": torch.tensor(0.)}
        mix_x_rec = out_net["reconstructed"]
        # logits, prob_cat: (A, C)
        mix_logits, mix_prob_cat = out_net['logits'], out_net['prob_cat']
        if self.inbalance_weight:
            mix_pred_labels = torch.argmax(mix_prob_cat, dim=-1)
            mix_pred_uni_label, mix_inverse, mix_pred_counts = torch.unique(mix_pred_labels, return_inverse=True, return_counts=True)
            mix_inbalance_weight = len(mix_pred_labels) / mix_pred_counts.float()
        # y_mus, y_logvars: (C, D)
        mix_y_mus, mix_y_logvars, mix_y_var_invs = out_net["y_means"], out_net["y_logvars"], out_net["y_var_invs"]
        # mus, logvars: (C, A, D)
        mix_mus, mix_logvars = out_net["means"], out_net["logvars"]
        x_rec = {}
        for dic_k, dic_v in mix_x_rec.items():
            # dic_v.shape: (C, A, D)
            x_rec[dic_k] = dic_v[:, output_nodes_idx, :]
        logits, prob_cat = mix_logits[output_nodes_idx], mix_prob_cat[output_nodes_idx]
        y_mus, y_logvars, y_var_invs = mix_y_mus, mix_y_logvars, mix_y_var_invs
        # print("output_nodes_idx", output_nodes_idx.shape)
        mus, logvars = mix_mus[:, output_nodes_idx, :], mix_logvars[:, output_nodes_idx, :]
        ref_attr = rec_neigh_attr[output_nodes_idx]
        loss_rec_dict = self.losses.batch_reconstruction_loss(x, x_rec, prob_cat, rec_mask=exp_rec_mask, rec_type=self.exp_rec_type, verbose=self.verbose_loss)
        del x_rec
        if self.inbalance_weight:
            loss_rec = (loss_rec_dict["loss"] * mix_inbalance_weight[mix_inverse][output_nodes_idx]).mean()
            loss_rec = loss_rec / loss_rec.detach() * loss_rec_dict["loss"].mean().detach()
        else:
            loss_rec = loss_rec_dict["loss"].mean()
        if self.verbose_loss:
            self.training_step_outputs["cluster_loss_rec"].append(loss_rec_dict["cluster_loss"][:, output_nodes_idx].detach().cpu().numpy())
            self.training_step_outputs["weighted_cluster_loss_rec"].append(loss_rec_dict["weighted_cluster_loss"][:, output_nodes_idx].detach().cpu().numpy())
        del loss_rec_dict
        loss_gauss_dict = self.losses.batch_expected_gaussian_loss(prob_cat, mus, logvars, y_mus, y_logvars, affi_weights=None, y_var_invs=y_var_invs, verbose=self.verbose_loss, kind=self.gaussian_kind)
        del mus, logvars
        if self.inbalance_weight:
            loss_gauss = (loss_gauss_dict["loss"] * mix_inbalance_weight[mix_inverse][output_nodes_idx]).mean()
            loss_gauss = loss_gauss / loss_gauss.detach() * loss_gauss_dict["loss"].mean().detach()
        else:
            loss_gauss = loss_gauss_dict["loss"].mean()
        if self.verbose_loss:
            self.training_step_outputs["cluster_loss_gauss"].append(loss_gauss_dict["cluster_loss"][:, output_nodes_idx].detach().cpu().numpy())
            self.training_step_outputs["weighted_cluster_loss_gauss"].append(loss_gauss_dict["weighted_cluster_loss"][:, output_nodes_idx].detach().cpu().numpy())
        del loss_gauss_dict
        loss_cat = self.losses.batch_expected_categorical_loss(prob_cat, affi_weights=None, prior=self.prior, learned_prior=self.updated_prior)
        if self.rec_neigh_num:
            ref_attr = rec_neigh_attr[output_nodes_idx]
            neigh_loss_rec = []
            neigh_loss_gauss = []
            neigh_prob_cat_list = []
            neigh_affi_list = []
            if self.dynamic_neigh_level == Dynamic_neigh_level.domain:
                for j, dynamic_neigh_num in enumerate(self.trainer.datamodule.data_val.dynamic_neigh_nums):
                    this_cluster_node_idx = self.trainer.datamodule.data_val.adata[data_index].obs["pred_labels"] == j
                    this_cluster_node_num = this_cluster_node_idx.sum()
                    if this_cluster_node_num == 0:
                        continue
                    # dynamic neigh idx in original adata: flatten(this_cluster_node_num x this_cluster_dynamic_neigh_num)
                    dynamic_neigh_idx = self.trainer.datamodule.data_val.adata[data_index].obsm['sp_k'][this_cluster_node_idx, :dynamic_neigh_num].flatten()
                    # dynamic neigh idx in batch: flatten(this_cluster_node_num x this_cluster_dynamic_neigh_num)
                    batch_dynamic_neigh_idx = torch.nonzero(input_nodes == torch.from_numpy(dynamic_neigh_idx).to(input_nodes).unsqueeze(1))[:, 1] #.reshape(-1, dynamic_neigh_num)
                    neigh_x_rec = {}
                    for dic_k, dic_v in mix_x_rec.items():
                        # dic_v.shape: (C, A, D)
                        neigh_x_rec[dic_k] = dic_v[:, batch_dynamic_neigh_idx, :]
                    neigh_prob_cat = mix_prob_cat[batch_dynamic_neigh_idx]
                    neigh_y_mus, neigh_y_logvars, neigh_y_var_invs = mix_y_mus, mix_y_logvars, mix_y_var_invs
                    neigh_mus, neigh_logvars = mix_mus[:, batch_dynamic_neigh_idx, :], mix_logvars[:, batch_dynamic_neigh_idx, :]
                    this_cluster_rec_neigh_attr = rec_neigh_attr[batch_dynamic_neigh_idx].view(this_cluster_node_num, dynamic_neigh_num, -1)
                    # print("this_cluster_rec_neigh_attr", this_cluster_rec_neigh_attr.shape) # B x dynamic_neigh_num x D
                    # print("ref_attr[this_cluster_node_idx]", ref_attr[this_cluster_node_idx].shape) # B x D
                    if self.rec_neigh_num:
                        if self.weighted_neigh:
                            neigh_affi = self.normalized_gaussian_kernel(torch.sqrt(torch.square(this_cluster_rec_neigh_attr - ref_attr[this_cluster_node_idx].unsqueeze(1)).sum(-1))).flatten()
                            neigh_affi_list.append(neigh_affi)
                        else:
                            neigh_affi = 1. / dynamic_neigh_num
                            neigh_affi_list = 1.
                    neigh_prob_cat_list.append(neigh_prob_cat)
                    # print("x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0)", x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0).shape)
                    # for dic_k, dic_v in neigh_x_rec.items():
                    #     print(dic_k, dic_v.shape)
                    if isinstance(exp_rec_mask, float):
                        tmp_neigh_loss_rec_dict = self.losses.batch_reconstruction_loss(x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), neigh_x_rec, neigh_prob_cat, rec_mask=exp_rec_mask, rec_type=self.exp_rec_type, real_origin=x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), affi_weights=neigh_affi, verbose=self.verbose_loss)
                    else:
                        tmp_neigh_loss_rec_dict = self.losses.batch_reconstruction_loss(x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), neigh_x_rec, neigh_prob_cat, rec_mask=exp_rec_mask[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), rec_type=self.exp_rec_type, real_origin=x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), affi_weights=neigh_affi, verbose=self.verbose_loss)
                    del neigh_x_rec
                    tmp_neigh_loss_rec = tmp_neigh_loss_rec_dict["loss"].view(-1, dynamic_neigh_num).sum(-1) * self.exp_neigh_ws[j]
                    del tmp_neigh_loss_rec_dict
                    neigh_loss_rec.append(tmp_neigh_loss_rec)
                    tmp_neigh_loss_gauss_dict = self.losses.batch_expected_gaussian_loss(neigh_prob_cat, neigh_mus, neigh_logvars, neigh_y_mus, neigh_y_logvars, affi_weights=neigh_affi, y_var_invs=neigh_y_var_invs, verbose=self.verbose_loss, kind=self.gaussian_kind)
                    tmp_neigh_loss_gauss = tmp_neigh_loss_gauss_dict["loss"].view(-1, dynamic_neigh_num).sum(-1) * self.exp_neigh_ws[j]
                    del tmp_neigh_loss_gauss_dict, neigh_mus, neigh_logvars
                    neigh_loss_gauss.append(tmp_neigh_loss_gauss)
            elif self.dynamic_neigh_level == Dynamic_neigh_level.unit or self.dynamic_neigh_level == Dynamic_neigh_level.unit_freq_self:
                dynamic_neigh_nums_arr = np.array(self.trainer.datamodule.data_val.dynamic_neigh_nums)
                for dynamic_neigh_num in np.unique(dynamic_neigh_nums_arr):
                    # the cluster here refers to units with the same number of dynamic neighbours (instead of referring to a spatial domain as above)
                    this_cluster_node_idx = dynamic_neigh_nums_arr[data_index] == dynamic_neigh_num
                    this_cluster_node_num = this_cluster_node_idx.sum()
                    if this_cluster_node_num == 0 or dynamic_neigh_num == 0:
                        continue
                    consist_k = self.trainer.datamodule.data_val.adata[data_index][this_cluster_node_idx].obsm["sp_k"][self.trainer.datamodule.data_val.adata[data_index][this_cluster_node_idx].obsm["consist_adj"]]
                    # dynamic neigh idx in original adata: flatten(this_cluster_node_num x this_cluster_dynamic_neigh_num)
                    dynamic_neigh_idx = consist_k.flatten()
                    # dynamic neigh idx in batch: flatten(this_cluster_node_num x this_cluster_dynamic_neigh_num)
                    batch_dynamic_neigh_idx = torch.nonzero(input_nodes == torch.from_numpy(dynamic_neigh_idx).to(input_nodes).unsqueeze(1))[:, 1] #.reshape(-1, dynamic_neigh_num)
                    neigh_x_rec = {}
                    for dic_k, dic_v in mix_x_rec.items():
                        # dic_v.shape: (C, A, D)
                        neigh_x_rec[dic_k] = dic_v[:, batch_dynamic_neigh_idx, :]
                    neigh_prob_cat = mix_prob_cat[batch_dynamic_neigh_idx]
                    neigh_y_mus, neigh_y_logvars, neigh_y_var_invs = mix_y_mus, mix_y_logvars, mix_y_var_invs
                    neigh_mus, neigh_logvars = mix_mus[:, batch_dynamic_neigh_idx, :], mix_logvars[:, batch_dynamic_neigh_idx, :]
                    this_cluster_rec_neigh_attr = rec_neigh_attr[batch_dynamic_neigh_idx].view(this_cluster_node_num, dynamic_neigh_num, -1)
                    # print("this_cluster_rec_neigh_attr", this_cluster_rec_neigh_attr.shape) # B x dynamic_neigh_num x D
                    # print("ref_attr[this_cluster_node_idx]", ref_attr[this_cluster_node_idx].shape) # B x D
                    if self.rec_neigh_num:
                        if self.weighted_neigh:
                            neigh_affi = self.normalized_gaussian_kernel(torch.sqrt(torch.square(this_cluster_rec_neigh_attr - ref_attr[this_cluster_node_idx].unsqueeze(1)).sum(-1))).flatten()
                            neigh_affi_list.append(neigh_affi)
                        else:
                            neigh_affi = 1. / dynamic_neigh_num
                            neigh_affi_list = 1.
                    neigh_prob_cat_list.append(neigh_prob_cat)
                    # print("x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0)", x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0).shape)
                    # for dic_k, dic_v in neigh_x_rec.items():
                    #     print(dic_k, dic_v.shape)
                    if isinstance(exp_rec_mask, float):
                        tmp_neigh_loss_rec_dict = self.losses.batch_reconstruction_loss(x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), neigh_x_rec, neigh_prob_cat, rec_mask=exp_rec_mask, rec_type=self.exp_rec_type, real_origin=x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), affi_weights=neigh_affi, verbose=self.verbose_loss)
                    else:
                        tmp_neigh_loss_rec_dict = self.losses.batch_reconstruction_loss(x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), neigh_x_rec, neigh_prob_cat, rec_mask=exp_rec_mask[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), rec_type=self.exp_rec_type, real_origin=x[this_cluster_node_idx].repeat_interleave(dynamic_neigh_num, dim=0), affi_weights=neigh_affi, verbose=self.verbose_loss)
                    del neigh_x_rec
                    tmp_neigh_loss_rec = tmp_neigh_loss_rec_dict["loss"].view(-1, dynamic_neigh_num).sum(-1) * self.exp_neigh_ws[output_nodes_idx][this_cluster_node_idx]
                    del tmp_neigh_loss_rec_dict
                    neigh_loss_rec.append(tmp_neigh_loss_rec)
                    tmp_neigh_loss_gauss_dict = self.losses.batch_expected_gaussian_loss(neigh_prob_cat, neigh_mus, neigh_logvars, neigh_y_mus, neigh_y_logvars, affi_weights=neigh_affi, y_var_invs=neigh_y_var_invs, verbose=self.verbose_loss, kind=self.gaussian_kind)
                    tmp_neigh_loss_gauss = tmp_neigh_loss_gauss_dict["loss"].view(-1, dynamic_neigh_num).sum(-1) * self.exp_neigh_ws[output_nodes_idx][this_cluster_node_idx]
                    del tmp_neigh_loss_gauss_dict, neigh_mus, neigh_logvars
                    neigh_loss_gauss.append(tmp_neigh_loss_gauss)
            elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_fix_domain or self.dynamic_neigh_level == Dynamic_neigh_level.unit_fix_domain_boundary or self.dynamic_neigh_level == Dynamic_neigh_level.unit_domain_boundary:
                used_k = self.trainer.datamodule.data_val.adata[data_index].obsm["used_k"]
                # dynamic neigh idx in original adata: flatten(B x K)
                dynamic_neigh_idx = used_k.flatten()
                # dynamic neigh idx in batch: flatten(B x K)
                batch_dynamic_neigh_idx = torch.nonzero(input_nodes == torch.from_numpy(dynamic_neigh_idx).to(input_nodes).unsqueeze(1))[:, 1] #.reshape(-1, dynamic_neigh_num)
                neigh_x_rec = {}
                for dic_k, dic_v in mix_x_rec.items():
                    # dic_v.shape: (C, A, D)
                    neigh_x_rec[dic_k] = dic_v[:, batch_dynamic_neigh_idx, :]
                del mix_x_rec, out_net["reconstructed"]
                neigh_prob_cat = mix_prob_cat[batch_dynamic_neigh_idx]
                neigh_y_mus, neigh_y_logvars, neigh_y_var_invs = mix_y_mus, mix_y_logvars, mix_y_var_invs
                this_cluster_rec_neigh_attr = rec_neigh_attr[batch_dynamic_neigh_idx].view(len(data_index), used_k.shape[1], -1)
                # print("this_cluster_rec_neigh_attr", this_cluster_rec_neigh_attr.shape) # B x K x D
                # print("ref_attr", ref_attr.shape) # B x D
                if self.rec_neigh_num:
                    if self.weighted_neigh:
                        neigh_affi = self.normalized_gaussian_kernel(torch.sqrt(torch.square(this_cluster_rec_neigh_attr - ref_attr.unsqueeze(1)).sum(-1))).flatten()
                        neigh_affi_list.append(neigh_affi)
                    else:
                        neigh_affi = 1. / used_k.shape[1]
                        neigh_affi_list = 1.
                neigh_prob_cat_list.append(neigh_prob_cat)
                # for dic_k, dic_v in neigh_x_rec.items():
                #     print(dic_k, dic_v.shape)
                if isinstance(exp_rec_mask, float):
                    tmp_neigh_loss_rec_dict = self.losses.batch_reconstruction_loss(x.repeat_interleave(used_k.shape[1], dim=0), neigh_x_rec, neigh_prob_cat, rec_mask=exp_rec_mask, rec_type=self.exp_rec_type, real_origin=x.repeat_interleave(used_k.shape[1], dim=0), affi_weights=neigh_affi, verbose=self.verbose_loss)
                else:
                    tmp_neigh_loss_rec_dict = self.losses.batch_reconstruction_loss(x.repeat_interleave(used_k.shape[1], dim=0), neigh_x_rec, neigh_prob_cat, rec_mask=exp_rec_mask.repeat_interleave(used_k.shape[1], dim=0), rec_type=self.exp_rec_type, real_origin=x.repeat_interleave(used_k.shape[1], dim=0), affi_weights=neigh_affi, verbose=self.verbose_loss)
                del neigh_x_rec
                tmp_neigh_loss_rec = tmp_neigh_loss_rec_dict["loss"].view(-1, used_k.shape[1]).sum(-1) * self.exp_neigh_ws[output_nodes_idx]
                del tmp_neigh_loss_rec_dict
                neigh_loss_rec.append(tmp_neigh_loss_rec)
                neigh_mus, neigh_logvars = mix_mus[:, batch_dynamic_neigh_idx, :], mix_logvars[:, batch_dynamic_neigh_idx, :]
                tmp_neigh_loss_gauss_dict = self.losses.batch_expected_gaussian_loss(neigh_prob_cat, neigh_mus, neigh_logvars, neigh_y_mus, neigh_y_logvars, affi_weights=neigh_affi, y_var_invs=neigh_y_var_invs, verbose=self.verbose_loss, kind=self.gaussian_kind)
                tmp_neigh_loss_gauss = tmp_neigh_loss_gauss_dict["loss"].view(-1, used_k.shape[1]).sum(-1) * self.exp_neigh_ws[output_nodes_idx]
                del tmp_neigh_loss_gauss_dict, neigh_mus, neigh_logvars, mix_logvars
                neigh_loss_gauss.append(tmp_neigh_loss_gauss)
            assert self.prior == "average_uniform_all_neighbors"
            if self.exp_neigh_w_rec != 0:
                neigh_loss_rec = torch.cat(neigh_loss_rec).mean()
            else:
                neigh_loss_rec = torch.tensor(0., device=self.data_device)
            if self.exp_neigh_w_gauss != 0:
                neigh_loss_gauss = torch.cat(neigh_loss_gauss).mean()
            else:
                neigh_loss_gauss = torch.tensor(0., device=self.data_device)
            if self.weighted_neigh:
                neigh_affi_list = torch.cat(neigh_affi_list)
            if self.exp_neigh_w_cat != 0:
                neigh_loss_cat = self.exp_neigh_ws.mean() * self.losses.batch_expected_categorical_loss(torch.cat(neigh_prob_cat_list), affi_weights=neigh_affi_list, prior=self.prior, learned_prior=self.updated_prior)
            else:
                neigh_loss_cat = torch.tensor(0., device=self.data_device)
            neigh_loss_total = self.exp_neigh_w_rec * neigh_loss_rec + self.exp_neigh_w_gauss * neigh_loss_gauss + self.exp_neigh_w_cat * neigh_loss_cat
            # self.time_dict["loss_sum"] = time.time()
            # print("Time for summing loss: ", self.time_dict["loss_sum"] - self.time_dict["neigh_loss"])
            neigh_loss_dict = {
                "total": neigh_loss_total,
                "reconstruction": neigh_loss_rec * self.exp_neigh_w_rec,
                "gaussian": neigh_loss_gauss * self.exp_neigh_w_gauss,
                "categorical": neigh_loss_cat * self.exp_neigh_w_cat,
            }
        loss_total = self.exp_w_rec * loss_rec + self.exp_w_gauss * loss_gauss + self.exp_w_cat * loss_cat
        loss_dict = {
            "total": loss_total,
            "reconstruction": loss_rec * self.exp_w_rec,
            "gaussian": loss_gauss * self.exp_w_gauss,
            "categorical": loss_cat * self.exp_w_cat,
            "logits": logits,
        }
        return loss_dict, neigh_loss_dict

    def gmvae_loss(self, x, out_net, exp_rec_mask=None, neigh_out_net=None, neigh_affi=None, cat=None, rec_cat=None, no_log=False):
        # with detect_anomaly():
        neigh_loss_dict = {"reconstruction": 0., "gaussian": 0., "categorical": 0.}
        debug_dict = None
        zs, x_rec = out_net["gaussians"], out_net["reconstructed"]
        # logits, prob_cat, y = out_net['logits'], out_net['prob_cat'], out_net['categorical']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        # print("prob_cat", prob_cat.shape)
        if self.supervise_cat:
            loss_sup_cat = CrossEntropyLoss()(logits, cat)
        # print("prob_cat", prob_cat.shape)
        y_mus, y_logvars, y_var_invs = out_net["y_means"], out_net["y_logvars"], out_net["y_var_invs"]
        mus, logvars = out_net["means"], out_net["logvars"]
        # self.time_dict["fetch_loss_data"] = time.time()
        # print("Time for fetching loss data: ", self.time_dict["fetch_loss_data"] - self.time_dict["forward"])
        # loss_rec, debug_dict = self.losses.batch_reconstruction_loss(x, x_rec, prob_cat, rec_type, debug=True)
        loss_rec_dict = self.losses.batch_reconstruction_loss(x, x_rec, prob_cat, rec_mask=exp_rec_mask, rec_type=self.exp_rec_type, verbose=self.verbose_loss)
        loss_gauss_dict = self.losses.batch_expected_gaussian_loss(prob_cat, mus, logvars, y_mus, y_logvars, affi_weights=None, y_var_invs=y_var_invs, verbose=self.verbose_loss, kind=self.gaussian_kind)
        loss_rec = loss_rec_dict["loss"].mean()
        loss_gauss = loss_gauss_dict["loss"].mean()
        loss_cat = self.losses.batch_expected_categorical_loss(prob_cat, affi_weights=None, prior=self.prior, learned_prior=self.updated_prior)
        if self.verbose_loss:
            self.training_step_outputs["cluster_loss_rec"].append(loss_rec_dict["cluster_loss"].detach().cpu().numpy())
            self.training_step_outputs["weighted_cluster_loss_rec"].append(loss_rec_dict["weighted_cluster_loss"].detach().cpu().numpy())
            self.training_step_outputs["cluster_loss_gauss"].append(loss_gauss_dict["cluster_loss"].detach().cpu().numpy())
            self.training_step_outputs["weighted_cluster_loss_gauss"].append(loss_gauss_dict["weighted_cluster_loss"].detach().cpu().numpy())
        if neigh_affi is not None:
            all_neigh_affi = neigh_affi.view(-1)
        else:
            all_neigh_affi = None
            curr_neigh_affi = None
        if neigh_out_net is not None:
        # if neigh_x is not None:
            # B, K, G = neigh_x.shape
            neigh_loss_rec = 0.
            neigh_loss_gauss = 0.
            neigh_loss_cat = 0.
            neigh_loss_sup_cat = 0.
            for k in range(self.rec_neigh_num):
                if neigh_affi is not None:
                    curr_neigh_affi = neigh_affi[:, k]
                neigh_zs, neigh_x_rec = neigh_out_net["gaussians"][k], {key: neigh_out_net["reconstructed"][key][k] for key in neigh_out_net["reconstructed"].keys()}
                # neigh_logits, neigh_prob_cat, neigh_y = neigh_out_net['logits'][k], neigh_out_net['prob_cat'][k], neigh_out_net['categorical'][k]
                neigh_logits, neigh_prob_cat = neigh_out_net['logits'][k], neigh_out_net['prob_cat'][k]
                # print("neigh_prob_cat", neigh_prob_cat.shape)
                # print("rec_cat", rec_cat.shape)
                # print("neigh_prob_cat", neigh_prob_cat.shape)
                neigh_y_mus, neigh_y_logvars = neigh_out_net["y_means"][k], neigh_out_net["y_logvars"][k]
                if neigh_out_net["y_var_invs"] != []:
                    neigh_y_var_invs = neigh_out_net["y_var_invs"][k]
                else:
                    neigh_y_var_invs = None
                neigh_mus, neigh_logvars = neigh_out_net["means"][k], neigh_out_net["logvars"][k]
                tmp_neigh_loss_rec_dict = self.losses.batch_reconstruction_loss(x, neigh_x_rec, neigh_prob_cat, rec_mask=exp_rec_mask, rec_type=self.exp_rec_type, real_origin=x, affi_weights=curr_neigh_affi, verbose=self.verbose_loss)
                tmp_neigh_loss_gauss_dict = self.losses.batch_expected_gaussian_loss(neigh_prob_cat, neigh_mus, neigh_logvars, neigh_y_mus, neigh_y_logvars, affi_weights=curr_neigh_affi, y_var_invs=neigh_y_var_invs, verbose=self.verbose_loss, kind=self.gaussian_kind)
                if self.prior != "average_uniform_all_neighbors":
                    tmp_neigh_loss_cat = self.losses.batch_expected_categorical_loss(neigh_prob_cat, affi_weights=curr_neigh_affi, prior=self.prior, learned_prior=self.updated_prior)
                tmp_neigh_loss_rec = tmp_neigh_loss_rec_dict["loss"]
                neigh_loss_rec += tmp_neigh_loss_rec.mean()
                neigh_loss_gauss += tmp_neigh_loss_gauss_dict["loss"].mean()
                if self.prior != "average_uniform_all_neighbors":
                    neigh_loss_cat += tmp_neigh_loss_cat
                if self.supervise_cat:
                    tmp_neigh_loss_sup_cat = CrossEntropyLoss()(neigh_logits, rec_cat[:, k]) / self.rec_neigh_num
                    neigh_loss_sup_cat += tmp_neigh_loss_sup_cat
            if self.prior == "average_uniform_all_neighbors":
                neigh_loss_cat = self.losses.batch_expected_categorical_loss(torch.stack(neigh_out_net['prob_cat'], 1).view(-1, neigh_prob_cat.shape[-1]), affi_weights=all_neigh_affi, prior=self.prior, learned_prior=self.updated_prior)
            # self.time_dict["neigh_loss"] = time.time()
            # print("Time for computing neighbor loss: ", self.time_dict["neigh_loss"] - self.time_dict["init_loss"])

            if self.weighted_neigh:
                pass
            else:
                neigh_loss_rec /= self.rec_neigh_num
                neigh_loss_gauss /= self.rec_neigh_num
            if self.prior == "average_uniform_all_neighbors":
                pass
            elif self.prior == "average_uniform":
                neigh_loss_cat /= self.rec_neigh_num
            else:
                raise
            if self.supervise_cat:
                neigh_loss_cat = neigh_loss_sup_cat
            neigh_loss_total = self.exp_neigh_w_rec * neigh_loss_rec + self.exp_neigh_w_gauss * neigh_loss_gauss + self.exp_neigh_w_cat * neigh_loss_cat
            # self.time_dict["loss_sum"] = time.time()
            # print("Time for summing loss: ", self.time_dict["loss_sum"] - self.time_dict["neigh_loss"])
            neigh_loss_dict = {
                "total": neigh_loss_total,
                "reconstruction": neigh_loss_rec * self.exp_neigh_w_rec,
                "gaussian": neigh_loss_gauss * self.exp_neigh_w_gauss,
                "categorical": neigh_loss_cat * self.exp_neigh_w_cat,
            }

        # print("loss_rec", loss_rec)
        # print("loss_gauss", loss_gauss)
        # print("loss_cat", loss_cat)
        # print("neigh_loss_rec", neigh_loss_rec)
        # print("neigh_loss_gauss", neigh_loss_gauss)
        # print("neigh_loss_cat", neigh_loss_cat)
        if self.supervise_cat:
            loss_cat = loss_sup_cat
        loss_total = self.exp_w_rec * loss_rec + self.exp_w_gauss * loss_gauss + self.exp_w_cat * loss_cat

        loss_dict = {
            "total": loss_total,
            "reconstruction": loss_rec * self.exp_w_rec,
            "gaussian": loss_gauss * self.exp_w_gauss,
            "categorical": loss_cat * self.exp_w_cat,
            "logits": logits,
            # "debug_dict": debug_dict,
        }

        return loss_dict, neigh_loss_dict

    # @profile
    def get_forward_outputs(self, exp_attr, forward_neigh_attr, rec_neigh_attr, rec_forward_neigh_attr):
        if not self.forward_neigh_num:
            exp_output_dict = self.EXPGMGAT(
                x=exp_attr,
            )
            # if self.rec_neigh_num and forward_neigh_attr is not None: # the rhs is for the case when we want to use semi_sup_rec_loss
            if self.rec_neigh_num:
                if self.max_dynamic_neigh:
                    return exp_output_dict, None
                else:
                    # B x K1 x G
                    # B, K, G = rec_neigh_attr.shape
                    # print("B, K, G", B, K, G)
                    # assert B % K == 0
                    neighbor_output_dict = collections.defaultdict(list)
                    for k in range(self.rec_neigh_num):
                        neighbor_temp_dict = self.EXPGMGAT(
                            x=rec_neigh_attr[:, k, :],
                        )
                        for key in neighbor_temp_dict:
                            if isinstance(neighbor_temp_dict[key], torch.Tensor):
                                neighbor_output_dict[key].append(neighbor_temp_dict[key])
                            elif isinstance(neighbor_temp_dict[key], dict):
                                if key not in neighbor_output_dict:
                                    neighbor_output_dict[key] = collections.defaultdict(list)
                                for subkey in neighbor_temp_dict[key]:
                                    neighbor_output_dict[key][subkey].append(neighbor_temp_dict[key][subkey])
            else:
                return exp_output_dict, None
            return exp_output_dict, neighbor_output_dict
        else:
            if self.max_dynamic_neigh:
                exp_output_dict = self.EXPGMGAT(
                    x=exp_attr,
                )
            else:
                exp_output_dict = self.EXPGMGAT(
                    x=exp_attr,
                    neighbor_x=forward_neigh_attr,
                )
                if self.rec_neigh_num:
                    # B x K1 x K2 x G
                    # B, K1, K2, G = rec_forward_neigh_attr.shape
                    neighbor_output_dict = collections.defaultdict(list)
                    for k in range(self.rec_neigh_num):
                        neighbor_temp_dict = self.EXPGMGAT(
                            x=rec_neigh_attr[:, k, :],
                            neighbor_x=rec_forward_neigh_attr[:, k, :, :],
                        )
                        for key in neighbor_temp_dict:
                            if isinstance(neighbor_temp_dict[key], torch.Tensor):
                                neighbor_output_dict[key].append(neighbor_temp_dict[key])
                            elif isinstance(neighbor_temp_dict[key], dict):
                                if key not in neighbor_output_dict:
                                    neighbor_output_dict[key] = collections.defaultdict(list)
                                for subkey in neighbor_temp_dict[key]:
                                    neighbor_output_dict[key][subkey].append(neighbor_temp_dict[key][subkey])
                    return exp_output_dict, neighbor_output_dict
            return exp_output_dict, None

    def add_prob_cat(self, exp_output_dict, neighbor_output_dict):
        if self.force_add_prob_cat_domain is not None:
            # # !!!!
            # self.prob_cat_to_add = self.prob_cat_to_add * self.num_classes
            # # !!!!
            if exp_output_dict is not None:
                exp_output_dict["prob_cat"] = exp_output_dict["prob_cat"] + self.prob_cat_to_add
                exp_output_dict["prob_cat"] = exp_output_dict["prob_cat"] / exp_output_dict["prob_cat"].sum(1, keepdim=True)
            if neighbor_output_dict is not None:
                for k in range(self.rec_neigh_num):
                    neighbor_output_dict["prob_cat"][k] = neighbor_output_dict["prob_cat"][k] + self.prob_cat_to_add
                    neighbor_output_dict["prob_cat"][k] = neighbor_output_dict["prob_cat"][k] / neighbor_output_dict["prob_cat"][k].sum(1, keepdim=True)
            if self.prior_generator.startswith("tensor"):
                for force_domain in self.force_add_prob_cat_domain:
                    self.EXPGMGAT.decoder.mu_prior[force_domain].data = nn.Parameter((torch.rand((self.EXPGMGAT.decoder.z_dim), device=self.device, requires_grad=self.EXPGMGAT.decoder.mu_prior.requires_grad) - 0.5) * 2 * np.sqrt(6 / (self.EXPGMGAT.decoder.y_dim + self.EXPGMGAT.decoder.z_dim)))
                    if self.EXPGMGAT.decoder.GMM_model_name == "VVI":
                        self.EXPGMGAT.decoder.logvar_prior[force_domain].data = nn.Parameter((torch.rand((self.EXPGMGAT.decoder.z_dim), device=self.device, requires_grad=self.EXPGMGAT.decoder.logvar_prior.requires_grad) - 0.5) * 2 * np.sqrt(6 / (self.EXPGMGAT.decoder.y_dim + self.EXPGMGAT.decoder.z_dim)))
                    elif self.EXPGMGAT.decoder.GMM_model_name == "EEE":
                        self.EXPGMGAT.decoder.logvar_prior = None
        return exp_output_dict, neighbor_output_dict

    # def on_train_start(self):
    #     pass

    # def on_train_epoch_start(self):
    #     if self.current_epoch == self.trainer.check_val_every_n_epoch and self.device.type == 'cuda':
    #         preallocate_mem(os.environ["CUDA_VISIBLE_DEVICES"])

    # def on_before_zero_grad(self, optimizer):
    #     print("start on_before_zero_grad")
    #     print("stop on_before_zero_grad")
    # def on_before_backward(self, loss):
    #     print("start on_before_backward")
    #     print("stop on_before_backward")
    # def backward(self, loss):
    #     print("start backward")
    #     print("loss", loss)
    #     loss.backward()
    #     print("stop backward")
    # def on_after_backward(self):
    #     print("start on_after_backward")
    #     print("stop on_after_backward")
    # def on_before_optimizer_step(self, optimizer):
    #     print("start on_before_optimizer_step")
    #     print("stop on_before_optimizer_step")
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     print("start on_train_batch_end")
    #     print("stop on_train_batch_end")

    # @profile
    def training_step(self, batch, batch_idx):
        # print(f"start training step {self.global_step}")
        # print(torch.cuda.memory_summary())
        if self.automatic_optimization is False:
            exp_opt = self.optimizers()
        # exp_attr: x
        # exp_rec_attr: rec_x
        # rec_neigh_attr: x's loss neighbors
        # forward_neigh_attr: x's forward neighbors
        # rec_forward_neigh_attr: x's forward neighbors' loss neighbors
        forward_neigh_attr = None
        rec_neigh_attr = None
        rec_forward_neigh_attr = None
        exp_rec_neigh_affi = 1.
        exp_rec_mask = 1.
        if self.max_dynamic_neigh:
            input_nodes, output_nodes, blocks = batch
            input_block = blocks[0]
            output_block = blocks[-1]
            if self.forward_neigh_num:
                rec_neigh_attr = output_block.srcdata["exp_feature"]
                if "exp_rec_feature" in output_block.dstdata:
                    exp_rec_attr = output_block.dstdata["exp_rec_feature"]
                else:
                    exp_rec_attr = output_block.dstdata["exp_feature"]
                input_nodes = input_block.dstdata[NID]
            else:
                rec_neigh_attr = input_block.srcdata["exp_feature"]
                if "exp_rec_feature" in output_block.dstdata:
                    exp_rec_attr = input_block.dstdata["exp_rec_feature"]
                else:
                    exp_rec_attr = input_block.dstdata["exp_feature"]
                exp_attr = input_block.dstdata["exp_feature"]
            data_index = output_nodes.to("cpu")
            # the index of output nodes in input_nodes: input_nodes[output_nodes_idx] == output_nodes
            output_nodes_idx = torch.nonzero(input_nodes == output_nodes.unsqueeze(1))[:, 1]
            if self.rec_mask_neigh_threshold:
                exp_rec_mask = input_block.dstdata["exp_rec_mask"]
            # self.trainig_step_debug_outputs["block"] = input_block
            # self.trainig_step_debug_outputs["input_nodes"] = input_nodes
            # self.trainig_step_debug_outputs["output_nodes"] = output_nodes

        else:
            data_index = batch['index'].to("cpu")
            exp_attr, exp_rec_attr = batch['exp_feature'], batch['exp_rec_feature']
            if self.rec_mask_neigh_threshold:
                exp_rec_mask = batch['exp_rec_mask']
            if self.supervise_cat:
                cat = batch["cat"].long()
            else:
                cat = None
            rec_cat = None
            if self.rec_neigh_num or self.forward_neigh_num:
                if self.rec_neigh_num:
                    if self.supervise_cat:
                        rec_cat = batch["rec_cat"].long()
                    rec_neigh_attr = batch['exp_rec_neigh_feature']
                    if self.weighted_neigh:
                        exp_rec_neigh_affi = batch["exp_rec_neigh_affi"]
                    else:
                        exp_rec_neigh_affi = None
                # print("exp_attr.shape", exp_attr.shape)
                # print("forward_neigh_attr.shape", forward_neigh_attr.shape)
                if self.forward_neigh_num:
                    forward_neigh_attr = batch['exp_forward_neigh_feature']
                if self.rec_neigh_num and self.forward_neigh_num:
                    rec_forward_neigh_attr = batch['exp_rec_forward_neigh_feature']
                    # exp_rec_forward_neigh_affi = batch["exp_rec_forward_neigh_affi"]
                    # exp_neigh_index = batch["exp_neigh_index"].long()
                    # exp_rec_forward_neigh_index = batch["exp_rec_forward_neigh_index"].long()
        if self.max_dynamic_neigh and self.forward_neigh_num:
            self.curr_batch_size = len(output_nodes)
        else:
            self.curr_batch_size = len(exp_attr)

        # self.time_dict["begin"] = time.time()
        # print("Time for loading data", self.time_dict["begin"] - self.time_dict["end"])

        if self.debug:
            debug_finite_param(self.EXPGMGAT, where="before forward")

        ##########################
        # Optimize EXPGMGAT #
        ##########################
        if self.max_dynamic_neigh:
            if not self.forward_neigh_num:
                exp_output_dict, neighbor_output_dict = self.get_forward_outputs(rec_neigh_attr, None, None, None)
                # # # (C x D), (C, B, D) -> (C, 1, B) -> (C, B) -> (B, C)
                # exp_output_dict["prob_cat"] = torch.cdist(exp_output_dict["y_means"].unsqueeze(1), exp_output_dict["means"]).squeeze(1).transpose(0, 1)
                # # # apply RBF kernel
                # exp_output_dict["prob_cat"] = F.softmax(torch.exp(-torch.square(exp_output_dict["prob_cat"]) / 2), 1)
            else:
                # TODO: can be simplified
                exp_output_dict, neighbor_output_dict = self.get_forward_outputs(input_block, None, None, None)
            if self.force_enough_classes:
                exp_output_dict, neighbor_output_dict = self.add_prob_cat(exp_output_dict, neighbor_output_dict)
            exp_loss_dict, neighbor_loss_dict = self.dynamic_loss(exp_rec_attr, exp_output_dict, output_nodes_idx, input_nodes, output_nodes, rec_neigh_attr, exp_rec_mask)
        else:
            exp_output_dict, neighbor_output_dict = self.get_forward_outputs(exp_attr, forward_neigh_attr, rec_neigh_attr, rec_forward_neigh_attr)

            exp_loss_dict, neighbor_loss_dict = self.gmvae_loss(exp_rec_attr, exp_output_dict, exp_rec_mask, neighbor_output_dict, exp_rec_neigh_affi, cat=cat, rec_cat=rec_cat)

        if self.debug:
            self.trainig_step_debug_outputs["exp_output_dict"].append(copy_dict_to_cpu(exp_output_dict))
            self.trainig_step_debug_outputs["neighbor_output_dict"].append(copy_dict_to_cpu(neighbor_output_dict))
            debug_finite_dict(exp_output_dict, "after forward")
            debug_finite_dict(neighbor_output_dict, "after forward")
            self.trainig_step_debug_outputs["exp_loss_dict"].append(copy_dict_to_cpu(exp_loss_dict))
            self.trainig_step_debug_outputs["neighbor_loss_dict"].append(copy_dict_to_cpu(neighbor_loss_dict))
            debug_finite_dict(exp_loss_dict, "loss")
            debug_finite_dict(neighbor_loss_dict, "loss")

        if self.max_dynamic_neigh:
            # already weigh neighbor loss when computing loss
            exp_reconstruction_loss = self.exp_self_w * exp_loss_dict["reconstruction"] + neighbor_loss_dict["reconstruction"]
            exp_gaussian_loss = self.exp_self_w * exp_loss_dict["gaussian"] + neighbor_loss_dict["gaussian"]
            exp_categorical_loss = self.exp_self_w * exp_loss_dict["categorical"] + neighbor_loss_dict["categorical"]
        else:
            exp_reconstruction_loss = self.exp_self_w * exp_loss_dict["reconstruction"] + self.exp_neigh_w * neighbor_loss_dict["reconstruction"]
            exp_gaussian_loss = self.exp_self_w * exp_loss_dict["gaussian"] + self.exp_neigh_w * neighbor_loss_dict["gaussian"]
            exp_categorical_loss = self.exp_self_w * exp_loss_dict["categorical"] + self.exp_neigh_w * neighbor_loss_dict["categorical"]
        # self.time_dict["loss"] = time.time()
        # print("Time for loss", self.time_dict["loss"] - self.time_dict["forward"])
        if self.gaussian_start_epoch_pct == 0.:
            exp_loss = exp_reconstruction_loss + exp_gaussian_loss + exp_categorical_loss
        else:
            if self.current_epoch < self.gaussian_start_epoch_pct * self.trainer.max_epochs:
                exp_loss = exp_reconstruction_loss + exp_categorical_loss
            elif (self.stop_cluster_init is False) and (self.current_epoch < (self.gaussian_start_epoch_pct * self.trainer.max_epochs + self.sup_epochs)):
                if self.prior_generator.endswith("mclust") or self.prior_generator.endswith("mclust_rec"):
                    cluster_init_loss = self.cluster_init_loss(exp_output_dict, cluster_labels=self.valid_cluster_init_labels[input_nodes][output_nodes_idx], output_nodes_idx=output_nodes_idx, degree=self.semi_ce_degree)
                    self.training_step_outputs["epoch_cluster_init_loss"].append(cluster_init_loss.detach().cpu())
                    exp_loss = self.exp_sup_w_rec * exp_reconstruction_loss + self.exp_sup_w_gauss * exp_gaussian_loss + \
                        self.exp_sup_w_cat * exp_categorical_loss + cluster_init_loss
                    # exp_loss = exp_reconstruction_loss + exp_gaussian_loss + cluster_init_loss
                    # exp_loss = exp_reconstruction_loss + cluster_init_loss
                elif (self.prior_generator == "tensor") or (self.prior_generator == "fc"):
                    # exp_loss = exp_reconstruction_loss + 0.1 * exp_gaussian_loss + exp_categorical_loss
                    exp_loss = exp_reconstruction_loss + exp_gaussian_loss + exp_categorical_loss
            else:
                exp_loss = exp_reconstruction_loss + exp_gaussian_loss + exp_categorical_loss
        # print("curr_epoch", self.current_epoch,
        #     "mu_prior.sum", exp_output_dict["y_means"].sum().cpu(),
        #     #   "exp_loss", exp_loss.detach().cpu().item(),
        #     #   "exp_reconstruction_loss", exp_reconstruction_loss.detach().cpu().item(),
        #     #   "exp_gaussian_loss", exp_gaussian_loss.detach().cpu().item(),
        #     #   "exp_categorical_loss", exp_categorical_loss.detach().cpu().item()
        #       )
        # if exp_output_dict["y_means"].grad is not None:
        #     print("mu_prior.grad.mean", exp_output_dict["y_means"].grad.mean().cpu())
        # print("exp_reconstruction_loss", exp_reconstruction_loss)
        # print("exp_gaussian_loss", exp_gaussian_loss)
        # print("exp_categorical_loss", exp_categorical_loss)
        if self.debug:
            self.trainig_step_debug_outputs["exp_reconstruction_loss"].append(exp_reconstruction_loss.detach().cpu())
            self.trainig_step_debug_outputs["exp_gaussian_loss"].append(exp_gaussian_loss.detach().cpu())
            self.trainig_step_debug_outputs["exp_categorical_loss"].append(exp_categorical_loss.detach().cpu())
            self.trainig_step_debug_outputs["exp_loss"].append(exp_loss.detach().cpu())
        if self.automatic_optimization is False:
            raise NotImplementedError
            exp_opt.zero_grad()
            # with detect_anomaly():
            # self.manual_backward(exp_loss)
            self.manual_backward(exp_reconstruction_loss, retain_graph=True)
            self.manual_backward(exp_gaussian_loss, retain_graph=True)
            self.manual_backward(exp_categorical_loss)
            if self.debug:
                debug_finite_grad(self.EXPGMGAT, "backward")
            exp_opt.step()
            # self.time_dict["backward"] = time.time()
            # print("Time for backward", self.time_dict["backward"] - self.time_dict["loss"])
            if self.use_pseudo_labels and self.is_well_initialized:
                exp_output_dict, neighbor_output_dict = self.get_forward_outputs(exp_attr, forward_neigh_attr, rec_neigh_attr, rec_forward_neigh_attr)
                exp_opt.zero_grad()
                semi_sup_loss = self.semi_sup_loss(exp_output_dict, neighbor_output_dict)
                self.manual_backward(semi_sup_loss)
                exp_opt.step()
                self.log("train/semi_sup_loss", semi_sup_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        if self.max_dynamic_neigh:
            exp_prob_cat = exp_output_dict["prob_cat"][output_nodes_idx].detach().half().cpu()
            exp_means = exp_output_dict["means"][:, output_nodes_idx].detach().half().cpu()
        else:
            exp_prob_cat = exp_output_dict["prob_cat"].detach().half().cpu()
            exp_means = exp_output_dict["means"].detach().half().cpu()
        exp_pred_labels = exp_prob_cat.argmax(-1).flatten()
        self.training_step_outputs["data_index"].append(data_index)
        self.training_step_outputs["epoch_exp_pred_labels"].append(exp_pred_labels)
        self.training_step_outputs["epoch_exp_prob_cat"].append(exp_prob_cat)
        self.training_step_outputs["epoch_exp_means"].append(exp_means)
        if self.verbose_loss:
            if self.exp_rec_type == "Gaussian":
                exp_reconstructed = exp_output_dict["reconstructed"]["z_1"].detach().cpu()
            elif self.exp_rec_type == "NegativeBinomial":
                exp_reconstructed = exp_output_dict["reconstructed"]["z_k"].detach().cpu()
            elif self.exp_rec_type == "MSE":
                exp_reconstructed = exp_output_dict["reconstructed"]["z"].detach().cpu()
            self.training_step_outputs["epoch_exp_reconstructed"].append(exp_reconstructed)
        self.training_step_outputs["epoch_exp_reconstruction_loss"].append(exp_reconstruction_loss.detach().cpu())
        self.training_step_outputs["epoch_exp_gaussian_loss"].append(exp_gaussian_loss.detach().cpu())
        self.training_step_outputs["epoch_exp_categorical_loss"].append(exp_categorical_loss.detach().cpu())
        self.training_step_outputs["epoch_exp_loss"].append(exp_loss.detach().cpu())
        self.training_step_outputs["exp_self_reconstruction_loss"].append(exp_loss_dict["reconstruction"].detach().cpu())
        self.training_step_outputs["exp_self_gaussian_loss"].append(exp_loss_dict["gaussian"].detach().float().cpu())
        self.training_step_outputs["exp_self_categorical_loss"].append(exp_loss_dict["categorical"].detach().cpu())
        self.training_step_outputs["exp_neigh_reconstruction_loss"].append(neighbor_loss_dict["reconstruction"].detach().cpu())
        self.training_step_outputs["exp_neigh_gaussian_loss"].append(neighbor_loss_dict["gaussian"].detach().cpu())
        self.training_step_outputs["exp_neigh_categorical_loss"].append(neighbor_loss_dict["categorical"].detach().cpu())
        # self.time_dict["end"] = time.time()
        # print("Time for logging: ", self.time_dict["end"] - self.time_dict["backward"])
        # print(f"stop training step {self.global_step}")
        # print(torch.cuda.memory_summary())
        return {"loss": exp_loss}

    # @profile
    def on_train_epoch_end(self):
        # print(f"start train epoch end {self.current_epoch}")
        # print(torch.cuda.memory_summary())
        # if (self.stop_cluster_init is True) and (self.prior_generator.startswith("tensor")) and (self.is_mu_prior_learnable is False):
        if (self.current_epoch == (self.gaussian_start_epoch_pct * self.trainer.max_epochs + self.sup_epochs - 1)) and (self.prior_generator.startswith("tensor")) and ("mclust" in self.prior_generator) and (self.is_mu_prior_learnable is False):
            self.EXPGMGAT.decoder.mu_prior = nn.Parameter(self.EXPGMGAT.decoder.mu_prior, requires_grad=True)
            print("mu_prior becomes learnable")
            # print("mu_prior remains fixed")
            self.is_mu_prior_learnable = True
        for k, v in self.training_step_outputs.items():
            if k == "epoch_exp_means":
                self.training_step_outputs[k] = torch.cat(v, dim=1)
            else:
                if isinstance(v[0], torch.Tensor):
                    if v[0].ndim > 0:
                        self.training_step_outputs[k] = torch.cat(v, dim=0)
                    else:
                        self.training_step_outputs[k] = torch.cat([vv.view(-1) for vv in v], dim=0)
        if self.trainer.is_global_zero:
            predicted_domains = torch.unique(self.training_step_outputs["epoch_exp_pred_labels"])
            if len(predicted_domains) < self.num_classes:
                self.not_enough_train_classes = True
                # if self.current_epoch >= (self.gaussian_start_epoch_pct * self.trainer.max_epochs + self.sup_epochs):
                self.force_add_prob_cat_domain = list(set(range(self.num_classes)) - set(predicted_domains.tolist()))
                self.prob_cat_to_add = torch.zeros((1, self.num_classes), dtype=torch.float32, device=self.device)
                self.prob_cat_to_add[0, self.force_add_prob_cat_domain] = 1. / self.num_classes
            else:
                self.not_enough_train_classes = False
                self.force_add_prob_cat_domain = None
        exp_reconstruction_loss = torch.mean(self.training_step_outputs["epoch_exp_reconstruction_loss"])
        exp_gaussian_loss = torch.mean(self.training_step_outputs["epoch_exp_gaussian_loss"])
        exp_categorical_loss = torch.mean(self.training_step_outputs["epoch_exp_categorical_loss"])
        exp_loss = torch.mean(self.training_step_outputs["epoch_exp_loss"])
        exp_self_reconstruction_loss = torch.mean(self.training_step_outputs["exp_self_reconstruction_loss"])
        exp_self_gaussian_loss = torch.mean(self.training_step_outputs["exp_self_gaussian_loss"])
        exp_self_categorical_loss = torch.mean(self.training_step_outputs["exp_self_categorical_loss"])
        exp_neigh_reconstruction_loss = torch.mean(self.training_step_outputs["exp_neigh_reconstruction_loss"])
        exp_neigh_gaussian_loss = torch.mean(self.training_step_outputs["exp_neigh_gaussian_loss"])
        exp_neigh_categorical_loss = torch.mean(self.training_step_outputs["exp_neigh_categorical_loss"])
        if "epoch_cluster_init_loss" in self.training_step_outputs:
            epoch_cluster_init_loss = torch.mean(self.training_step_outputs["epoch_cluster_init_loss"])
            self.log("train/epoch_cluster_init_loss", epoch_cluster_init_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        # logging loss curve
        self.log("train/exp_reconstruction_loss", exp_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/exp_self_reconstruction_loss", exp_self_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/exp_neigh_reconstruction_loss", exp_neigh_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/exp_gaussian_loss", exp_gaussian_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/exp_self_gaussian_loss", exp_self_gaussian_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/exp_neigh_gaussian_loss", exp_neigh_gaussian_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/exp_categorical_loss", exp_categorical_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/exp_self_categorical_loss", exp_self_categorical_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/exp_neigh_categorical_loss", exp_neigh_categorical_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        self.log("train/loss", exp_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        if self.print_loss:
            if self.trainer.is_global_zero:
                print("rank0: rec, gauss, cat", exp_reconstruction_loss, exp_gaussian_loss, exp_categorical_loss)
                self.trainer.strategy.barrier()
        # self.training_step_outputs["data_index"] = np.concatenate(self.training_step_outputs["data_index"], axis=0)
        # self.training_step_outputs["epoch_exp_pred_labels"] = np.concatenate(self.training_step_outputs["epoch_exp_pred_labels"], axis=0)
        # self.training_step_outputs["epoch_exp_prob_cat"] = np.concatenate(self.training_step_outputs["epoch_exp_prob_cat"], axis=0)
        # self.training_step_outputs["epoch_exp_means"] = np.concatenate(self.training_step_outputs["epoch_exp_means"], axis=1)
        if self.trainer.world_size == 1:
            data_index = self.training_step_outputs["data_index"]
            data_label = self.training_step_outputs["epoch_exp_pred_labels"]
        else:
            data_index = [torch.empty_like(self.training_step_outputs["data_index"]) for _ in range(self.trainer.world_size)]
            data_label = [torch.empty_like(self.training_step_outputs["epoch_exp_pred_labels"]) for _ in range(self.trainer.world_size)]
            with torch.no_grad():
                # gather(self.training_step_outputs["data_index"], data_index)
                # gather(self.training_step_outputs["epoch_exp_pred_labels"], data_label)
                all_gather(data_index, self.training_step_outputs["data_index"])
                all_gather(data_label, self.training_step_outputs["epoch_exp_pred_labels"])
            # !!! data_index may be repeated
            data_index = torch.cat(data_index, dim=0)
            data_label = torch.cat(data_label, dim=0)
        data_index = data_index.numpy()
        data_label = data_label.numpy()
        # self.trainer.datamodule.data_val.adata.obs.iloc[data_index, adata_ref.obs.columns.get_loc("pred_labels")] = self.training_step_outputs["epoch_exp_pred_labels"]
        # epoch_exp_pred_df = pd.DataFrame({"exp_pred_labels": self.training_step_outputs["epoch_exp_pred_labels"]}, index=data_index)
        self.trainer.datamodule.data_val.adata.obs.iloc[data_index, self.trainer.datamodule.data_val.adata.obs.columns.get_loc("pred_labels")] = data_label
        adata_ref = self.trainer.datamodule.data_val.adata
        epoch_exp_pred_df = pd.DataFrame({"exp_pred_labels": data_label}, index=data_index)
        if self.last_epoch_exp_pred_df is not None:
            common_index = epoch_exp_pred_df.index.intersection(self.last_epoch_exp_pred_df.index)
            pred_diff_pct = (epoch_exp_pred_df.loc[common_index, "exp_pred_labels"].to_numpy() != self.last_epoch_exp_pred_df.loc[common_index, "exp_pred_labels"].to_numpy()).mean()
            # print("compare", np.unique(epoch_exp_pred_labels, return_counts=True), np.unique(self.last_epoch_exp_pred_labels, return_counts=True))
            # print("len(common_index)", len(common_index))
            # print("pred_diff_pct", pred_diff_pct)
            # if (self.stop_cluster_init is False) and (pred_diff_pct < 0.01) and (self.prior_generator == "tensor_mclust"):
            #     self.stop_cluster_init = True
            self.log("train/pred_diff_pct", pred_diff_pct, on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.sync_dist)
            if self.current_epoch >= self.patience_start_epoch_pct * self.trainer.max_epochs - 1: # [0, 49], 50~, data preparation starts at the end of the 49th epoch
                self.trainer.datamodule.recreate_dgl_dataset = True
                if self.max_dynamic_neigh:
                    if self.dynamic_neigh_level == Dynamic_neigh_level.domain:
                        for j in range(len(self.trainer.datamodule.data_val.dynamic_neigh_nums)):
                            this_cluster_node_idx = adata_ref[data_index].obs["pred_labels"] == j
                            if this_cluster_node_idx.sum() == 0:
                                continue
                            # neigh idx in original adata: this_cluster_node_num x max_dynamic_neigh_num
                            this_cluster_neigh_idx = adata_ref[data_index].obsm['sp_k'][this_cluster_node_idx, :]
                            this_cluster_neigh_label = adata_ref.obs["pred_labels"].to_numpy()[this_cluster_neigh_idx]
                            self.trainer.datamodule.data_val.dynamic_neigh_nums[j] = min(self.max_dynamic_neigh, max(1, int(np.quantile((this_cluster_neigh_label == j).sum(1), self.dynamic_neigh_quantile))))
                        # print(self.trainer.datamodule.data_val.dynamic_neigh_nums)
                    elif self.dynamic_neigh_level == Dynamic_neigh_level.unit:
                        # self.trainer.datamodule.data_val.adata.obsm["consist_adj"][data_index] = adata_ref.obs["pred_labels"].to_numpy()[adata_ref[data_index].obsm["sp_k"]] == adata_ref[data_index].obs["pred_labels"].to_numpy()[:, np.newaxis]
                        all_cat_count = self.batched_bincount(torch.from_numpy(adata_ref.obs["pred_labels"].to_numpy()[adata_ref[data_index].obsm["sp_k"]]), dim=1, max_value=self.num_classes).numpy()
                        each_obs_neighbor_cat = all_cat_count.argmax(1)
                        self.trainer.datamodule.data_val.adata.obsm["consist_adj"][data_index] = adata_ref.obs["pred_labels"].to_numpy()[adata_ref[data_index].obsm["sp_k"]] == each_obs_neighbor_cat[:, np.newaxis]
                        # self.trainer.datamodule.data_val.dynamic_neigh_nums = np.minimum(self.max_dynamic_neigh, np.maximum(1, self.trainer.datamodule.data_val.adata.obsm["consist_adj"].sum(1)))
                        self.trainer.datamodule.data_val.dynamic_neigh_nums = np.minimum(self.max_dynamic_neigh, self.trainer.datamodule.data_val.adata.obsm["consist_adj"].sum(1))
                    elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_freq_self:
                        all_cat_count = self.batched_bincount(torch.from_numpy(adata_ref.obs["pred_labels"].to_numpy()[np.concatenate((data_index[:, np.newaxis], adata_ref[data_index].obsm["sp_k"]), 1)]), dim=1, max_value=self.num_classes).numpy()
                        # each_obs_neighbor_cat_values, each_obs_neighbor_cats = torch.topk(all_cat_count, k=2, dim=1)
                        # is_first_two_large_neighbor_domains_equal = (each_obs_neighbor_cat_values[:, 0] == each_obs_neighbor_cat_values[:, 1]).numpy()
                        # # for each obs, if the first two largest neighbor domains are not equal, then the obs has neighbors of the largest domain
                        # each_obs_neighbor_cat = each_obs_neighbor_cats[:, 0].numpy()
                        # self.trainer.datamodule.data_val.adata.obsm["consist_adj"][data_index][~is_first_two_large_neighbor_domains_equal] = \
                        #     adata_ref.obs["pred_labels"].to_numpy()[np.concatenate((data_index, adata_ref[data_index].obsm["sp_k"]), 1)][~is_first_two_large_neighbor_domains_equal] == each_obs_neighbor_cat[~is_first_two_large_neighbor_domains_equal][:, np.newaxis]
                        # self.trainer.datamodule.data_val.dynamic_neigh_nums[data_index][~is_first_two_large_neighbor_domains_equal] = \
                        #     np.minimum(self.max_dynamic_neigh, self.trainer.datamodule.data_val.adata.obsm["consist_adj"][data_index][~is_first_two_large_neighbor_domains_equal].sum(1))

                        # for each obs, if the first two largest neighbor domains are not equal, then the obs has neighbors of one of the largest domain
                        each_obs_neighbor_cat = all_cat_count.argmax(1)
                        self.trainer.datamodule.data_val.adata.obsm["consist_adj"][data_index] = adata_ref.obs["pred_labels"].to_numpy()[adata_ref[data_index].obsm["sp_k"]] == each_obs_neighbor_cat[:, np.newaxis]
                        self.trainer.datamodule.data_val.dynamic_neigh_nums = np.minimum(self.max_dynamic_neigh, self.trainer.datamodule.data_val.adata.obsm["consist_adj"].sum(1))
                    elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_fix_domain or self.dynamic_neigh_level == Dynamic_neigh_level.unit_fix_domain_boundary:
                        self.trainer.datamodule.start_use_domain_neigh = True
                    elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_domain_boundary:
                        self.trainer.datamodule.data_val.adata.obsm["consist_adj"][data_index] = adata_ref.obs["pred_labels"].to_numpy()[adata_ref[data_index].obsm["sp_k"]] == adata_ref.obs["pred_labels"].to_numpy()[:, np.newaxis]
                        self.trainer.datamodule.data_val.dynamic_neigh_nums = np.minimum(self.max_dynamic_neigh, self.trainer.datamodule.data_val.adata.obsm["consist_adj"].sum(1))
                    if self.exp_neigh_w == "auto":
                        self.exp_neigh_ws = torch.tensor(self.trainer.datamodule.data_val.dynamic_neigh_nums, dtype=torch.float32, device=self.data_device)
                if pred_diff_pct < self.patience_diff_pct:
                    self.pred_diff_patience += 1.
                else:
                    self.pred_diff_patience = max(0., self.pred_diff_patience - self.patience)
                    # self.early_stopping_patience += 1
        else:
            self.log("train/pred_diff_pct", 0., on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.sync_dist)
        self.is_well_initialized = self.pred_diff_patience >= self.patience
        self.log("train/pred_diff_patience", self.pred_diff_patience, on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.sync_dist)
        self.last_epoch_exp_pred_df = epoch_exp_pred_df

        # print(self.current_epoch, "lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0])
        if self.prior_lr > 0:
            mean_prob_cat = self.training_step_outputs["epoch_exp_prob_cat"].mean(0).numpy()
            if self.allow_fewer_classes or mean_prob_cat.min() * len(self.training_step_outputs["epoch_exp_prob_cat"]) >= 100:
                # updated_prior = self.updated_prior + self.prior_lr * (mean_prob_cat - self.updated_prior)
                # updated_prior = updated_prior / updated_prior.sum()
                # if updated_prior.max() > self.updated_prior.max():
                #     self.updated_prior = self.updated_prior + self.prior_lr * (mean_prob_cat - self.updated_prior)
                # else:
                #     self.updated_prior = self.updated_prior + self.prior_lr * 0.01 * (mean_prob_cat - self.updated_prior)
                self.updated_prior = self.updated_prior + self.prior_lr * (mean_prob_cat - self.updated_prior)
                self.updated_prior = self.updated_prior / self.updated_prior.sum()
            # if self.num_classes < 8:
            #     print("updated_prior", self.updated_prior)
        if self.verbose_loss:
            self.training_step_outputs["epoch_exp_reconstructed"] = torch.concatenate(self.training_step_outputs["cluster_loss_rec"], axis=1)
            self.training_step_outputs["cluster_loss_rec"] = torch.concatenate(self.training_step_outputs["cluster_loss_rec"], axis=1)
            self.training_step_outputs["cluster_loss_gauss"] = torch.concatenate(self.training_step_outputs["cluster_loss_gauss"], axis=1)
            self.training_step_outputs["weighted_cluster_loss_rec"] = torch.concatenate(self.training_step_outputs["weighted_cluster_loss_rec"], axis=1)
            self.training_step_outputs["weighted_cluster_loss_gauss"] = torch.concatenate(self.training_step_outputs["weighted_cluster_loss_gauss"], axis=1)
            # raise
            # self.training_step_outputs["cluster_loss_rec"] = np.concatenate(self.training_step_outputs["cluster_loss_rec"], axis=1).mean(1)
            # self.training_step_outputs["cluster_loss_gauss"] = np.concatenate(self.training_step_outputs["cluster_loss_gauss"], axis=1).mean(1)
            # self.training_step_outputs["neigh_cluster_loss_rec"] = np.zeros_like(self.training_step_outputs["cluster_loss_rec"])
            # self.training_step_outputs["neigh_cluster_loss_gauss"] = np.zeros_like(self.training_step_outputs["cluster_loss_gauss"])
            # for k in range(self.trainer.datamodule.k):
            #     self.training_step_outputs[f"{k}_neigh_cluster_loss_rec"] = np.concatenate(self.training_step_outputs[f"{k}_neigh_cluster_loss_rec"], axis=1).mean(1)
            #     self.training_step_outputs[f"{k}_neigh_cluster_loss_gauss"] = np.concatenate(self.training_step_outputs[f"{k}_neigh_cluster_loss_gauss"], axis=1).mean(1)
            #     self.training_step_outputs["neigh_cluster_loss_rec"] += self.training_step_outputs[f"{k}_neigh_cluster_loss_rec"]
            #     self.training_step_outputs["neigh_cluster_loss_gauss"] += self.training_step_outputs[f"{k}_neigh_cluster_loss_gauss"]
            # self.training_step_outputs["neigh_cluster_loss_rec"] /= self.trainer.datamodule.k
            # self.training_step_outputs["neigh_cluster_loss_gauss"] /= self.trainer.datamodule.k
            # for c in range(len(self.training_step_outputs["cluster_loss_rec"])):
            #     self.log(f"train/cluster_{c}_loss_rec", self.training_step_outputs["cluster_loss_rec"][c], on_step=False, on_epoch=True, prog_bar=False)
            #     self.log(f"train/cluster_{c}_loss_gauss", self.training_step_outputs["cluster_loss_gauss"][c], on_step=False, on_epoch=True, prog_bar=False)
            #     self.log(f"train/cluster_{c}_neigh_loss_rec", self.training_step_outputs["neigh_cluster_loss_rec"][c], on_step=False, on_epoch=True, prog_bar=False)
            #     self.log(f"train/cluster_{c}_neigh_loss_gauss", self.training_step_outputs["neigh_cluster_loss_gauss"][c], on_step=False, on_epoch=True, prog_bar=False)
        self.training_step_outputs.clear()
        if self.debug:
            self.trainig_step_debug_outputs.clear()
        # print(f"stop train epoch end {self.current_epoch}")
        # print(torch.cuda.memory_summary())

    # def on_validation_model_eval(self):
    #     print("start validation_model_eval")
    #     print("stop validation_model_eval")

    # def on_validation_start(self):
    #     print("start validation_start")
    #     print("stop validation_start")

    # def on_validation_epoch_start(self):
    #     print("start validation_epoch_start")
    #     print("stop validation_epoch_start")

    # def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     print("start validation_batch_start")
    #     print("stop validation_batch_start")

    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     print("start before_batch_transfer")
    #     print("stop before_batch_transfer")
    #     return batch

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     print("start transfer_batch_to_device")
    #     print("stop transfer_batch_to_device")
    #     batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
    #     return batch

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     print("start after_batch_transfer")
    #     print("stop after_batch_transfer")
    #     return batch

    # @profile
    def validation_step(self, batch, batch_idx):
        # print(f"start valid step {self.global_step}")
        # print(torch.cuda.memory_summary())
        # print("max_memory_allocated", torch.cuda.max_memory_allocated() / 10**9, "GB")
        # print("max_memory_reserved", torch.cuda.max_memory_reserved() / 10**9, "GB")
        if self.max_dynamic_neigh:
            # input_nodes, output_nodes, blocks = batch
            _, output_nodes, blocks = batch
            input_block = blocks[0]
            if self.forward_neigh_num:
                # input_nodes = input_block.dstdata[NID]
                # input_nodes = input_block.srcdata[NID]
                # self.validation_step_debug_outputs["input_nodes"] = input_nodes
                # self.validation_step_debug_outputs["output_nodes"] = output_nodes
                # self.validation_step_debug_outputs["block"] = blocks
                # return
                pass
            else:
                exp_attr = input_block.dstdata["exp_feature"]
            data_index = output_nodes.cpu()
        else:
            forward_neigh_attr = None
            rec_neigh_attr = None
            rec_forward_neigh_attr = None
            data_index = batch["index"].to("cpu")
            exp_attr = batch['exp_feature']
            # exp_vis_bi_adj, sp_vis_bi_adj, labels = batch['exp_vis_bi_adj'], batch['sp_vis_bi_adj'], batch['labels']
            # sp_vis_bi_adj, labels = batch['sp_vis_bi_adj'], batch['labels']
            if self.forward_neigh_num:
                forward_neigh_attr = batch['exp_forward_neigh_feature']
            if self.rec_neigh_num:
                rec_neigh_attr = batch['exp_rec_neigh_feature']
                # exp_neigh_index = batch['exp_neigh_index'].long()
            if self.forward_neigh_num and self.rec_neigh_num:
                rec_forward_neigh_attr = batch['exp_rec_forward_neigh_feature']
                # exp_rec_forward_neigh_index = batch['exp_rec_forward_neigh_index'].long()

        if not self.forward_neigh_num:
            exp_output_dict = self.EXPGMGAT(
                x=exp_attr,
            )
        else:
            if self.max_dynamic_neigh:
                exp_output_dict = self.EXPGMGAT(
                    x=input_block,
                )
            else:
                exp_output_dict = self.EXPGMGAT(
                    x=exp_attr,
                    neighbor_x=forward_neigh_attr,
                )

        # # # (C x D), (C, B, D) -> (C, 1, B) -> (C, B) -> (B, C)
        # exp_output_dict["prob_cat"] = torch.cdist(exp_output_dict["y_means"].unsqueeze(1), exp_output_dict["means"]).squeeze(1).transpose(0, 1)
        # # # apply RBF kernel
        # exp_output_dict["prob_cat"] = F.softmax(torch.exp(-torch.square(exp_output_dict["prob_cat"]) / 2), 1)

        exp_prob_cat = exp_output_dict["prob_cat"].detach().half().cpu()
        if self.debug and exp_output_dict["att_x"] is not None:
            exp_att_x = exp_output_dict["att_x"].detach().cpu()
        # exp_latent = exp_output_dict["gaussians"].detach().cpu()
        exp_y_means = exp_output_dict["y_means"].detach().half().cpu()
        # exp_y_logvars = exp_output_dict["y_logvars"].detach().cpu()
        exp_means = exp_output_dict["means"].detach().half().cpu()
        # exp_logvars = exp_output_dict["logvars"].detach().cpu()
        if self.exp_rec_type == "Gaussian":
            # exp_rec = exp_output_dict["reconstructed"]["z_1"].detach().half().cpu() # huge memory consumption for large datasets
            exp_mix_rec = (exp_output_dict["reconstructed"]["z_1"].detach() * exp_output_dict["prob_cat"].detach().T.unsqueeze(-1)).half().cpu()
        # elif self.exp_rec_type == "MSE":
        #     exp_rec = exp_output_dict["reconstructed"]["z"].detach().cpu()
        if self.max_dynamic_neigh:
            pass
            # self.validation_step_outputs["input_nodes"] = input_nodes.detach().cpu()
            # self.validation_step_outputs["output_nodes"] = output_nodes.detach().cpu()
        else:
            if self.rec_neigh_num:
                # B x K1 x G
                B, K, G = rec_neigh_attr.shape
                neighbor_output_dict = collections.defaultdict(list)
                for k in range(self.rec_neigh_num):
                    if not self.forward_neigh_num:
                        neighbor_temp_dict = self.EXPGMGAT(
                            x=rec_neigh_attr[:, k, :],
                        )
                    else:
                        neighbor_temp_dict = self.EXPGMGAT(
                            x=rec_neigh_attr[:, k, :],
                            neighbor_x=rec_forward_neigh_attr[:, k, :, :],
                        )
                    for key in neighbor_temp_dict:
                        if isinstance(neighbor_temp_dict[key], torch.Tensor):
                            neighbor_output_dict[key].append(neighbor_temp_dict[key])
                        elif isinstance(neighbor_temp_dict[key], dict):
                            if key not in neighbor_output_dict:
                                neighbor_output_dict[key] = collections.defaultdict(list)
                            for subkey in neighbor_temp_dict[key]:
                                neighbor_output_dict[key][subkey].append(neighbor_temp_dict[key][subkey])
                for key in neighbor_output_dict:
                    # print(key, type(neighbor_output_dict[key]))
                    if isinstance(neighbor_output_dict[key], list):
                        # print(key, neighbor_output_dict[key][0].shape)
                        if key in ["means", "logvars", "gaussians"]:
                            # C x B x D -> C x B x K x D
                            neighbor_output_dict[key] = torch.stack(neighbor_output_dict[key], dim=2).reshape(self.num_classes, B, self.rec_neigh_num, -1)
                        # elif key in ["logits", "prob_cat", "categorical"]:
                        elif key in ["logits", "prob_cat"]:
                            # B x C -> B x K x C
                            neighbor_output_dict[key] = torch.stack(neighbor_output_dict[key], dim=1).reshape(B, self.rec_neigh_num, self.num_classes)
                        elif key in ["y_means", "y_logvars"]:
                            # C x D
                            # assert (neighbor_output_dict[key][0] == neighbor_output_dict[key][-1]).all()
                            if not (neighbor_output_dict[key][0] == neighbor_output_dict[key][-1]).all():
                                print("Nan found in output")
                            neighbor_output_dict[key] = neighbor_output_dict[key][0]
                    elif isinstance(neighbor_output_dict[key], collections.defaultdict):
                        # a dict of lists of Tensors
                        for subkey in neighbor_output_dict[key]:
                            # print(key, subkey, type(neighbor_output_dict[key][subkey][0]))
                            if subkey in ['z_r', 'z_p']:
                                # D
                                # print(key, subkey, neighbor_output_dict[key][subkey][0].shape)
                                neighbor_output_dict[key][subkey] = neighbor_output_dict[key][subkey][0]
                            elif subkey in ['z']:
                                # C x B x D -> C x B x K x D
                                neighbor_output_dict[key][subkey] = torch.stack(neighbor_output_dict[key][subkey], dim=2).reshape(self.num_classes, B, self.rec_neigh_num, -1)
                            elif subkey in ['z_r_type']:
                                # str
                                neighbor_output_dict[key][subkey] = neighbor_output_dict[key][subkey][0]
                exp_neigh_prob_cat = neighbor_output_dict["prob_cat"].detach().half().cpu()
                exp_neigh_pred_labels = exp_neigh_prob_cat.argmax(-1)
                self.validation_step_outputs["epoch_exp_neigh_pred_labels"].append(exp_neigh_pred_labels)
        exp_pred_labels = exp_prob_cat.argmax(-1).flatten()
        self.validation_step_outputs["data_index"].append(data_index)
        self.validation_step_outputs["epoch_exp_pred_labels"].append(exp_pred_labels)
        self.validation_step_outputs["epoch_exp_prob_cat"].append(exp_prob_cat)
        # if self.device.type == "cpu":
        #     self.validation_step_keep_outputs["epoch_exp_prob_cat"].append(exp_prob_cat)
        #     self.validation_step_keep_outputs["epoch_exp_gaussians"].append(exp_output_dict["gaussians"].detach().cpu())
        if self.debug and exp_output_dict["att_x"] is not None:
            self.validation_step_debug_outputs["epoch_exp_att_x"].append(exp_att_x)
        # self.validation_step_outputs["epoch_exp_latent"].append(exp_latent)
        self.validation_step_outputs["epoch_exp_y_means"] = exp_y_means
        # self.validation_step_outputs["epoch_exp_y_logvars"] = exp_y_logvars
        self.validation_step_outputs["epoch_exp_means"].append(exp_means)
        # self.validation_step_outputs["epoch_exp_logvars"].append(exp_logvars)
        # self.validation_step_outputs["epoch_exp_rec"].append(exp_rec)
        self.validation_step_outputs["epoch_exp_mix_rec"].append(exp_mix_rec)
        # print(f"stop valid step {self.global_step}")
        # print(torch.cuda.memory_summary())

    # @profile
    def on_validation_epoch_end(self):
        # print(f"start valid epoch end {self.current_epoch}")
        # print(torch.cuda.memory_summary())
        # if self.max_dynamic_neigh and self.forward_neigh_num:
        #     raise NotImplementedError
        # else:
        # model_checkpoint.every_n_epochs
        if self.trainer.is_global_zero:
            if (self.current_epoch + 1) % 50 == 0:
                self.trainer.datamodule.data_val.adata.obs["pred_labels"].to_csv(osp.join(self.log_path, f"pred_labels_epoch_{self.current_epoch}.csv.gz"))
                written_adata = self.trainer.datamodule.data_val.adata.copy()
                del written_adata.uns
                written_adata.write_h5ad(osp.join(self.log_path, f"epoch_{self.current_epoch}.h5ad"), compression="gzip")
        for k, v in self.validation_step_outputs.items():
            if (k == "epoch_exp_means") or (k == "epoch_exp_rec") or (k == "epoch_exp_mix_rec"):
                self.validation_step_outputs[k] = torch.cat(v, dim=1) # huge memory consumption for large datasets
            elif k == "epoch_exp_y_means":
                pass
            else:
                self.validation_step_outputs[k] = torch.cat(v, dim=0)

        # print('self.validation_step_outputs["epoch_exp_pred_labels"] before no grad', self.validation_step_outputs["epoch_exp_pred_labels"].shape)
        with torch.no_grad():
            if self.trainer.world_size == 1:
                epoch_data_index = self.validation_step_outputs["data_index"].numpy()
                epoch_exp_pred_labels = self.validation_step_outputs["epoch_exp_pred_labels"].numpy()
                epoch_exp_prob_cat = self.validation_step_outputs["epoch_exp_prob_cat"].numpy()
                epoch_exp_y_means = self.validation_step_outputs["epoch_exp_y_means"].numpy()
                epoch_exp_means = self.validation_step_outputs["epoch_exp_means"].numpy()
                # epoch_exp_rec = self.validation_step_outputs["epoch_exp_rec"].numpy()
                epoch_exp_mix_rec = self.validation_step_outputs["epoch_exp_mix_rec"].numpy()
            else:
                epoch_data_index = gather_nd_to_rank(self.validation_step_outputs["data_index"], rank="all").numpy()
                epoch_exp_pred_labels = gather_nd_to_rank(self.validation_step_outputs["epoch_exp_pred_labels"], rank="all").numpy()
                epoch_exp_prob_cat = gather_nd_to_rank(self.validation_step_outputs["epoch_exp_prob_cat"], rank="all").numpy()
                # epoch_exp_y_means = gather_nd_to_rank(self.validation_step_outputs["epoch_exp_y_means"]).numpy()
                epoch_exp_means = gather_nd_to_rank(self.validation_step_outputs["epoch_exp_means"], axis=1).numpy()
                # epoch_exp_rec = gather_nd_to_rank(self.validation_step_outputs["epoch_exp_rec"], axis=1).numpy()
                epoch_exp_mix_rec = gather_nd_to_rank(self.validation_step_outputs["epoch_exp_mix_rec"], axis=1).numpy()

        # print("epoch_exp_pred_labels at no_grad", epoch_exp_pred_labels.shape)
        if self.max_dynamic_neigh:
            # sorted_epoch_data_index = np.argsort(epoch_data_index) # assume epoch_data_index is non-repeating
            sorted_epoch_data_index = np.unique(epoch_data_index, return_index=True)[1] # assume epoch_data_index is full (containing 0th to the last index)
            epoch_exp_pred_labels = epoch_exp_pred_labels[sorted_epoch_data_index]
            # print("epoch_exp_pred_labels after sorting", epoch_exp_pred_labels.shape)
            epoch_exp_prob_cat = epoch_exp_prob_cat[sorted_epoch_data_index]
            if self.trainer.is_global_zero:
                epoch_exp_means = epoch_exp_means[:, sorted_epoch_data_index]
                # epoch_exp_rec = epoch_exp_rec[:, sorted_epoch_data_index] # huge memory consumption for large datasets
                epoch_exp_mix_rec = epoch_exp_mix_rec[:, sorted_epoch_data_index]
        if self.gaussian_start_epoch_pct > 0.:
            if self.current_epoch == (self.gaussian_start_epoch_pct * self.trainer.max_epochs - 1):
                if self.trainer.is_global_zero:
                    epoch_exp_prob_cat_pseudo = np.zeros_like(epoch_exp_prob_cat)
                    epoch_exp_prob_cat_pseudo[np.arange(len(epoch_exp_prob_cat)), epoch_exp_pred_labels] = 1
                    # epoch_exp_prob_cat_means = np.take_along_axis(epoch_exp_means, epoch_exp_pred_labels[None, :, None], 0).squeeze(0)
                    epoch_exp_embed = np.multiply(epoch_exp_means, np.expand_dims(epoch_exp_prob_cat.T, -1))
                    epoch_exp_prob_cat_means = epoch_exp_embed.sum(0)
                    # epoch_exp_mix_rec = np.multiply(epoch_exp_rec, np.expand_dims(epoch_exp_prob_cat.T, -1))
                    epoch_exp_prob_cat_rec = epoch_exp_mix_rec.sum(0)

                    # epoch_exp_prob_cat_pseudo = torch.empty_like(epoch_exp_prob_cat)
                    # epoch_exp_prob_cat_pseudo[torch.arange(len(epoch_exp_prob_cat)), epoch_exp_pred_labels] = 1
                    # epoch_exp_embed = torch.multiply(epoch_exp_means, torch.unsqueeze(epoch_exp_prob_cat.T, -1))
                    # epoch_exp_prob_cat_means = epoch_exp_embed.sum(0)
                    # epoch_exp_mix_rec = torch.multiply(epoch_exp_rec, torch.unsqueeze(epoch_exp_prob_cat.T, -1))
                    # epoch_exp_prob_cat_rec = epoch_exp_mix_rec.sum(0)
                if self.prior_generator.endswith("mclust") or self.prior_generator.endswith("mclust_rec"):
                    C, B, D = epoch_exp_means.shape
                    gt_center = np.empty((C, D))
                    sorted_pseudo_center = np.empty((C, D))
                    original_pseudo_center = np.empty((C, D))
                    if self.trainer.is_global_zero:
                        for j in range(C):
                            # gt_center[j] = np.mean(epoch_exp_means[j, self.trainer.datamodule.data_val.adata.obs[f"{self.trainer.datamodule.data_val.annotation_key}_int"] == j], axis=0)
                            if self.is_labelled:
                                gt_center[j] = np.mean(epoch_exp_prob_cat_means[self.trainer.datamodule.data_val.adata.obs[f"{self.trainer.datamodule.data_val.annotation_key}_int"] == j], axis=0) # the same as above
                            # original_pseudo_center[j] = np.mean(epoch_exp_prob_cat_means[epoch_exp_pred_labels == j], axis=0)
                            original_pseudo_center[j] = np.average(epoch_exp_embed[j], axis=0, weights=epoch_exp_prob_cat[:, j])
                        # original_pseudo_center_cp = original_pseudo_center.copy()

                        rpy2.robjects.packages.quiet_require("mclust")
                        # robjects.r.library("mclust")
                        rpy2.robjects.numpy2ri.activate()
                        r_random_seed = robjects.r['set.seed']
                        r_random_seed(self.seed)
                        rmclust_options = robjects.r['mclust.options']
                        init_subset_num = 10000
                        while True:
                            # rmclust_options(hcModelName=hc_model_name, subset=10000)
                            rmclust_options(subset=init_subset_num)
                            rmclust = robjects.r['Mclust']
                            if self.prior_generator.endswith("rec"):
                                mclust_pca = PCA(n_components=20, random_state=self.seed)
                                mclust_pca_embedding = mclust_pca.fit_transform(epoch_exp_prob_cat_rec.astype(np.float32))
                                rmclust_res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(mclust_pca_embedding), self.num_classes, GMM_model.EEE.name, verbose=False)
                            else:
                                rmclust_res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(epoch_exp_prob_cat_means.astype(np.float32)), self.num_classes, GMM_model.EEE.name, verbose=False)
                            mclust_prob_cat = rmclust_res[-3]
                            if len(np.unique(mclust_prob_cat.argmax(1))) == self.num_classes:
                                break
                            else:
                                if init_subset_num >= len(epoch_exp_prob_cat_means):
                                    print("Warning: mclust failed to obtain enough clusters.")
                                    break
                                init_subset_num *= 2

                        # (B x D, B x C) -> (B x 1 x D, B x C x 1) -> (B x C x D) -> (C x D)
                        pseudo_center = np.multiply(np.expand_dims(epoch_exp_prob_cat_means, 1), np.expand_dims(mclust_prob_cat, -1)).sum(0) / np.expand_dims(mclust_prob_cat.sum(0), -1)
                        # Find the closest pseudo, ground truth center pair and remove it until each pseudo has a corresponding ground truth center
                        remaining_pseudo = np.arange(C)
                        remaining_gt = np.arange(C)
                        if "mclust" in self.prior_generator:
                            sorted_mclust_prob_cat = np.empty_like(mclust_prob_cat)
                        for i in range(C):
                            # Compute the distance matrix between the pseudo centers and ground truth centers
                            dist_matrix = cdist(pseudo_center, original_pseudo_center, metric="euclidean")

                            # Find the indices of the closest pseudo, ground truth center pair
                            pseudo_idx, gt_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

                            # Update the sorted_gt_center variable with the matching ground truth center
                            # sorted_pseudo_center[remaining_pseudo[pseudo_idx]] = original_pseudo_center[gt_idx] # this seems wrong
                            # remaining_pseudo = np.delete(remaining_pseudo, pseudo_idx)
                            sorted_pseudo_center[remaining_gt[gt_idx]] = pseudo_center[pseudo_idx]
                            if "mclust" in self.prior_generator:
                                sorted_mclust_prob_cat[:, remaining_gt[gt_idx]] = mclust_prob_cat[:, remaining_pseudo[pseudo_idx]]
                            remaining_pseudo = np.delete(remaining_pseudo, pseudo_idx)
                            remaining_gt = np.delete(remaining_gt, gt_idx)
                            # Remove the closest pair
                            pseudo_center = np.delete(pseudo_center, pseudo_idx, axis=0)
                            original_pseudo_center = np.delete(original_pseudo_center, gt_idx, axis=0)
                    if self.prior_generator.startswith("tensor"):
                        if self.trainer.is_global_zero:
                            mu_prior = torch.from_numpy(sorted_pseudo_center.astype(np.float32)).to(self.data_device)
                        else:
                            mu_prior = torch.empty_like(sorted_pseudo_center.astype(np.float32)).to(self.data_device)
                        if self.trainer.world_size != 1:
                            self.trainer.strategy.barrier()
                            broadcast(mu_prior, 0)
                        # self.EXPGMGAT.decoder.mu_prior = nn.Parameter(torch.from_numpy(sorted_pseudo_center.astype(np.float32)).to(self.data_device), requires_grad=False)
                        self.EXPGMGAT.decoder.mu_prior = nn.Parameter(mu_prior, requires_grad=False)
                        # self.EXPGMGAT.decoder.mu_prior = nn.Parameter(torch.from_numpy(sorted_pseudo_center.astype(np.float32)).to(self.data_device))
                    # print("replace mu_prior with sorted_gt_center")
                    if self.trainer.is_global_zero:
                        if self.prior_generator.endswith("mclust") or self.prior_generator.endswith("mclust_rec"):
                            # self.valid_cluster_init_labels = torch.tensor(sorted_mclust_prob_cat.argmax(1), dtype=torch.long, device=self.device)
                            self.valid_cluster_init_labels = torch.tensor(cdist(epoch_exp_prob_cat_means, sorted_pseudo_center).argmin(1), dtype=torch.long, device=self.device)
                        else:
                            self.valid_cluster_init_labels = torch.tensor(cdist(epoch_exp_prob_cat_means, sorted_pseudo_center).argmin(1), dtype=torch.long, device=self.device)
                    else:
                        self.valid_cluster_init_labels = torch.empty(B, dtype=torch.long, device=self.device)
                    if self.trainer.world_size != 1:
                        self.trainer.strategy.barrier()
                        broadcast(self.valid_cluster_init_labels, 0)
                    # print("mclust", np.unique(sorted_mclust_prob_cat.argmax(1), return_counts=True))
                    # print("old new centroids dist", cdist(sorted_pseudo_center, original_pseudo_center_cp, metric="euclidean"))
                    # print("sorted_pseudo_center", sorted_pseudo_center)
                    print("replace mu_prior with fixed sorted_pseudo_center")

        if self.trainer.is_global_zero:
            if self.rec_neigh_num:
                if self.max_dynamic_neigh:
                    if self.dynamic_neigh_level == Dynamic_neigh_level.domain:
                        if len(np.unique(self.trainer.datamodule.data_val.dynamic_neigh_nums)) == 1:
                            neigh_consistency_sum_score = 0.
                            # print("epoch_exp_pred_labels.shape", epoch_exp_pred_labels.shape)
                            # print("self.trainer.datamodule.data_val.adata.obsm['sp_k'].shape", self.trainer.datamodule.data_val.adata.obsm['sp_k'].shape)
                            # print("self.trainer.datamodule.data_val.dynamic_neigh_nums[0]", self.trainer.datamodule.data_val.dynamic_neigh_nums[0])
                            # print("epoch_exp_pred_labels[self.trainer.datamodule.data_val.adata.obsm['sp_k'][:, :self.trainer.datamodule.data_val.dynamic_neigh_nums[0]]].shape", epoch_exp_pred_labels[self.trainer.datamodule.data_val.adata.obsm['sp_k'][:, :self.trainer.datamodule.data_val.dynamic_neigh_nums[0]]].shape)
                            neigh_consistency_score = np.mean(epoch_exp_pred_labels[self.trainer.datamodule.data_val.adata.obsm['sp_k'][:, :self.trainer.datamodule.data_val.dynamic_neigh_nums[0]]] == np.expand_dims(epoch_exp_pred_labels, -1))
                        else:
                            neigh_consistency_score = []
                            neigh_consistency_sum_score = []
                            neigh_consistency_sum_score_factor = 0.
                            for j, dynamic_neigh_num in enumerate(self.trainer.datamodule.data_val.dynamic_neigh_nums):
                                this_cluster_node_idx = self.trainer.datamodule.data_val.adata.obs["pred_labels"] == j
                                score_neighbors = self.trainer.datamodule.data_val.adata.obsm['sp_k'][this_cluster_node_idx, :dynamic_neigh_num]
                                score_neighbors_label = epoch_exp_pred_labels[score_neighbors]
                                is_score_neighbors_consistency = score_neighbors_label == np.expand_dims(epoch_exp_pred_labels[this_cluster_node_idx], -1)
                                # self.validation_step_debug_outputs["score_neighbors"] = score_neighbors
                                # self.validation_step_debug_outputs["score_neighbors_label"] = score_neighbors_label
                                # print(j, dynamic_neigh_num, score_neighbors.shape, is_score_neighbors_consistency.shape)
                                neigh_consistency_score.append(np.mean(is_score_neighbors_consistency, axis=1))
                                neigh_consistency_sum_score.append(np.sum(is_score_neighbors_consistency))
                                neigh_consistency_sum_score_factor += this_cluster_node_idx.sum() * dynamic_neigh_num
                            neigh_consistency_score = np.mean(np.concatenate(neigh_consistency_score))
                            neigh_consistency_sum_score = np.sum(neigh_consistency_sum_score) / neigh_consistency_sum_score_factor
                else:
                    epoch_exp_neigh_pred_labels = np.concatenate(self.validation_step_outputs["epoch_exp_neigh_pred_labels"])
                    neigh_consistency_score = np.mean(np.expand_dims(epoch_exp_pred_labels, -1) == epoch_exp_neigh_pred_labels)
            # epoch_gt_labels = np.concatenate(epoch_gt_labels)
            # epoch_exp_att_x = np.concatenate(self.validation_step_outputs["epoch_exp_att_x"])
            # print("epoch_exp_latent", epoch_exp_latent[0].shape, epoch_exp_latent[-1].shape)
            # epoch_exp_latent = np.concatenate(self.validation_step_outputs["epoch_exp_latent"], axis=1)
            # epoch_exp_means = np.concatenate(self.validation_step_outputs["epoch_exp_means"], axis=1)
            # epoch_exp_logvars = np.concatenate(self.validation_step_outputs["epoch_exp_logvars"], axis=1)
            # epoch_exp_vis_bi_adj = np.concatenate(epoch_exp_vis_bi_adj)
            # epoch_sp_vis_bi_adj = np.concatenate(epoch_sp_vis_bi_adj)
            # epoch_sp_vis_bi_adj = self.trainer.datamodule.data_val.adata.obsm['sp_adj']
            # epoch_exp_y_means = np.concatenate(self.validation_step_outputs["epoch_exp_y_means"])
            # epoch_exp_y_logvars = np.concatenate(self.validation_step_outputs["epoch_exp_y_logvars"])
        if self.device.type == "cpu":
            self.validation_step_keep_outputs["epoch_exp_prob_cat"] = epoch_exp_prob_cat
            self.validation_step_keep_outputs["epoch_exp_pred_labels"] = epoch_exp_pred_labels
            self.validation_step_keep_outputs["epoch_exp_y_means"] = epoch_exp_y_means
            self.validation_step_keep_outputs["epoch_exp_means"] = epoch_exp_means
        # if self.last_epoch_exp_pred_labels is not None:
        #     pred_diff_pct = (epoch_exp_pred_labels != self.last_epoch_exp_pred_labels).mean()
        #     self.log("test/pred_diff_pct", pred_diff_pct, on_step=False, on_epoch=True, prog_bar=False)
        #     if pred_diff_pct < 0.01:
        #         self.pred_diff_patience += 1.
        #     else:
        #         self.pred_diff_patience = max(0., self.pred_diff_patience - self.patience)
        #         # self.early_stopping_patience += 1
        # else:
        #     self.log("test/pred_diff_pct", 0., on_step=False, on_epoch=True, prog_bar=False)
        # self.log("test/pred_diff_patience", self.pred_diff_patience, on_step=False, on_epoch=True, prog_bar=False)
        # self.last_epoch_exp_pred_labels = epoch_exp_pred_labels

        # self.trainer.datamodule.data_val.adata.obs.iloc[epoch_data_index, self.trainer.datamodule.data_val.adata.obs.columns.get_loc("pred_labels")] = epoch_exp_pred_labels
        # self.trainer.datamodule.data_val.adata.obsm['prob_cat'] = epoch_exp_prob_cat[np.unique(epoch_data_index, return_index=True)[1]]
        if self.trainer.datamodule.data_val.annotation_key is not None:
            epoch_gt_labels = self.trainer.datamodule.data_val.adata.obs[f'{self.trainer.datamodule.data_val.annotation_key}_int']
        self.trainer.datamodule.data_val.adata.obs["pred_labels"] = epoch_exp_pred_labels
        self.trainer.datamodule.data_val.adata.obsm['prob_cat'] = epoch_exp_prob_cat
        if self.trainer.datamodule.data_type == "custom":
            self.trainer.datamodule.data_val.adata.obs["pred_labels"].to_csv(osp.join(self.log_path, f"pred_labels_epoch_{self.current_epoch}.csv"))
        # if self.trainer.datamodule.data_val.full_adata is not None:
        #     epoch_gt_labels = self.trainer.datamodule.data_val.full_adata.obs[f'{self.trainer.datamodule.data_val.annotation_key}_int'].to_numpy()
            # epoch_exp_pred_labels = self.trainer.datamodule.data_val.full_adata.obs["pred_labels"].to_numpy()
        # print('gt', np.unique(gt_labels, return_counts=True))

        if self.is_labelled:
            # exp_ARI = adjusted_rand_score(epoch_gt_labels[self.trainer.datamodule.data_val.adata.obs['is_labelled']], epoch_exp_pred_labels[self.trainer.datamodule.data_val.adata.obs['is_labelled']])
            exp_ARI = adjusted_rand_score(epoch_gt_labels[self.trainer.datamodule.data_val.adata.obs['is_labelled']], self.trainer.datamodule.data_val.adata.obs["pred_labels"].loc[self.trainer.datamodule.data_val.adata.obs['is_labelled']])
            self.last_epoch_ARI = exp_ARI
            self.highest_exp_ARI = max(self.highest_exp_ARI, exp_ARI)
            self.log("test/exp_ARI", exp_ARI, on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.sync_dist) # may be used for early stopping
        if self.trainer.is_global_zero:
            predicted_domains = np.unique(epoch_exp_pred_labels)
            if len(predicted_domains) < self.num_classes:
                self.not_enough_classes = True
                self.force_add_prob_cat_domain = list(set(range(self.num_classes)) - set(predicted_domains.tolist()))
                self.prob_cat_to_add = torch.zeros((1, self.num_classes), dtype=torch.float32, device=self.device)
                self.prob_cat_to_add[0, self.force_add_prob_cat_domain] = 1. / self.num_classes
            else:
                self.not_enough_classes = False
                self.force_add_prob_cat_domain = None
            # logging graph for visualization.
            # if self.log_path.startswith("/mnt/f/") or self.log_path.startswith("/home/kali/"):
            # print("exp_ARI - self.highest_exp_ARI:", exp_ARI - self.highest_exp_ARI)
            # print("exp_ARI - self.highest_exp_ARI >= 0.001:", exp_ARI - self.highest_exp_ARI >= 0.001)
            # print("self.plot_gt or exp_ARI - self.highest_exp_ARI >= 0.001", self.plot_gt or exp_ARI - self.highest_exp_ARI >= 0.001)
            # if self.plot_gt or exp_ARI - self.highest_exp_ARI >= 0.01:
            # if True:
            # if self.plot_gt or exp_ARI - self.last_epoch_ARI >= 0.01:
            if self.plot_gt:
                # gt_exp_graph_path, pred_exp_graph_path = log_exp_graph(
                #     plot_node_idx=self.plot_node_idx,
                #     exp_vis_bi_adj=epoch_exp_vis_bi_adj,
                #     log_path=self.log_path,
                #     current_epoch=self.current_epoch,
                #     gt_labels=epoch_gt_labels,
                #     pred_labels=epoch_exp_pred_labels,
                #     plot_gt=self.plot_gt
                # )
                # gt_sp_graph_path, pred_sp_graph_path = log_sp_graph(
                #     plot_node_idx=self.plot_node_idx,
                #     sp_bi_adj=epoch_sp_vis_bi_adj,
                #     log_path=self.log_path,
                #     current_epoch=self.current_epoch,
                #     gt_labels=epoch_gt_labels,
                #     pred_labels=epoch_exp_pred_labels, #epoch_sp_pred_labels,
                #     plot_gt=self.plot_gt
                # )

                if self.is_labelled:
                    gt_tissue_graph_path, pred_tissue_graph_path = log_tissue_graph(
                        data_val=self.trainer.datamodule.data_val,
                        log_path=self.log_path,
                        current_epoch=self.current_epoch,
                        gt_labels=epoch_gt_labels,
                        pred_labels=epoch_exp_pred_labels,
                        plot_gt=self.plot_gt,
                    )
                    if isinstance(self.logger, CometLogger):
                        self.logger.experiment.log_image(pred_tissue_graph_path, "test/pred_tissue_graph_path")

            # if self.current_epoch % 100 == 0:
            # if self.current_epoch % 100 == 99:
            # if True:
            if self.is_labelled:
                if self.current_epoch % 10 == 9:
                    gt_tissue_graph_path, pred_tissue_graph_path = log_tissue_graph(
                        data_val=self.trainer.datamodule.data_val,
                        log_path=self.log_path,
                        current_epoch=self.current_epoch,
                        gt_labels=epoch_gt_labels,
                        pred_labels=epoch_exp_pred_labels,
                        plot_gt=self.plot_gt,
                    )
                    if isinstance(self.logger, CometLogger):
                        self.logger.experiment.log_image(pred_tissue_graph_path, "test/pred_tissue_graph_path")

            epoch_exp_pred_label_values, epoch_exp_pred_label_counts = np.unique(epoch_exp_pred_labels, return_counts=True)
            print(epoch_exp_pred_label_values, epoch_exp_pred_label_counts)
            # if self.is_labelled:
            #     if self.rec_neigh_num and (not self.max_dynamic_neigh or (self.max_dynamic_neigh and self.dynamic_neigh_level == Dynamic_neigh_level.domain)):
            #         print("Epoch", self.current_epoch, "exp_ARI:", exp_ARI, "neigh_consistency:", neigh_consistency_score, "curr_time:", datetime.now().strftime("%H:%M:%S"))
            #     else:
            #         print("Epoch", self.current_epoch, "exp_ARI:", exp_ARI, "curr_time:", datetime.now().strftime("%H:%M:%S"))
            #     self.log("test/cluster_num", float(len(epoch_exp_pred_label_values)), on_step=False, on_epoch=True, prog_bar=False)
            #     self.log("test/min_cluster_node_num", float(epoch_exp_pred_label_counts.min()), on_step=False, on_epoch=True, prog_bar=False)
            #     self.log("test/highest_exp_ARI", self.highest_exp_ARI, on_step=False, on_epoch=True, prog_bar=False)
            if self.rec_neigh_num:
                if self.dynamic_neigh_level == Dynamic_neigh_level.domain:
                    if self.max_dynamic_neigh and neigh_consistency_sum_score != 0:
                        print("Epoch", self.current_epoch, "neigh_consist:", neigh_consistency_score, "neigh_consist_sum:", neigh_consistency_sum_score, "curr_time:", datetime.now().strftime("%H:%M:%S"))
                    print("Epoch", self.current_epoch, "neigh_consistency:", neigh_consistency_score, "curr_time:", datetime.now().strftime("%H:%M:%S"))
                elif self.dynamic_neigh_level.name.startswith("unit"):
                    print("Epoch", self.current_epoch, "curr_time:", datetime.now().strftime("%H:%M:%S"))
            else:
                print("Epoch", self.current_epoch, "curr_time:", datetime.now().strftime("%H:%M:%S"))

            # Visualize latent space.
            # result_tsne_figure_path = log_tsne_figure(
            #     labels=labels,
            #     latent=(torch.squeeze(latent) * torch.squeeze(prob_cat.transpose(1, 2)).unsqueeze(-1)).sum(0),
            #     log_path=self.log_path,
            # )

            if self.rec_neigh_num:
                if not self.max_dynamic_neigh or (self.max_dynamic_neigh and self.dynamic_neigh_level == Dynamic_neigh_level.domain):
                    self.log("test/neigh_consistency_score", neigh_consistency_score, on_step=False, on_epoch=True, prog_bar=False)
                    if self.max_dynamic_neigh:
                        if neigh_consistency_sum_score != 0.:
                            self.log("test/neigh_consistency_sum_score", neigh_consistency_sum_score, on_step=False, on_epoch=True, prog_bar=False)
            if self.plot_gt:
                if isinstance(self.logger, CometLogger):
                    # self.logger.experiment.log_image(gt_exp_graph_path, "test/gt_exp_graph_path")
                    # self.logger.experiment.log_image(gt_sp_graph_path, "test/gt_sp_graph_path")
                    self.logger.experiment.log_image(gt_tissue_graph_path, "test/gt_tissue_graph_path")
                self.plot_gt = False
        self.validation_step_outputs.clear()
        if self.debug:
            pass
            # self.validation_step_debug_outputs.clear()
        # print("max_memory_allocated", torch.cuda.max_memory_allocated() / 10**9, "GB")
        # print("max_memory_reserved", torch.cuda.max_memory_reserved() / 10**9, "GB")
        # print(f"stop valid epoch end {self.current_epoch}")
        # print(torch.cuda.memory_summary())

    # def test_step(self, batch, batch_idx):
    #     pass

    # def test_epoch_end(self, test_step_outputs):
    #     pass

    # def get_total_steps(self) -> int:
    #     if self.trainer.max_steps:
    #         return self.trainer.max_steps
    #     return len(self.train_dataloader()) * self.trainer.max_epochs // (self.trainer.accumulate_grad_batches * max(1, self.trainer.num_devices))

    def configure_optimizers(self):
        # print("start configure opt")
        # exp_opt = AdamW(self.EXPGMGAT.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # exp_opt = SGD(self.EXPGMGAT.parameters(), lr=self.lr, momentum=0.9)
        # exp_opt = SGD(self.EXPGMGAT.parameters(), lr=self.lr)
        # exp_ls = CosineAnnealingLR(exp_opt, T_max=100, eta_min=0.000001)
        # exp_ls = CosineAnnealingLR(exp_opt, T_max=self.T_max, eta_min=self.lr * 0.01)
        if self.lr_scheduler == "cosine":
            exp_opt = Adam(self.EXPGMGAT.parameters(), lr=self.lr)
            exp_ls = CosineAnnealingLR(exp_opt, T_max=self.T_max, eta_min=self.lr * 0.01)
            # exp_opt = Adam(self.EXPGMGAT.parameters(), lr=self.lr * 0.1)
            # exp_opt = Adam(self.EXPGMGAT.parameters(), lr=self.lr * 0.01)
            # exp_ls = CosineAnnealingLR(exp_opt, T_max=self.T_max, eta_min=self.lr)
            # if (not self.debug) and (not self.verbose_loss):
            #     assert (self.trainer.max_epochs % 10 == 0) and ((self.trainer.max_epochs // self.T_max) % 2 == 0)
        elif self.lr_scheduler == "onecycle":
            exp_opt = Adam(self.EXPGMGAT.parameters(), lr=self.lr)
            # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer.estimated_stepping_batches
            stepping_batches = self.trainer.estimated_stepping_batches
            exp_ls = OneCycleLR(exp_opt, max_lr=self.lr * 10, total_steps=stepping_batches)
        elif self.lr_scheduler == "cyclic":
            exp_opt = Adam(self.EXPGMGAT.parameters(), lr=self.lr)
            if isinstance(exp_opt, SGD) or isinstance(exp_opt, Adam):
                cycle_momentum = False
            else:
                cycle_momentum = True
                raise
            exp_ls = CyclicLR(exp_opt, self.lr, self.lr * 10, step_size_up=self.trainer.max_epochs // 10, step_size_down=self.trainer.max_epochs // 10, mode="exp_range", cycle_momentum=cycle_momentum, gamma=self.cyclic_gamma)
        # exp_ls1 = CosineAnnealingLR(exp_opt, T_max=self.T_max, eta_min=self.lr * 0.01)
        # exp_ls2 = ConstantLR(exp_opt, factor=0.01, total_iters=self.T_max)
        # exp_ls = SequentialLR(exp_opt, [exp_ls1, exp_ls2], [self.trainer.max_epochs - self.T_max]) # BUG: has no effect using lightning and SequentialLR has PyTorch bug as well
        # print("stop configure opt")
        return (
            {"optimizer": exp_opt, "lr_scheduler": exp_ls},
        )

    # def lr_scheduler_step(self, scheduler, metric):
    #     # if self.current_epoch < self.trainer.max_epochs - self.T_max:
    #     #     scheduler.step(epoch=self.current_epoch)
    #     scheduler.step()

    def normalized_gaussian_kernel(self, dist, axis=-1):
        if self.weighted_neigh:
            return F.softmax(torch.exp(-torch.square(dist) / (2 * np.square(self.weighted_neigh))), axis)
        else:
            return torch.ones_like(dist)

    # https://discuss.pytorch.org/t/batched-bincount/72819/3
    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        target.scatter_add_(dim, x.to(torch.int64), torch.ones_like(x))
        return target

    def get_cluster_consistency(self, mix_prob_cat, j, this_cluster_node_idx, output_nodes_idx, input_nodes, output_nodes, get_incosistency=False, no_log=True):
        data_index = output_nodes.cpu().numpy()
        dynamic_neigh_num = self.trainer.datamodule.data_val.dynamic_neigh_nums[j]
        # dynamic neigh idx in original adata: flatten(this_cluster_node_num x this_cluster_dynamic_neigh_num)
        dynamic_neigh_idx = self.trainer.datamodule.data_val.adata[data_index].obsm['sp_k'][this_cluster_node_idx, :dynamic_neigh_num].flatten()
        # dynamic neigh idx in batch: flatten(this_cluster_node_num x this_cluster_dynamic_neigh_num)
        batch_dynamic_neigh_idx = torch.nonzero(input_nodes == torch.from_numpy(dynamic_neigh_idx).to(input_nodes).unsqueeze(1))[:, 1] #.reshape(-1, dynamic_neigh_num)
        x_cat = mix_prob_cat[output_nodes_idx][this_cluster_node_idx].argmax(-1)
        neigh_cat = mix_prob_cat[batch_dynamic_neigh_idx].argmax(-1).reshape(-1, dynamic_neigh_num)
        # print("x_cat.shape", x_cat.shape)
        # print("neigh_cat.shape", neigh_cat.shape)
        all_cat = torch.cat([x_cat.unsqueeze(1), neigh_cat], dim=1).to(torch.int32)
        return self.get_consistency_common(x_cat, neigh_cat.T, all_cat, get_incosistency=get_incosistency, no_log=no_log)

    def get_consistency(self, prob_cat, neigh_out_net, get_incosistency=False, no_log=False):
        # B x C -> B
        x_cat = prob_cat.argmax(-1)
        # K x B x C
        neigh_prob_cats = torch.stack(neigh_out_net["prob_cat"])
        # K x B x C -> K x B
        neigh_cat = neigh_prob_cats.argmax(-1)
        # B x (K + 1)
        all_cat = torch.cat([x_cat.unsqueeze(1), neigh_cat.T], dim=1).to(torch.int32)
        return self.get_consistency_common(x_cat, neigh_cat, all_cat, get_incosistency=get_incosistency, no_log=no_log)

    def get_consistency_common(self, x_cat, neigh_cat, all_cat, get_incosistency=False, no_log=False):
        # self.time_dict["1_consistency"] = time.time()
        # print("Time for 1_consistency", self.time_dict["1_consistency"] - self.time_dict["fetch_loss_data"])
        # all_cat_count = torch.stack([
        #     torch.bincount(x_i, minlength=self.num_classes) for x_i in torch.unbind(all_cat, dim=0)
        # ], dim=0)
        all_cat_count = self.batched_bincount(all_cat, dim=1, max_value=self.num_classes)
        # self.time_dict["2_consistency"] = time.time()
        # print("Time for 2_consistency", self.time_dict["2_consistency"] - self.time_dict["1_consistency"])
        sorted_all_cat_count, sorted_all_cat_count_idx = torch.sort(all_cat_count, descending=True)
        # self.time_dict["3_consistency"] = time.time()
        # print("Time for 3_consistency", self.time_dict["3_consistency"] - self.time_dict["2_consistency"])
        # whether pseudo labels for each local area (x and its rec neigh) are consistent
        whether_consistent = sorted_all_cat_count[:, 0] > sorted_all_cat_count[:, 1] * 2
        whether_consistent_mean = whether_consistent.float().mean()
        # whether pseudo label of the target unit is the same as the consistent pseudo label (if local area has one)
        x_consistent_idx = whether_consistent * (x_cat == sorted_all_cat_count_idx[:, 0])
        x_consistent_weight = x_consistent_idx.float()
        if not no_log:
            if not get_incosistency:
                self.log("train/whether_consistent_mean", whether_consistent_mean, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
            else:
                self.log("train/whether_consistent_mean_after_semi", whether_consistent_mean, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.curr_batch_size, sync_dist=self.sync_dist)
        # self.time_dict["4_consistency"] = time.time()
        # print("Time for 4_consistency", self.time_dict["4_consistency"] - self.time_dict["3_consistency"])
        # self.time_dict["before_consistency"] = time.time()
        # print("Time for before consistency: ", self.time_dict["before_consistency"] - self.time_dict["fetch_loss_data"])
        if self.is_well_initialized:
            # K x B
            # whether pseudo labels of the rec neigh are the same as the consistent pseudo label (if local area has one)
            neigh_consistent_idx = whether_consistent.unsqueeze(0) * (neigh_cat == sorted_all_cat_count_idx[:, 0].unsqueeze(0))
            neigh_consistent_weight = torch.nan_to_num(neigh_consistent_idx / neigh_consistent_idx.sum(0, keepdim=True))
        else:
            neigh_consistent_idx, neigh_consistent_weight = None, None
        # self.time_dict["after_consistency"] = time.time()
        # print("Time for after consistency: ", self.time_dict["after_consistency"] - self.time_dict["before_consistency"])
        if get_incosistency:
            # whether local area has consistent pseudo label but the target unit does not
            x_inconsistent_idx = whether_consistent * (x_cat != sorted_all_cat_count_idx[:, 0])
            # pesudo_labels = F.one_hot(sorted_all_cat_count_idx[:, 0][x_inconsistent_idx].detach().cpu(), num_classes=self.num_classes).float().to(prob_cat.device)
            pesudo_labels = sorted_all_cat_count_idx[x_inconsistent_idx, 0]
            return whether_consistent, x_consistent_idx, x_consistent_weight, neigh_consistent_idx, neigh_consistent_weight, x_inconsistent_idx, pesudo_labels
        return whether_consistent, x_consistent_idx, x_consistent_weight, neigh_consistent_idx, neigh_consistent_weight

    def semi_sup_loss(self, out_net, neigh_out_net=None):
        assert self.add_cat_bias is False
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        whether_consistent, x_consistent_idx, x_consistent_weight, neigh_consistent_idx, neigh_consistent_weight, x_inconsistent_idx, pesudo_labels = self.get_consistency(prob_cat, neigh_out_net, get_incosistency=True)
        # if True:
        #     print("prob_cat[x_inconsistent_idx].shape", prob_cat[x_inconsistent_idx].shape)
        #     print("pesudo_labels.shape", pesudo_labels.shape)
        if self.is_well_initialized:
            # semi_super_loss = CrossEntropyLoss()(prob_cat[x_inconsistent_idx], pesudo_labels)
            semi_super_loss = CrossEntropyLoss()(logits[x_inconsistent_idx], pesudo_labels)
        return semi_super_loss

    def cluster_init_loss(self, out_net, cluster_labels=None, output_nodes_idx=None, degree="just"):
        if output_nodes_idx is not None:
            logits, prob_cat = out_net['logits'][output_nodes_idx], out_net['prob_cat'][output_nodes_idx]
        else:
            logits, prob_cat = out_net['logits'], out_net['prob_cat']

        if degree == "always":
            cluster_init_loss = CrossEntropyLoss(reduction="sum")(logits, cluster_labels)
        else:
            if len(cluster_labels.shape) == 1:
                if degree == "just":
                    need_init_idx = cluster_labels != prob_cat.argmax(1) # we do not want to make predicted very close to 1 but just a little bit larger than the second largest
                else:
                    prob_cat_sorted, prob_cat_sorted_idx = torch.sort(prob_cat, descending=True)
                    need_init_idx = torch.logical_or(cluster_labels != prob_cat_sorted[:, 0], prob_cat_sorted[:, 0] < float(degree) * prob_cat_sorted[:, 1])
            elif len(cluster_labels.shape) == 2:
                if degree == "just":
                    need_init_idx = cluster_labels.argmax(1) != prob_cat.argmax(1)
                else:
                    raise NotImplementedError
            cluster_init_loss = CrossEntropyLoss(reduction="sum")(logits[need_init_idx], cluster_labels[need_init_idx])
        return cluster_init_loss