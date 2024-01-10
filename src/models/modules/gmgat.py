import torch
import torch.nn as nn
import torch.nn.init as init
from src.models.modules.modules import (
    Encoder,
    Decoder,
)

class VAE(nn.Module):
    """VAE module, main architecture of GMGAT.

    Args:
        encoder_in_channels (list): list of input dimension for encoder each block.
        encoder_out_channels (list): list of output dimension for encoder each block.
        num_heads (int): number of heads.
        y_dim (int): number of gaussain mixture distributions.
        latent_dim (int): dimension of latent embedding (mu, sigma).
        dropout (int): dropout ratio.
        use_bias (boolean): whether to use bias in attention block.
        block_type (string): block type in encoder, support GCN and Transformer.

    """
    def __init__(
        self,
        encoder_in_channels=[],
        encoder_out_channels=[],
        decoder_in_channels=[],
        decoder_out_channels=[],
        attention_dim=128,
        num_heads=16,
        c_dim=2048,
        y_dim=30,
        latent_dim=512,
        dropout=0.,
        use_bias=True,
        y_block_type="Dense",
        z_block_type="Dense",
        rec_type=None,
        activation=nn.ReLU(),
        max_mu=10.,
        max_logvar=5.,
        min_logvar=-5.,
        forward_neigh_num=False,
        use_batch_norm=False,
        add_cat_bias=False,
        prior_generator="fc",
        GMM_model_name="VVI",
        use_kl_bn=False,
        kl_bn_gamma=32.,
        adata=None,
        device=torch.device("cpu"),
        legacy=False,
        debug=False,
    ):
        super(VAE, self).__init__()
        self.encoder_in_channels = encoder_in_channels
        self.encoder_out_channels = encoder_out_channels
        self.decoder_in_channels = decoder_in_channels
        self.decoder_out_channels = decoder_out_channels
        self.num_heads = num_heads
        self.c_dim = c_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.y_block_type = y_block_type
        self.z_block_type = z_block_type
        self.rec_type = rec_type
        self.activation = activation
        self.max_mu = max_mu
        self.max_logvar = max_logvar
        self.min_logvar = min_logvar
        self.forward_neigh_num = forward_neigh_num
        self.use_batch_norm = use_batch_norm
        self.add_cat_bias = add_cat_bias
        self.prior_generator = prior_generator
        self.GMM_model_name = GMM_model_name
        self.use_kl_bn = use_kl_bn
        self.kl_bn_gamma = kl_bn_gamma
        self.adata = adata
        self.device = device
        self.legacy = legacy
        self.debug = debug

        assert len(self.encoder_in_channels) == len(self.encoder_out_channels)
        self.encoder = Encoder(
            y_dim=self.y_dim,
            z_dim=self.latent_dim,
            attention_dim=self.attention_dim,
            in_channels=self.encoder_in_channels,
            out_channels=self.encoder_out_channels,
            y_block_type=self.y_block_type,
            z_block_type=self.z_block_type,
            use_bias=self.use_bias,
            num_heads=self.num_heads,
            activation=self.activation,
            max_mu=self.max_mu,
            max_logvar=self.max_logvar,
            min_logvar=self.min_logvar,
            forward_neigh_num=self.forward_neigh_num,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
            add_cat_bias=self.add_cat_bias,
            GMM_model_name=self.GMM_model_name,
            use_kl_bn=self.use_kl_bn,
            kl_bn_gamma=self.kl_bn_gamma,
            adata=self.adata,
            device=self.device,
            legacy=self.legacy,
        )

        self.decoder = Decoder(
            y_dim=self.y_dim,
            z_dim=self.latent_dim,
            in_channels=self.decoder_in_channels,
            out_channels=self.decoder_out_channels,
            rec_type=self.rec_type,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            prior_generator=self.prior_generator,
            g_dim=self.decoder_out_channels[-1],
            GMM_model_name=self.GMM_model_name,
            device=self.device,
            legacy=self.legacy,
        )

        self.weight_initialization()

    def weight_initialization(self):
        # raise
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                # torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None and m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def encode(self, x, neighbor_x):
        y, mus, logvars, zs, var_invs, logits, prob_cat, att_x = self.encoder(x, neighbor_x)
        return mus, logvars, y, zs, var_invs, logits, prob_cat, att_x

    def decode(self, y, zs):
        y_mus, y_logvars, reconstructed = self.decoder(y, zs)
        return y_mus, y_logvars, reconstructed

    # @profile
    def forward(self, x, neighbor_x=None):
        """Forward function of VAE model, which takes the batch item,
        a dictionary object as input, including graph attributes and
        adjacency matrix and mask matrix.

        Args:
            "means": C x B x D;
            "y_means": C x D
            "prob_cat": B x C
            "reconstructed": C x B x D
        """
        y_var_invs = None
        mus, logvars, y, zs, var_invs, logits, prob_cat, att_x = self.encode(
            x=x,
            neighbor_x=neighbor_x,
        )
        y_mus, y_logvars, reconstructed = self.decode(y, zs)
        if self.GMM_model_name == "EEE":
            # C x D x D -> C x 1 x D^2
            logvars = logvars.reshape(logvars.shape[0], 1, -1)
            # C x 1 x D^2 -> C x D^2
            y_logvars = logvars.squeeze(1)
            # C x 1 x D^2 -> C x B x D^2
            logvars = torch.broadcast_to(logvars, (logvars.shape[0], mus.shape[1], logvars.shape[2]))
            y_var_invs = var_invs.reshape(logvars.shape[0], 1, -1).squeeze(1)
            # print("mus", mus.shape)
            # print("logvars", logvars.shape)
            # print("y_mus", y_mus.shape)
            # print("y_logvars", y_logvars.shape)
        output_dict = {
            "means": mus,
            "logvars": logvars,
            "gaussians": zs,
            "logits": logits,
            "prob_cat": prob_cat,
            "y_means": y_mus,
            "y_logvars": y_logvars,
            "y_var_invs": y_var_invs,
            "reconstructed": reconstructed,
        }
        if self.debug:
            output_dict['att_x'] = att_x
        return output_dict
