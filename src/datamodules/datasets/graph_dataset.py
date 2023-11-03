import logging
import scipy
import numpy as np
from torch.utils.data import Dataset
from src.utils.util import *
import scanpy as sc
import pandas as pd
from os import path as osp
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import time
import torch
import gc
from src.utils.states import Dynamic_neigh_level

class GraphDataset(Dataset):
    """Graph dataset object, loading dataset in a batch-wise manner.

    Args:
        k (int): k nearest neighbor used in neighbors.
        rec_neigh_num (boolean): whether to use the reconstructing
            neighbors strategy.

    """
    def __init__(
        self,
        data_dir: str = "",
        dataset_dir: str = "",
        data_type: str = "10x",
        in_type: str = "raw",
        out_type: str = "raw",
        compared_type: str = "raw",
        count_key=None,
        num_classes="auto",
        data_file_name=None,
        num_hvg: int = 2048,
        lib_norm=True,
        n_pc: int = 50,
        max_dynamic_neigh=False,
        dynamic_neigh_level=Dynamic_neigh_level.unit,
        unit_fix_num=None,
        unit_dynamic_num=None,
        k: int = 18,
        resolution="bulk",
        rec_neigh_num=1,
        rec_mask_neigh_threshold=None,
        use_cell_mask=False,
        keep_tiles=False,
        device="cpu",
        seed: int = 42,
        test_with_gt_sp: bool = False,
        forward_neigh_num=0,
        exchange_forward_neighbor_order=False,
        sample_id: str = "sample_id",
        z_scale: float = 2.,
        weighted_neigh=False,
        supervise_cat=False,
        n_jobs=-1,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.data_type = data_type
        self.in_type = in_type
        self.out_type = out_type
        self.compared_type = compared_type
        self.count_key = count_key
        self.annotation_key = None
        self.num_classes = num_classes
        self.data_file_name = data_file_name
        self.num_hvg = num_hvg
        self.lib_norm = lib_norm
        self.n_pc = n_pc
        self.max_dynamic_neigh = max_dynamic_neigh
        self.dynamic_neigh_level = dynamic_neigh_level
        self.unit_fix_num = unit_fix_num
        self.unit_dynamic_num = unit_dynamic_num
        self.k = k
        self.rec_neigh_num = rec_neigh_num
        if self.rec_neigh_num == 'half':
            self.rec_neigh_num = self.k // 2
        self.rec_mask_neigh_threshold = rec_mask_neigh_threshold
        self.use_cell_mask = use_cell_mask
        self.keep_tiles = keep_tiles
        self.supervise_cat = supervise_cat
        self.device = device
        self.seed = seed
        self.test_with_gt_sp = test_with_gt_sp
        self.forward_neigh_num = forward_neigh_num
        if self.forward_neigh_num == 'half':
            self.forward_neigh_num = self.k // 2
        self.exchange_forward_neighbor_order = exchange_forward_neighbor_order
        self.weighted_neigh = weighted_neigh
        self.rng = np.random.default_rng(seed)
        self.hexagonal_num_list = [n * (n + 1) * 3 for n in range(1, 100)]
        self.square_num_list = [n * (n + 1) * 4 for n in range(1, 100)]
        self.n_jobs = n_jobs
        self.resolution = resolution
        self.sample_id = str(sample_id)
        self.z_scale = z_scale
        self.full_adata = None

        if self.data_type == "custom":
            self.get_custom_data()
        self.compute_nearest()
        self.compute_out_log_sigma()
        if self.rec_mask_neigh_threshold:
            self.get_rec_mask()
        if self.forward_neigh_num:
            self.get_item_rng = np.random.default_rng(seed)
            if self.rec_neigh_num:
                # forward_neigh_order 1 x K2
                # rec_forward_neigh_order K1 x K2
                # (K1 + 1) x K2
                self.rng_input = np.broadcast_to(np.expand_dims(np.arange(self.forward_neigh_num), 0), (self.rec_neigh_num + 1, self.forward_neigh_num))
            else:
                # forward_neigh_order 1 x K2
                self.rng_input = np.expand_dims(np.arange(self.forward_neigh_num), 0)
            self.torch_rng_input = torch.from_numpy(self.rng_input.copy()).to(torch.long)

        gc.collect()

    def __getitem__(self, index: int):
        if not self.max_dynamic_neigh:
            # print("getitem begin")
            exp_feature, exp_rec_feature = get_io_feature(self.adata, self.in_type, self.out_type, self.count_key)

            ret_dict = {
                'index': index,
                'exp_feature': exp_feature[index],
                'exp_rec_feature': exp_rec_feature[index],
            }
            if self.supervise_cat == "random":
                ret_dict['cat'] = self.adata[index].obs['random_label'].to_numpy().squeeze(0)
            elif self.supervise_cat is True:
                ret_dict['cat'] = self.adata[index].obs[f"{self.annotation_key}_int"].to_numpy().squeeze(0)
            if self.rec_mask_neigh_threshold:
                ret_dict["exp_rec_mask"] = self.adata.layers["rec_mask"][index]
            if self.rec_neigh_num or self.forward_neigh_num:
                if self.forward_neigh_num:
                    if self.exchange_forward_neighbor_order:
                        sampled_order = np.apply_along_axis(self.get_item_rng.permutation, axis=1, arr=self.rng_input)
                        torch_sampled_order = torch.from_numpy(sampled_order).to(torch.long)
                    else:
                        sampled_order = self.rng_input
                        torch_sampled_order = self.torch_rng_input
                    forward_neigh_order = sampled_order[0]
                    rec_forward_neigh_order = sampled_order[1:]
                    torch_forward_neigh_order = torch_sampled_order[0]
                    torch_rec_forward_neigh_order = torch_sampled_order[1:]
                # N x D
                ref_attr = exp_feature
                # K1 x D
                if self.rec_neigh_num:
                    rec_neigh_attr = exp_feature[self.rec_neigh_idx[index], :]
                # K2 x D
                if self.forward_neigh_num:
                    forward_neigh_attr = exp_feature[self.forward_neigh_idx[index], :]
                # K2 x 2
                # sp_forward_neigh_attr = self.adata.obsm['spatial'][self.forward_neigh_idx[index], :]
                # K1
                if self.rec_neigh_num:
                    if self.weighted_neigh:
                        neigh_affi = self.normalized_gaussian_kernel(np.sqrt(np.square(rec_neigh_attr - ref_attr[[index]]).sum(-1)))
                        ret_dict['exp_rec_neigh_affi'] = neigh_affi

                if self.rec_neigh_num:
                    ret_dict['exp_rec_neigh_feature'] = rec_neigh_attr
                    if self.supervise_cat == "random":
                        ret_dict['rec_cat'] = self.adata[self.rec_neigh_idx[index].flatten()].obs['random_label'].to_numpy()
                    elif self.supervise_cat:
                        ret_dict['rec_cat'] = self.adata[self.rec_neigh_idx[index].flatten()].obs[f"{self.annotation_key}_int"].to_numpy()
                if self.forward_neigh_num:
                    ret_dict['exp_forward_neigh_feature'] = forward_neigh_attr[forward_neigh_order]
                if self.rec_neigh_num and self.forward_neigh_num:
                    # K1 x K2 x D
                    rec_forward_neigh_attr = exp_feature[self.rec_forward_neigh_idx[index], :]
                    # K1 x K2 x 2
                    # sp_rec_forward_neigh_attr = self.adata.obsm['spatial'][self.rec_forward_neigh_idx[index], :]
                    # K x K
                    # if self.weighted_neigh:
                    #     rec_neigh_affi = self.normalized_gaussian_kernel(np.sqrt(np.square(rec_forward_neigh_attr - np.expand_dims(forward_neigh_attr, 1)).sum(-1)))
                    # else:
                    #     rec_neigh_affi = np.ones(self.rec_neigh_num, self.rec_neigh_num)
                    ret_dict['exp_rec_forward_neigh_feature'] = np.take_along_axis(rec_forward_neigh_attr, indices=rec_forward_neigh_order[:, :, np.newaxis], axis=1)
            return ret_dict

    def __len__(self):
        return len(self.adata)

    def compute_out_log_sigma(self):
        if self.out_type == "raw":
            self.out_log_sigma = np.log(self.adata.layers[self.count_key].std(0))
        elif self.out_type == "unscaled":
            self.out_log_sigma = np.log(self.adata.layers['unscaled'].std(0))
        elif self.out_type == "scaled":
            self.out_log_sigma = np.log(self.adata.layers['scaled'].std(0))
        elif self.out_type.startswith("pca"):
            self.out_log_sigma = np.log(self.adata.obsm['X_pca'].std(0))
        # assert np.isfinite(self.out_log_sigma).all()

    def normalized_gaussian_kernel(self, dist, axis=-1):
        if self.weighted_neigh:
            return scipy.special.softmax(np.exp(-np.square(dist) / (2 * np.square(self.weighted_neigh))), axis)
        else:
            return np.ones_like(dist)

    def compute_nearest(self):
        if 'X_pca' in self.adata.obsm.keys():
            assert isinstance(self.adata.obsm['X_pca'], np.ndarray)
        assert isinstance(self.adata.obsm['spatial'], np.ndarray)
        if self.rec_neigh_num:
            # sp_rec_neigh_adj = self.adata.obsm['sp_rec_adj']
            # N x K1
            # self.rec_neigh_idx = sp_rec_neigh_adj.nonzero()[1].reshape(len(self.adata.X), self.rec_neigh_num)
            self.rec_neigh_idx = self.adata.obsm['sp_k'][:, :self.rec_neigh_num]
            self.torch_rec_neigh_idx = torch.from_numpy(self.rec_neigh_idx).to(torch.long)
        if self.forward_neigh_num:
            # sp_forward_neigh_adj = self.adata.obsm['sp_forward_adj']
            # N x K2
            # self.forward_neigh_idx = sp_forward_neigh_adj.nonzero()[1].reshape(len(self.adata.X), self.forward_neigh_num)
            self.forward_neigh_idx = self.adata.obsm['sp_k'][:, :self.forward_neigh_num]
            self.torch_forward_neigh_idx = torch.from_numpy(self.forward_neigh_idx).to(torch.long)
        if self.rec_neigh_num and self.forward_neigh_num:
            # rec_forward_neigh_idx = np.nonzero(sp_neighbors_adj[rec_neigh_idx, :])[2].reshape(len(self.adata.X), self.rec_neigh_num, self.rec_neigh_num)
            # rec_forward_neigh_idx = sp_neighbors_adj[rec_neigh_idx, :].nonzero()[2].reshape(len(self.adata.X), self.rec_neigh_num, self.rec_neigh_num) # will be supported in future scipy
            # N x K1 x K2
            self.rec_forward_neigh_idx = self.forward_neigh_idx[self.rec_neigh_idx]
            # self.rec_forward_neigh_idx = []
            # # for i in trange(self.rec_neigh_num):
            # for i in range(self.rec_neigh_num):
            #     self.rec_forward_neigh_idx.append(sp_forward_neigh_adj[self.rec_neigh_idx[:, i]].nonzero()[1].reshape(len(self.adata.X), self.forward_neigh_num))
            # self.rec_forward_neigh_idx = np.stack(self.rec_forward_neigh_idx, axis=1)
            self.torch_rec_forward_neigh_idx = torch.from_numpy(self.rec_forward_neigh_idx).to(torch.long)

    def get_gt_neighbors(self, expected_row_repeat_time):
        ks = np.array(self.hexagonal_num_list)
        for possible_neighbors in ks[ks > expected_row_repeat_time]:
            neigh = NearestNeighbors(n_neighbors=possible_neighbors, n_jobs=self.n_jobs)
            neigh.fit(self.adata.obsm['spatial'])
            neighbor_dist, neighbors = neigh.kneighbors(return_distance=True)
            # if all target spots have at least k neighbors with the same label
            if ((self.adata.obs["spatialLIBD"][neighbors.flatten()].to_numpy().reshape(neighbors.shape) == self.adata.obs["spatialLIBD"].to_numpy().reshape(-1, 1)).sum(1) < expected_row_repeat_time).sum() == 0:
                print(f"Search for {possible_neighbors} neighbors.")
                break
        row_idx = []
        col_idx = []
        row_repeat_time = 0
        last_row = 0
        # identify exactly k neighbors (because of some spots may have more than k neighbors with the same label)
        for r, c in zip(*(self.adata.obs["spatialLIBD"][neighbors.flatten()].to_numpy().reshape(neighbors.shape) == self.adata.obs["spatialLIBD"].to_numpy().reshape(-1, 1)).nonzero()):
            if r == last_row:
                if row_repeat_time < expected_row_repeat_time:
                    row_repeat_time += 1
                    row_idx.append(r)
                    col_idx.append(c)
            else:
                row_repeat_time = 1
                last_row = r
                row_idx.append(r)
                col_idx.append(c)
        row_idx = np.array(row_idx)
        col_idx = np.array(col_idx)
        gt_neighbor_dist = neighbor_dist[row_idx, col_idx].reshape(self.adata.X.shape[0], -1)
        gt_neighbors = neighbors[row_idx, col_idx].reshape(self.adata.X.shape[0], -1)
        gt_neighbor_graph = np.zeros((self.adata.X.shape[0], self.adata.X.shape[0]))
        gt_neighbor_graph[row_idx, gt_neighbors.flatten()] = 1.
        return gt_neighbor_dist, gt_neighbors, gt_neighbor_graph

    def get_rec_mask(self):
        print("rec_mask_neigh_threshold", self.rec_mask_neigh_threshold)
        if isinstance(self.adata.layers[self.count_key], np.ndarray):
            self.adata.layers["rec_mask"] = self.adata.layers[self.count_key] != 0
        else:
            self.adata.layers["rec_mask"] = self.adata.layers[self.count_key].A != 0
        if self.rec_neigh_num:
            if isinstance(self.adata.layers[self.count_key], np.ndarray):
                for i in range(len(self.adata)):
                    self.adata.layers["rec_mask"][i] *= ((self.adata.layers[self.count_key][self.rec_neigh_idx[i], :] > 0).mean(0) >= self.rec_mask_neigh_threshold)
            else:
                for i in range(len(self.adata)):
                    self.adata.layers["rec_mask"][i] *= ((self.adata.layers[self.count_key][self.rec_neigh_idx[i], :].A > 0).mean(0) >= self.rec_mask_neigh_threshold)

    def set_class_num(self):
        if self.annotation_key is not None:
            auto_classes = len(np.unique(self.adata.obs[f'{self.annotation_key}_int'][self.adata.obs['is_labelled']]))
        if self.num_classes == "auto" or self.num_classes == 0 or self.num_classes == "0":
            self.num_classes = auto_classes
        elif isinstance(self.num_classes, int):
            pass
        elif self.num_classes.startswith("plus_"):
            self.num_classes = auto_classes + int(self.num_classes[5:])
        elif self.num_classes.startswith("minus_"):
            self.num_classes = auto_classes - int(self.num_classes[6:])
        elif self.num_classes.startswith("multiply_"):
            self.num_classes = auto_classes * int(self.num_classes[9:])
        elif self.num_classes.startswith("divide_"):
            self.num_classes = auto_classes / int(self.num_classes[7:])

    def init_neighs_num(self):
        assert self.rec_neigh_num <= self.max_dynamic_neigh
        assert self.forward_neigh_num <= self.max_dynamic_neigh
        if self.dynamic_neigh_level == Dynamic_neigh_level.domain:
            self.dynamic_neigh_nums = [self.k] * self.num_classes
        elif self.dynamic_neigh_level.name.startswith("unit"):
            if not self.dynamic_neigh_level.name.startswith("unit_fix_domain"):
                self.dynamic_neigh_nums = [self.k] * len(self.adata)
            else:
                self.dynamic_neigh_nums = [self.unit_fix_num] * len(self.adata)

    # @profile
    def preprocess_data(self, filter_by_counts=True, reuse_existing_pca=False, reuse_existing_knn=False):
        if self.annotation_key is not None:
            self.adata.obs[f"{self.annotation_key}"] = pd.Categorical(self.adata.obs[f"{self.annotation_key}"])
            sorted_cluster_idx = self.adata.obs[f"{self.annotation_key}"].value_counts().sort_index().index
            cluster_str2int_dict = dict(zip(sorted_cluster_idx, range(len(sorted_cluster_idx))))
            self.adata.obs[f"{self.annotation_key}_int"] = self.adata.obs[f"{self.annotation_key}"].map(cluster_str2int_dict)
            if np.sum(pd.isna(self.adata.obs[f"{self.annotation_key}_int"])) > 0:
                self.adata.obs[f"{self.annotation_key}_int"] = self.adata.obs[f"{self.annotation_key}_int"].cat.add_categories([-1])
                self.adata.obs.loc[pd.isna(self.adata.obs[f"{self.annotation_key}_int"]), f"{self.annotation_key}_int"] = -1
            if self.supervise_cat:
                self.adata = self.adata[self.adata.obs[f"{self.annotation_key}_int"] != -1]
                if self.supervise_cat == "random":
                    self.adata.obs["random_label"] = self.rng.integers(len(np.unique(self.adata.obs[f"{self.annotation_key}_int"].to_numpy())), size=self.adata.shape[0])
                    # from sklearn.metrics import adjusted_rand_score
                    # raise ValueError(f"Random label ARI: {adjusted_rand_score(self.adata.obs[f'{self.annotation_key}_int'], self.adata.obs['random_label'])}")
            self.adata.obs[f"{self.annotation_key}_int"] = self.adata.obs[f"{self.annotation_key}_int"].astype(np.int32)
            self.adata.obs['is_labelled'] = ~self.adata.obs[f"{self.annotation_key}"].isna()
            if self.test_with_gt_sp:
                self.adata = self.adata[~self.adata.obs[f"{self.annotation_key}"].isna(), :].copy()
        assert self.adata.obsm['spatial'] is not None
        if reuse_existing_knn:
            assert self.adata.obsm['sp_k'] is not None
            raise
        self.adata.var_names_make_unique()

        if self.count_key is not None:
            whether_copy = self.adata.X != self.adata.layers[self.count_key]
            if (isinstance(whether_copy, np.ndarray) and (whether_copy).any()) or (scipy.sparse.issparse(whether_copy) and whether_copy.nnz > 0):
                self.adata.X = self.adata.layers[self.count_key].astype(np.float32).copy()
        logging.debug("Filtering genes and cells.")
        tic = time.time()

        if filter_by_counts:
            sc.pp.filter_genes(self.adata, min_counts=1)
            sc.pp.filter_cells(self.adata, min_counts=1)
        toc = time.time()
        logging.debug(f"Filtering genes and cells takes {(toc - tic)/60:.2f} mins.")
        if self.device.type.startswith("cpu"):
            self.cpu_adata = self.adata.copy()
        if reuse_existing_pca:
            assert self.in_type == 'pca_scaled' or self.in_type == 'pca'
            self.adata.obsm["X_pca"] = self.adata.obsm[self.pca_key]
        else:
            if self.num_hvg is not None:
                if isinstance(self.num_hvg, float):
                    self.num_hvg = int(self.num_hvg * self.adata.shape[1])
                elif isinstance(self.num_hvg, int):
                    logging.debug("Selecting highly variable genes.")
                    tic = time.time()
                    # sc.pp.highly_variable_genes(self.adata, n_top_genes=self.num_hvg, flavor="cell_ranger", subset=True)
                    sc.pp.highly_variable_genes(self.adata, n_top_genes=self.num_hvg, flavor="seurat_v3", subset=True)
                    toc = time.time()
                    logging.debug(f"Selecting highly variable genes takes {(toc - tic)/60:.2f} mins.")

            if filter_by_counts:
                sc.pp.filter_cells(self.adata, min_counts=1)
            try:
                self.adata.X = self.adata.X.astype(np.float32).toarray()
            except:
                pass
            if self.lib_norm:
                logging.debug("Normalizing data to the median libaray size.")
                tic = time.time()
                self.adata.uns['target_library_size'] = np.median(self.adata.X.sum(1)).astype(np.int32)
                sc.pp.normalize_total(self.adata)
                # sc.pp.normalize_total(self.adata, target_sum=1e4)
                toc = time.time()
                logging.debug(f"Normalizing data to the median libaray size takes {(toc - tic)/60:.2f} mins.")

            logging.debug("Log transforming data.")
            tic = time.time()
            sc.pp.log1p(self.adata)
            toc = time.time()
            logging.debug(f"Log transforming data takes {(toc - tic)/60:.2f} mins.")
            self.adata.layers['unscaled'] = self.adata.X.copy()
            logging.debug("Scaling data.")
            tic = time.time()
            sc.pp.scale(self.adata)
            # sc.pp.scale(self.adata, zero_center=False, max_value=10)
            self.adata.var['mean'] = self.adata.var['mean'].astype(np.float32)
            self.adata.var['std'] = self.adata.var['std'].astype(np.float32)
            toc = time.time()
            logging.debug(f"Scaling data takes {(toc - tic)/60:.2f} mins.")
            self.adata.layers['scaled'] = self.adata.X.copy()
            logging.debug("Running PCA.")
            tic = time.time()
            if self.adata.shape[1] > self.n_pc:
                pca = PCA(n_components=self.n_pc, random_state=self.seed)
                if self.in_type == 'pca_unscaled':
                    self.adata.obsm["X_pca"] = pca.fit_transform(self.adata.layers['unscaled'])
                elif self.in_type == 'pca_scaled' or self.in_type == 'pca':
                    self.adata.obsm["X_pca"] = pca.fit_transform(self.adata.layers['scaled'])
                else:
                    self.adata.obsm["X_pca"] = pca.fit_transform(self.adata.X)
                self.adata.uns["normalized_pca_explained_variance_ratio"] = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
                self.adata.uns['pca'] = pca
            else:
                logging.warning(f"Number of genes is smaller than the number of PCs. Using scaled genes as PCs.")
                self.adata.obsm["X_pca"] = self.adata.layers['scaled'].copy()
            toc = time.time()
            logging.debug(f"Running PCA takes {(toc - tic)/60:.2f} mins.")
        if self.out_type == "raw":
            gene_mean = self.adata.layers[self.count_key].mean(axis=0)
            gene_var = self.adata.layers[self.count_key].var(axis=0)
            good_gene_idx = gene_var < np.quantile(gene_var, 0.99)
            popt, _ = curve_fit(nb_func, gene_mean[good_gene_idx], gene_var[good_gene_idx])
            self.adata.uns["r"] = popt[0]
        else:
            self.adata.uns["r"] = -1
        self.adata.obs["pred_labels"] = 0
        self.adata = self.adata.copy()
        self.set_class_num()
        if self.max_dynamic_neigh:
            self.init_neighs_num()
        logging.debug("Computing spatial nearest neighbors.")
        tic = time.time()
        if not self.test_with_gt_sp:
            if self.max_dynamic_neigh:
                neigh = NearestNeighbors(n_neighbors=self.max_dynamic_neigh, n_jobs=self.n_jobs)
            else:
                neigh = NearestNeighbors(n_neighbors=max(self.rec_neigh_num, self.forward_neigh_num), n_jobs=self.n_jobs)
            if self.data_type == "10x" and len(self.sample_id.split('_')) > 1:
                neigh.fit(self.adata[self.adata.obs["batch"] == 0].obsm['spatial'])
                first_slide_sp_dist, _ = neigh.kneighbors(return_distance=True)
                min_unit_dist = first_slide_sp_dist.min()
                self.adata.obsm['spatial'] = np.hstack((self.adata.obsm['spatial'], (self.adata.obs["batch"] * min_unit_dist * self.z_scale).to_numpy().reshape(-1, 1)))
            neigh.fit(self.adata.obsm['spatial'])
            self.adata.obsm['sp_dist'], self.adata.obsm['sp_k'] = neigh.kneighbors(return_distance=True)
        else:
            if len(self.sample_id.split('_')) > 1:
                raise NotImplementedError
            if self.max_dynamic_neigh:
                gt_neighbor_dist, gt_neighbors, gt_neighbor_graph = self.get_gt_neighbors(expected_row_repeat_time=self.max_dynamic_neigh)
            else:
                gt_neighbor_dist, gt_neighbors, gt_neighbor_graph = self.get_gt_neighbors(expected_row_repeat_time=max(self.rec_neigh_num, self.forward_neigh_num))
            self.adata.obsm['sp_k'] = gt_neighbors
            self.adata.obsm['sp_dist'] = gt_neighbor_dist
        self.adata.obsm['sp_dist'] = self.adata.obsm['sp_dist'].astype(np.float32)
        if self.max_dynamic_neigh:
            if self.supervise_cat or self.exchange_forward_neighbor_order:
                raise NotImplementedError
            if not self.test_with_gt_sp:
                self.adata.obsm['sp_adj'] = neigh.kneighbors_graph()
            else:
                self.adata.obsm['sp_adj'] = gt_neighbor_graph
            self.adata.obsm["consist_adj"] = np.ones_like(self.adata.obsm["sp_k"]).astype(np.bool_)
        toc = time.time()
        logging.debug(f"Computing spatial nearest neighbors takes {(toc - tic)/60:.2f} mins.")
        gc.collect()

    def get_custom_data(self):
        self.adata = sc.read_h5ad(osp.join(self.data_dir, self.dataset_dir, self.data_file_name))
        self.preprocess_data()
