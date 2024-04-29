import numpy as np
from dgl.data import DGLDataset
from src.utils.util import *
import dgl
import torch
import gc
from src.utils.states import Dynamic_neigh_level
# import time
# from sklearn.neighbors import NearestNeighbors

# import logging
# import scipy
# from scipy.spatial.distance import cdist, pdist
# import scanpy as sc
# import pandas as pd
# from os import path as osp
# import os
# from memory_profiler import profile

class MyDGLDataset(DGLDataset):
    """Graph dataset object, loading dataset in a batch-wise manner.

    Args:
        k (int): k nearest neighbor used in neighbors.
        rec_neigh_num (boolean): whether to use the reconstructing
            neighbors strategy.

    """
    def __init__(
        self,
        in_type: str = "raw",
        out_type: str = "raw",
        count_key="counts",
        sample_id: str = "151673",
        dynamic_neigh_nums=None,
        dynamic_neigh_level=Dynamic_neigh_level.unit,
        unit_fix_num=None,
        unit_dynamic_num=None,
        start_use_domain_neigh=False,
        adata=None,
        load_whole_graph_on_gpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        super().__init__(name=sample_id)
        self.in_type = in_type
        self.out_type = out_type
        self.count_key = count_key
        self.adata = adata
        self.dynamic_neigh_nums = dynamic_neigh_nums
        self.dynamic_neigh_level = dynamic_neigh_level
        self.unit_fix_num = unit_fix_num
        self.unit_dynamic_num = unit_dynamic_num
        self.start_use_domain_neigh = start_use_domain_neigh
        self.load_whole_graph_on_gpu = load_whole_graph_on_gpu
        self.rng = np.random.default_rng(seed)
        us = []
        vs = []
        es = []
        if self.dynamic_neigh_level == Dynamic_neigh_level.domain:
            if len(np.unique(self.dynamic_neigh_nums)) == 1:
                us = self.adata.obsm['sp_k'][:, :self.dynamic_neigh_nums[0]].flatten()
                vs = np.repeat(np.arange(self.adata.shape[0]), self.dynamic_neigh_nums[0])
                es = self.adata.obsm["sp_dist"][:, :self.dynamic_neigh_nums[0]].flatten()
            else:
                for j, dynamic_neigh_num in enumerate(self.dynamic_neigh_nums):
                    us += self.adata.obsm['sp_k'][self.adata.obs["pred_labels"] == j, :dynamic_neigh_num].flatten().tolist()
                    vs += np.repeat(np.where(self.adata.obs["pred_labels"] == j)[0], dynamic_neigh_num).tolist()
                    es += self.adata.obsm["sp_dist"][self.adata.obs["pred_labels"] == j, :dynamic_neigh_num].flatten().tolist()
        elif self.dynamic_neigh_level == Dynamic_neigh_level.unit or self.dynamic_neigh_level == Dynamic_neigh_level.unit_freq_self:
            dynamic_neigh_nums_arr = np.array(self.dynamic_neigh_nums)
            dynamic_neigh_nums_uni = np.unique(dynamic_neigh_nums_arr)
            for dynamic_neigh_num in dynamic_neigh_nums_uni:
                this_cluster_node_idx = dynamic_neigh_nums_arr == dynamic_neigh_num
                # raise ValueError(f"consist_adj type {type(self.adata[this_cluster_node_idx].obsm['consist_adj'])}")
                us += self.adata[this_cluster_node_idx].obsm["sp_k"][self.adata[this_cluster_node_idx].obsm["consist_adj"]].flatten().tolist()
                vs += np.repeat(np.arange(self.adata.shape[0])[this_cluster_node_idx], dynamic_neigh_num).tolist()
                es += self.adata[this_cluster_node_idx].obsm["sp_dist"][self.adata[this_cluster_node_idx].obsm["consist_adj"]].flatten().tolist()
                # print("this_cluster_unit", this_cluster_node_idx.sum())
                # print("us", len(us), "vs", len(vs), "es", len(es))
        elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_fix_domain or self.dynamic_neigh_level == Dynamic_neigh_level.unit_fix_domain_boundary:
            # add units relevant to fixed spatial neighbors
            us += self.adata.obsm["sp_k"][:, :self.unit_fix_num].flatten().tolist()
            vs += np.repeat(np.arange(self.adata.shape[0]), self.unit_fix_num).tolist()
            # es += self.adata.obsm["sp_dist"][:, :self.unit_fix_num].flatten().tolist()
            es += [1] * self.adata.shape[0] * self.unit_fix_num
            if not self.start_use_domain_neigh:
                self.adata.obsm["used_k"] = self.adata.obsm["sp_k"][:, :self.unit_fix_num]
            else:
                self.adata.obsm["used_k"] = self.adata.obsm["sp_k"].copy()
                # uni_domains = np.unique(self.adata.obs["pred_labels"])
                # if (len(uni_domains) == 1) and (not torch.cuda.is_available()):
                #     print("Load pred_labels for validation postprocessing.")
                #     self.adata.obs["pred_labels"] = pd.read_csv(label_path)
                # add units relevant to dynamic domain neighbors
                for j in np.unique(self.adata.obs["pred_labels"]):
                    curr_domain_unit_idx = self.adata.obs["pred_labels"] == j
                    curr_domain_unit_num_idx = np.where(curr_domain_unit_idx)[0]
                    # !! what if rng choice select the target unit itself as neighbors?
                    if self.dynamic_neigh_level == Dynamic_neigh_level.unit_fix_domain_boundary:
                        # only for boundary units, we use the same domain units
                        curr_domain_boundary_unit_num_idx = get_domain_boundary_unit_indices(self.adata, domain_idx=j)
                        same_domain_random_unit_idx = self.rng.choice(curr_domain_unit_num_idx, (len(curr_domain_boundary_unit_num_idx), self.unit_dynamic_num), replace=True)
                        self.adata.obsm["used_k"][curr_domain_boundary_unit_num_idx, self.unit_fix_num:] = same_domain_random_unit_idx
                    elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_fix_domain:
                        # for all current domain unit, we use the same domain units
                        curr_domain_all_unit_num_idx = curr_domain_unit_num_idx
                        same_domain_random_unit_idx = self.rng.choice(curr_domain_unit_num_idx, (len(curr_domain_all_unit_num_idx), self.unit_dynamic_num), replace=True)
                        self.adata.obsm["used_k"][curr_domain_all_unit_num_idx, self.unit_fix_num:] = same_domain_random_unit_idx
                # add domain neighbors
                us += self.adata.obsm["used_k"][:, self.unit_fix_num:].flatten().tolist()
                vs += np.repeat(np.arange(self.adata.shape[0]), self.unit_dynamic_num).tolist()
                es += [1] * self.adata.shape[0] * self.unit_dynamic_num
        elif self.dynamic_neigh_level == Dynamic_neigh_level.unit_domain_boundary:
            if not self.start_use_domain_neigh:
                # add units relevant to fixed spatial neighbors
                us += self.adata.obsm["sp_k"][:, :self.unit_fix_num].flatten().tolist()
                vs += np.repeat(np.arange(self.adata.shape[0]), self.unit_fix_num).tolist()
                # es += self.adata.obsm["sp_dist"][:, :self.unit_fix_num].flatten().tolist()
                es += [1] * self.adata.shape[0] * self.unit_fix_num
                self.adata.obsm["used_k"] = self.adata.obsm["sp_k"][:, :self.unit_fix_num]
            else:
                self.adata.obsm["used_k"] = self.adata.obsm["sp_k"].copy()
                spatial_dynamic_neigh_nums_arr = np.array(self.dynamic_neigh_nums)
                spatial_dynamic_neigh_nums_uni = np.unique(spatial_dynamic_neigh_nums_arr)
                boundary_unit_indice_dict = {}
                boundary_domain_neighbor_indice_dict = {}
                used_neighbor_pointer = {}
                for j in np.unique(self.adata.obs["pred_labels"]):
                    boundary_unit_indice_dict[j] = get_domain_boundary_unit_indices(self.adata, domain_idx=j)
                    boundary_domain_neighbor_indice_dict[j] = self.rng.choice(curr_domain_unit_num_idx, (len(boundary_unit_indice_dict[j]), self.adata.obsm["sp_k"].shape[1]), replace=True)
                    used_neighbor_pointer[j] = 0
                for spatial_dynamic_neigh_num in spatial_dynamic_neigh_nums_uni:
                    this_cluster_node_idx = spatial_dynamic_neigh_nums_arr == spatial_dynamic_neigh_num
                    # add units relevant to dynamic spatial domain neighbors
                    us += self.adata[this_cluster_node_idx].obsm["sp_k"][self.adata[this_cluster_node_idx].obsm["consist_adj"]].flatten().tolist()
                    vs += np.repeat(np.arange(self.adata.shape[0])[this_cluster_node_idx], spatial_dynamic_neigh_num).tolist()
                    es += self.adata[this_cluster_node_idx].obsm["sp_dist"][self.adata[this_cluster_node_idx].obsm["consist_adj"]].flatten().tolist()
                    # add units relevant to dynamic domain neighbors
                    for j in np.unique(self.adata[this_cluster_node_idx].obs["pred_labels"]):
                        curr_domain_unit_idx = self.adata[this_cluster_node_idx].obs["pred_labels"] == j
                        curr_domain_unit_num_idx = np.where(curr_domain_unit_idx)[0]
                        # !! what if rng choice select the target unit itself as neighbors?
                        # only for boundary units, we use the same domain units
                        this_dynamic_num_boundary_unit_idx = list(set(boundary_unit_indice_dict[j].tolist()) & set(curr_domain_unit_num_idx.tolist()))
                        self.adata.obsm["used_k"][this_dynamic_num_boundary_unit_idx, spatial_dynamic_neigh_num:] = boundary_domain_neighbor_indice_dict[j][used_neighbor_pointer[j]:len(this_dynamic_num_boundary_unit_idx), :(self.adata.obsm["sp_k"].shape[1] - spatial_dynamic_neigh_num)]
                        used_neighbor_pointer[j] += len(this_dynamic_num_boundary_unit_idx)
        self.graph = dgl.graph((us, vs), num_nodes=len(self.adata))
        self.graph.edata["sp_dist"] = torch.from_numpy(np.array(es))
        # self.adj = self.graph.adjacency_matrix().to_dense().numpy()
        # print(self.adj.sum(0).min(), self.adj.sum(0).max())
        # print(min(self.dynamic_neigh_nums), max(self.dynamic_neigh_nums))
        exp_feature, exp_rec_feature = get_io_feature(self.adata, self.in_type, self.out_type, self.count_key)
        self.graph.ndata["exp_feature"] = torch.from_numpy(exp_feature)
        if "rec_mask" in self.adata.layers.keys():
            self.graph.ndata["exp_rec_mask"] = torch.from_numpy(self.adata.layers["rec_mask"])
        if self.in_type != self.out_type:
            self.graph.ndata["exp_rec_feature"] = torch.from_numpy(exp_rec_feature)
        if self.load_whole_graph_on_gpu and torch.cuda.is_available():
            self.graph = self.graph.to(torch.device("cuda"))
        # gc.collect()

    def __getitem__(self, index: int):
        return self.graph

    def process(self):
        pass

    # def __len__(self):
    #     return 1