import logging
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from src.utils.plot import update_graph_labels, COLOUR_DICT
from os import path as osp
import os
import pandas as pd
import scanpy as sc
import numba
from scipy.spatial.distance import pdist
from scipy.linalg import fractional_matrix_power
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import warnings
import torch.distributed as dist
import networkx as nx

# https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
def gather_nd_to_rank(tensor, axis=0, rank=0):
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is axis. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.
    TODO: add memory dict for this function to avoid reading the tensor size from all processes

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        (Tensor): output list of tensors that can be of different sizes
    """
    world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_length = max(size[axis] for size in all_sizes)
    # print(f"rank: {dist.get_rank()}, tensor_size: {tensor.size()}, max_length: {max_length}")
    # length_diff = max_length.item() - local_size[0].item()
    length_diff = max_length - local_size[axis]
    if length_diff:
        pad_size = (*tensor.size()[0:axis], length_diff, *tensor.size()[(axis+1):])
        padding = torch.empty(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding), dim=axis)
        # print(f"rank: {dist.get_rank()}, pad_size: {pad_size}")

    all_tensors_padded = [torch.empty_like(tensor) for _ in range(world_size)]
    if rank == "all":
        dist.all_gather(all_tensors_padded, tensor)
    elif isinstance(rank, int):
        dist.gather(tensor, all_tensors_padded if dist.get_rank() == rank else None, dst=rank)
    else:
        raise ValueError("rank should be either 'all' or int")
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        # https://stackoverflow.com/questions/41418499/selecting-which-dimension-to-index-in-a-numpy-array
        indices = {
            axis: slice(None, size[axis]),
        }
        ix = [indices.get(dim, slice(None)) for dim in range(tensor_.ndim)]
        all_tensors.append(tensor_[ix])
        # all_tensors.append(tensor_[:size[0]])
        # print(f"rank: {dist.get_rank()}, tensor_.shape: {tensor_.shape}, ix: {ix}, size: {size}")
    return torch.cat(all_tensors, dim=axis)
    # return_tensor = torch.cat(all_tensors, dim=axis)
    # print(f"rank: {dist.get_rank()}, return_tensor_size: {return_tensor.size()}")
    # return return_tensor

def check_mem(cuda_device):
    devices_info = os.popen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used

def preallocate_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.98)
    block_mem = max(0, max_mem - used)
    # x = torch.cuda.FloatTensor(256, 1024, block_mem)
    x = torch.zeros(256, 1024, block_mem, device=torch.device(f"cuda:{cuda_device}"))
    del x

def copy_dict_delete_item(d, key):
    d = d.copy()
    del d[key]
    return d

def supress_warning():
    warnings.filterwarnings("ignore", message="Variable names are not unique. ")
    warnings.filterwarnings("ignore", message="To make them unique, call `.var_names_make_unique`.")
    warnings.filterwarnings("ignore", message="The loaded checkpoint was produced with Lightning")
    warnings.filterwarnings("ignore", message="Dataloader CPU affinity opt is not enabled, consider switching it on")
    warnings.filterwarnings("ignore", message="inaccurate if each worker is not configured independently to avoid having duplicate data")
    warnings.filterwarnings("ignore", message="on epoch level in distributed setting to accumulate the metric across devices.")

def get_io_feature(adata, in_type, out_type, count_key):
    if in_type == "raw":
        exp_feature = adata.layers[count_key]
    elif in_type == "unscaled":
        exp_feature = adata.layers['unscaled']
    elif in_type == "scaled":
        exp_feature = adata.layers['scaled']
    elif in_type.startswith("pca"):
        if in_type.endswith("harmony"):
            exp_feature = adata.obsm['X_pca_harmony']
        else:
            exp_feature = adata.obsm['X_pca']
    if out_type == "raw":
        exp_rec_feature = adata.layers[count_key]
    elif out_type == "unscaled":
        exp_rec_feature = adata.layers['unscaled']
    elif out_type == "scaled":
        exp_rec_feature = adata.layers['scaled']
    elif out_type.startswith("pca"):
        if out_type.endswith("harmony"):
            exp_rec_feature = adata.obsm['X_pca_harmony']
        else:
            exp_rec_feature = adata.obsm['X_pca']
    return exp_feature, exp_rec_feature

def nb_func(gene_mean, r):
    return gene_mean + gene_mean**2 / r

def GraphST_refine_label(adata, radius=50, key='label'):
    from scipy.spatial.distance import squareform, pdist
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # Compute the pairwise distance matrix
    distance = pdist(adata.obsm['spatial'], metric='euclidean')
    # Convert the pairwise distance matrix to square form
    distance = squareform(distance)
    n_cell = distance.shape[0]

    for i in range(n_cell):
        sorted_index = distance[i, :].argsort()
        neigh_type = old_type[sorted_index[1:n_neigh+1]].tolist()
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    return np.array(new_type)

def refine_label(adata, n_neigh=60, label_key="pred_labels", neighbor_key="sp_k", samll_domain_threshold=50, refine_threshold=0.8):
    # 1. keep clusters with unit# <=50 from consideration, their labels will not be changed but will still be used to refine other clusters (e.g., change other domain nodes into this domain)
    # 2. for each unit in each cluster, if its largest neighbor cluster takes up more than refine_threshold * 100% of its neighbors, then change its label to the largest neighbor cluster
    assert n_neigh % 2 == 0
    old_type = adata.obs[label_key].values
    new_type = old_type.copy()
    if (neighbor_key not in adata.obs.keys()) or (adata.obs[neighbor_key].values.shape[1] != n_neigh):
        neigh = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1)
        neigh.fit(adata.obsm['spatial'])
        adata.obsm[neighbor_key] = neigh.kneighbors(return_distance=False)
    domain_indices, domain_unit_counts = np.unique(adata.obs[label_key], return_counts=True)
    target_domain_indices = domain_indices[domain_unit_counts > samll_domain_threshold]
    for domain_idx in target_domain_indices:
        this_domain_unit_indices = (adata.obs[label_key] == domain_idx).values
        this_domain_neighbor_bincounts = batched_bincount_for_row(np.concatenate((old_type[this_domain_unit_indices].reshape(-1, 1), get_neighbor_domain(adata, this_domain_unit_indices, label_key, neighbor_key)), 1))
        sorted_this_domain_neighbor_bincounts_idx = np.argsort(this_domain_neighbor_bincounts, axis=1, kind="stable")[:, ::-1]
        sorted_this_domain_neighbor_bincounts = np.take_along_axis(this_domain_neighbor_bincounts, sorted_this_domain_neighbor_bincounts_idx, 1)
        changed_unit_cond1 = sorted_this_domain_neighbor_bincounts[:, 0] >= refine_threshold * n_neigh
        changed_unit_cond2 = (sorted_this_domain_neighbor_bincounts_idx[:, 0] != domain_idx) & (sorted_this_domain_neighbor_bincounts_idx[:, 1] != domain_idx)
        changed_unit_idx = changed_unit_cond1 | changed_unit_cond2

        # prevent the whole domain from being changed
        if changed_unit_idx.sum() != this_domain_unit_indices.sum():
            new_type[np.nonzero(this_domain_unit_indices)[0][changed_unit_idx]] = sorted_this_domain_neighbor_bincounts_idx[changed_unit_idx, 0]
            assert (new_type[this_domain_unit_indices][changed_unit_idx] != sorted_this_domain_neighbor_bincounts_idx[changed_unit_idx, 0]).sum() == 0
            print("#", (sorted_this_domain_neighbor_bincounts_idx[changed_unit_idx, 0] != old_type[this_domain_unit_indices][changed_unit_idx]).sum(), "units in domain", domain_idx, "changed their labels")

    assert (new_type != old_type).sum() != 0
    return new_type

def expand_neighbor_for_existing_spots(adata, target_component_idx, max_iter=100, expanded_spot_num="equal"):
    target_component_num_idx = adata.obs_names.get_indexer_for(target_component_idx)
    too_near_spots_list = [s for s in np.hstack(adata.obsm["sp_r"][target_component_num_idx]) if s not in target_component_num_idx]
    too_near_spots_list = list(dict.fromkeys(too_near_spots_list))
    expand_on = too_near_spots_list
    for i in range(max_iter):
        expanded_near_spots_list = [s for s in np.hstack(adata.obsm["sp_r"][expand_on]) if s not in target_component_num_idx and s not in too_near_spots_list]
        expanded_near_spots_list = list(dict.fromkeys(expanded_near_spots_list))
        if expanded_spot_num == "equal" and len(expanded_near_spots_list) >= len(target_component_num_idx):
            return expanded_near_spots_list[:len(target_component_num_idx)]
        elif isinstance(expanded_spot_num, int) and len(expanded_near_spots_list) >= expanded_spot_num:
            return expanded_near_spots_list[:expanded_spot_num]
        expand_on = expanded_near_spots_list
        if i == max_iter - 1:
            print("Warning: reach max iteration, return all expanded spots")
            return expanded_near_spots_list

def construct_neighbor_graph_for_10x(adata):
    # print(pdist(adata.obsm['spatial']).min())
    p = pdist(adata.obsm['spatial'])
    p.sort()
    # display(p)
    # display(p[p > (p[0] * 1.1)])
    radius = p[p > (p[0] * 1.1)][0] - 1
    assert radius > p[0] * 1.5
    neigh = NearestNeighbors(radius=radius, n_jobs=-1)
    neigh.fit(adata.obsm['spatial'])
    adata.obsm['sp_r'] = neigh.radius_neighbors(return_distance=False)
    adata.obsm['sp_r_adj'] = neigh.radius_neighbors_graph()
    assert adata.obsm['sp_r_adj'].sum(1).max() == 6

def load_10x_with_meta(data_dir, dataset_dir, sample_id, count_key, annotation_key, filter_genes=1, filter_cells=1, filter_unlabelled=False, load_sce=False):
    if len(sample_id.split('_')) > 1:
        adata = sc.read_h5ad(osp.join(data_dir, dataset_dir, sample_id, "{}.h5ad".format(sample_id)))
    else:
        adata = sc.read_visium(osp.join(data_dir, dataset_dir, sample_id), count_file="{}_filtered_feature_bc_matrix.h5".format(sample_id))
        adata.X = adata.X.toarray()
    if load_sce:
        sce_adata = sc.read_h5ad(osp.join(data_dir, dataset_dir, sample_id, f"{sample_id}.sce.h5ad"))
        adata.layers["logcounts"] = sce_adata.layers["logcounts"].copy()
        adata.obsm["X_pca"] = sce_adata.obsm["PCA"].to_numpy().astype(np.float32)
    adata.var_names_make_unique()
    if len(sample_id.split('_')) == 1:
        cell_meta_df = pd.read_csv(osp.join(data_dir, dataset_dir, "spatialLIBD", "{}.meta.csv.gz".format(sample_id)), index_col=0)
        cell_meta_df = cell_meta_df.drop(columns=['in_tissue'])
        adata.obs = cell_meta_df
    elif len(sample_id.split('_')) > 1:
        sample_id_list = sample_id.split('_')
        new_obs_df = pd.DataFrame()
        for i, a_sample_id in enumerate(sample_id_list):
            cell_meta_df = pd.read_csv(osp.join(data_dir, dataset_dir, "spatialLIBD", "{}.meta.csv.gz".format(a_sample_id)), index_col=0)
            cell_meta_df = cell_meta_df.drop(columns=['in_tissue'])
            cell_meta_df = cell_meta_df.set_index("key")
            new_obs_df = pd.concat([new_obs_df, cell_meta_df], axis=0)
        adata.obs.index = new_obs_df.index
        new_obs_df["batch"] = adata.obs["batch"].astype(np.int32)
        adata.obs = new_obs_df
    sorted_cluster_idx = adata.obs[annotation_key].value_counts().sort_index().index
    cluster_str2int_dict = dict(zip(sorted_cluster_idx, range(len(sorted_cluster_idx))))
    adata.obs[f"{annotation_key}_int"] = adata.obs[annotation_key].map(cluster_str2int_dict)
    if np.sum(pd.isna(adata.obs[f"{annotation_key}_int"])) > 0:
        try:
            adata.obs.loc[pd.isna(adata.obs[f"{annotation_key}_int"]), f"{annotation_key}_int"] = -1
        except:
            adata.obs.loc[:, f"{annotation_key}_int"] = adata.obs.loc[:, f"{annotation_key}_int"].cat.add_categories([-1]).fillna(-1)
    adata.obs[f"{annotation_key}_int"] = adata.obs[f"{annotation_key}_int"].astype(np.int32)
    adata.obs['sum_umi'] = adata.obs['sum_umi'].astype(np.int32)
    adata.obs['is_labelled'] = ~adata.obs[annotation_key].isna()
    if filter_unlabelled:
        adata = adata[~adata.obs[annotation_key].isna(), :].copy()
    # adata._inplace_subset_var([not s.startswith("MT-") and not s.startswith("ERCC") for s in adata.var_names])
    # adata._inplace_subset_var([not s.startswith("MT-")for s in adata.var_names])
    adata._inplace_subset_var([not s.startswith("ERCC") for s in adata.var_names])
    sc.pp.filter_genes(adata, min_counts=filter_genes)
    sc.pp.filter_cells(adata, min_counts=filter_cells)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
    adata.layers[count_key] = adata.X.copy()
    return adata

def get_spatial_neighbors(adata, index='all', k=6):
    if index == 'all':
        index = np.arange(adata.shape[0])
    spatial_neighbors = pd.DataFrame(index=adata.obs_names[index])
    ks = [n * (n + 1) * 3 for n in range(1, 100)]
    assert k in ks

    for spot in tqdm(index):
        start_k = 0
        end_k = ks.index(k)
        curr_spot = adata[spot]
        for i in range(1, k + 1):
            if i > ks[start_k]:
                start_k += 1
            try:
                if i == 1:
                    spatial_neighbors.loc[curr_spot.obs_names, 'neighbor_' + str(i)] = adata[(adata.obs['array_row'].values == (curr_spot.obs['array_row'] - 1).values) & (adata.obs['array_col'].values == (curr_spot.obs['array_col'] - 1).values)].obs_names
                elif i == 2:
                    spatial_neighbors.loc[curr_spot.obs_names, 'neighbor_' + str(i)] = adata[(adata.obs['array_row'].values == (curr_spot.obs['array_row'] - 1).values) & (adata.obs['array_col'].values == (curr_spot.obs['array_col'] + 1).values)].obs_names
                elif i == 3:
                    spatial_neighbors.loc[curr_spot.obs_names, 'neighbor_' + str(i)] = adata[(adata.obs['array_row'].values == curr_spot.obs['array_row'].values) & (adata.obs['array_col'].values == (curr_spot.obs['array_col'] + 2).values)].obs_names
                elif i == 4:
                    spatial_neighbors.loc[curr_spot.obs_names, 'neighbor_' + str(i)] = adata[(adata.obs['array_row'].values == (curr_spot.obs['array_row'] + 1).values) & (adata.obs['array_col'].values == (curr_spot.obs['array_col'] + 1).values)].obs_names
                elif i == 5:
                    spatial_neighbors.loc[curr_spot.obs_names, 'neighbor_' + str(i)] = adata[(adata.obs['array_row'].values == (curr_spot.obs['array_row'] + 1).values) & (adata.obs['array_col'].values == (curr_spot.obs['array_col'] - 1).values)].obs_names
                elif i == 6:
                    spatial_neighbors.loc[curr_spot.obs_names, 'neighbor_' + str(i)] = adata[(adata.obs['array_row'].values == curr_spot.obs['array_row'].values) & (adata.obs['array_col'].values == (curr_spot.obs['array_col'] - 2).values)].obs_names
            except ValueError:
                pass
    # return adata.obs.iloc[index][['array_row', 'array_col']].min().values, adata.obs.iloc[index][['array_row', 'array_col']].max().values
    return spatial_neighbors

def get_spot_and_neighbors_attribute(adata, attr, k=6):
    columns = ['spot'] + ["spot_neigh_{}".format(i) for i in range(1, k + 1)]
    spot_and_neighbors_attrs = pd.DataFrame(index=adata.obs_names, columns=columns)
    spot_and_neighbors = np.concatenate([np.reshape(np.arange(len(adata)), (-1, 1)), adata.obsm['sp_k']], axis=1)
    # print("spot_and_neighbors.shape", spot_and_neighbors.shape)
    # print(np.vstack([adata.obs[attr].iloc[spot_and_neighbors[:, i]].values for i in range(spot_and_neighbors.shape[1])]).shape)
    spot_and_neighbors_attrs[columns] = np.vstack([adata.obs[attr].iloc[spot_and_neighbors[:, i]].values for i in range(spot_and_neighbors.shape[1])]).T
    return spot_and_neighbors_attrs

def copy_dict_to_cpu(dic):
    new_dic = {}
    if dic is not None:
        for k, v in dic.items():
            if isinstance(v, dict) or isinstance(v, defaultdict):
                new_dic[k] = copy_dict_to_cpu(v)
            elif isinstance(v, torch.Tensor):
                new_dic[k] = v.detach().cpu()
            else:
                # print(k, v) # att_x
                new_dic[k] = v
        return new_dic
    else:
        return None

def debug_finite_param(module, where=""):
    stop_flag = False
    for name, param in module.named_parameters():
        if torch.isnan(param).any():
            logging.error(f"{where}: Parameter of {name} has NaN")
            stop_flag = True
        if torch.isinf(param).any():
            logging.error(f"{where}: Parameter of {name} has infinite")
            stop_flag = True
    if stop_flag:
        raise

def debug_finite_grad(module, where=""):
    stop_flag = False
    for name, param in module.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                logging.error(f"{where}: Gradient of {name} has NaN")
                stop_flag = True
            if torch.isinf(param.grad).any():
                logging.error(f"{where}: Gradient of {name} has infinite")
                stop_flag = True
    if stop_flag:
        raise

def debug_finite_dict(dic, what=""):
    stop_flag = False
    if dic is not None:
        for k, v in dic.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    logging.error(f"{what} Tensor with key {k} has NaN")
                    stop_flag = True
                if torch.isinf(v).any():
                    logging.error(f"{what} Dictionary with key {k} has infinite")
                    stop_flag = True
            elif isinstance(v, defaultdict):
                debug_finite_dict(v, what=f"{what} (neighbor)")
    if stop_flag:
        raise

def create_sp_graph(plot_node_idx, sp_bi_adj):
    """Create sp graph based on the sampled contig list and its neighbors,
    remember to reindex the contig id, cause igraph.Graph object indexing
    from 0 to self.plot_graph_size-1.

    Args:
        plot_node_idx (list): list of sampled contig ids.
        batch (dictionary): batch from datamodule to get the neighbors id.

    Returns:
        sp_graph (igraph.Graph): reindexed igraph.Graph object.
    """
    # sp_graph = nx.from_numpy_array(np.squeeze(sp_bi_adj), create_using=nx.DiGraph)
    sp_graph = nx.from_scipy_sparse_array(sp_bi_adj, create_using=nx.DiGraph)
    # print(sp_graph)
    sp_graph.remove_nodes_from([n for n in sp_graph if n not in plot_node_idx])
    # print(sp_graph)
    return sp_graph


def create_exp_graph(plot_node_idx, exp_vis_bi_adj):
    """Create exp graph based on the sampled contig list and its neighbors,
    remember to reindex the contig id, cause igraph.Graph object indexing
    from 0 to self.plot_graph_size-1.

    Args:
        plot_node_idx.
        batch (dictionary): batch from datamodule to get the neighbors id.

    Returns:
        exp_graph (igraph.Graph): reindexed igraph.Graph object.
    """

    # exp_graph = nx.from_numpy_array(np.squeeze(exp_vis_bi_adj), create_using=nx.DiGraph)
    exp_graph = nx.from_scipy_sparse_array(exp_vis_bi_adj, create_using=nx.DiGraph)
    # print(exp_graph)
    exp_graph.remove_nodes_from([n for n in exp_graph if n not in plot_node_idx])
    # print(exp_graph)
    return exp_graph

def plot_graph(graph, log_path, current_epoch, graph_type, labels):
    """Plot graph to disk.

    Args:
        graph (igraph.Graph): igraph.Graph object.
        log_path (string): predefined path to store the visualized image.
        graph_type (string): graph type to store.
        labels: labels of the graph.

    Returns:
        path (string): output path of plotting grpah.
    """
    import igraph as ig
    relative_path = "{}_{}.png".format(graph_type, current_epoch)
    path = osp.join(log_path, relative_path)
    # fig, ax = plt.subplots()
    # nx.draw_networkx(graph, ax=ax, with_labels=False, node_size=300, node_color=labels[plot_node_idx])
    # nx.draw(graph, ax=ax, node_size=300, node_color=labels[plot_node_idx])
    # fig.savefig(path, dpi=300)
    # print("converting to igraph...")
    igraph = ig.Graph.from_networkx(graph)
    # print("converting to igraph finished")
    # print(igraph.get_vertex_dataframe().head())
    update_graph_labels(igraph, labels[igraph.vs["_nx_name"]])
    # layout = igraph.layout_fruchterman_reingold()
    layout = igraph.layout_auto(dim=2)
    visual_style = {}
    visual_style["layout"] = layout
    visual_style["vertex_size"] = 10
    visual_style["bbox"] = (800, 800)
    # print("plotting...")
    ig.plot(igraph, path, **visual_style)
    # print("plotting finished")
    return path

def log_sp_graph(
    plot_node_idx,
    sp_bi_adj,
    log_path,
    current_epoch,
    gt_labels,
    pred_labels,
    plot_gt=True
):
    """Wrapper function inside validation step, plot graph to disk.

    Args:
        plot_node_idx.
        processed_zarr_dataset_path (string): path of processed zarr dataset.
        plotting_contig_list (list): list of sampled contig ids.
        graph (igraph.Graph): igraph.Graph object created from plotting contig list.
        log_path (string): predefined path to store the visualized image.
        gt_labels: ground truth labels.
        pred_labels: predicted labels.

    Returns:
        gt_sp_graph_path (string): output path of plotting gt_sp_grpah.
        pred_sp_graph_path (string): output path of plotting pred_sp_grpah.
    """
    if plot_gt:
        gt_sp_graph = create_sp_graph(
            plot_node_idx, sp_bi_adj
        )

        gt_sp_graph_path = plot_graph(
            graph=gt_sp_graph,
            log_path=log_path,
            current_epoch=current_epoch,
            graph_type="gt_sp",
            labels=gt_labels,
        )
    else:
        gt_sp_graph_path = None

    pred_sp_graph = create_sp_graph(
        plot_node_idx, sp_bi_adj
    )

    pred_sp_graph_path = plot_graph(
        graph=pred_sp_graph,
        log_path=log_path,
        current_epoch=current_epoch,
        graph_type="pred_sp",
        labels=pred_labels,
    )
    return gt_sp_graph_path, pred_sp_graph_path


def log_exp_graph(
    plot_node_idx,
    exp_vis_bi_adj,
    log_path,
    current_epoch,
    gt_labels,
    pred_labels,
    plot_gt=True
):
    """Wrapper function inside validation step, plot graph to disk.

    Args:
        plot_node_idx.
        plotting_contig_list (list): list of sampled contig ids.
        k (int): k neighbors used in exp graph.
        batch (dictionary):  batch from datamodule to get the neighbors id.
        log_path (string): predefined path to store the visualized image.
        gt_labels: ground truth labels.
        pred_labels: predicted labels.

    Returns:
        gt_exp_graph_path (string): output path of plotting gt_exp_grpah.
        pred_exp_graph_path (string): output path of plotting pred_exp_grpah.
    """
    if plot_gt:
        gt_exp_graph = create_exp_graph(
            plot_node_idx=plot_node_idx,
            exp_vis_bi_adj=exp_vis_bi_adj,
        )
        gt_exp_graph_path = plot_graph(
            graph=gt_exp_graph,
            log_path=log_path,
            current_epoch=current_epoch,
            graph_type="gt_exp",
            labels=gt_labels,
        )
    else:
        gt_exp_graph_path = None

    pred_exp_graph = create_exp_graph(
        plot_node_idx=plot_node_idx,
        exp_vis_bi_adj=exp_vis_bi_adj,
    )

    pred_exp_graph_path = plot_graph(
        graph=pred_exp_graph,
        log_path=log_path,
        current_epoch=current_epoch,
        graph_type="pred_exp",
        labels=pred_labels,
    )
    return gt_exp_graph_path, pred_exp_graph_path


def log_tissue_graph(
    data_val,
    log_path,
    current_epoch,
    gt_labels,
    pred_labels,
    plot_gt=True,
):
    """Wrapper function inside validation step, plot graph to disk.

    Args:
        log_path (string): predefined path to store the visualized image.
        gt_labels: ground truth labels.
        pred_labels: predicted labels.

    Returns:
        gt_exp_graph_path (string): output path of plotting gt_exp_grpah.
        pred_exp_graph_path (string): output path of plotting pred_exp_grpah.
    """
    if data_val.full_adata:
        adata = data_val.full_adata
    else:
        adata = data_val.adata
    data_type = data_val.data_type
    annotation_key = data_val.annotation_key
    if plot_gt:
        graph_type="gt_tissue"
        relative_path = "{}_{}.png".format(graph_type, current_epoch)
        gt_tissue_graph_path = osp.join(log_path, relative_path)
        adata.obs["gt_labels"] = pd.Categorical(gt_labels)
        if data_type == "10x" and data_val.sample_id != "all" and (not data_val.sample_id.startswith("GSM483813")) and data_val.sample_id != "zebrafish_tumor" and adata.obsm["spatial"].shape[1] == 2:
            # ax_dict = sc.pl.spatial(adata, img_key="hires", color="gt_labels", show=False)
            # ax_dict = sc.pl.spatial(adata, img_key="lowres", color="gt_labels", alpha_img=0., show=False)
            ax_dict = sc.pl.spatial(adata, img_key="lowres", color="gt_labels", show=False)
        else:
            if adata.obsm["spatial"].shape[1] == 2:
                scatter_df_cols = ["x", "y"]
            elif adata.obsm["spatial"].shape[1] == 3:
                scatter_df_cols = ["x", "y", "z"]
            scatter_df = pd.DataFrame(adata.obsm["spatial"], columns=scatter_df_cols)
            if "annotation_colors" in adata.uns:
                try:
                    scatter_df[f"{annotation_key}"] = adata.obs[f"{annotation_key}"].cat.rename_categories(adata.uns["annotation_colors"]).values
                except:
                    # scatter_df[f"{annotation_key}"] = adata.obs[f"{annotation_key}"].map(dict(zip(adata.obs[f"{annotation_key}"].cat.categories, adata.uns["annotation_colors"])))
                    unique_color = set(adata.uns["annotation_colors"].tolist())
                    if len(unique_color) < len(adata.obs[f"{annotation_key}"].cat.categories):
                        if len(adata.obs[f"{annotation_key}"].cat.categories) - len(unique_color) == 1:
                            if "#ff000000" not in unique_color: # balck
                                unique_color.add("#ff000000")
                            elif "#ffffffff" not in unique_color: # white
                                unique_color.add("#ffffffff")
                            else:
                                raise NotImplementedError
                        elif len(adata.obs[f"{annotation_key}"].cat.categories) - len(unique_color) == 2:
                            assert "#ff000000" not in unique_color and "#ffffffff" not in unique_color
                            unique_color.add("#ff000000")
                            unique_color.add("#ffffffff")
                        else:
                            raise NotImplementedError
                    scatter_df[f"{annotation_key}"] = adata.obs[f"{annotation_key}"].cat.rename_categories(list(unique_color)).values
            else:
                tmp_anno_series = adata.obs[f"{annotation_key}"].cat.add_categories("Unknown")
                tmp_anno_series = tmp_anno_series.fillna("Unknown")
                scatter_df[f"{annotation_key}"] = tmp_anno_series.cat.rename_categories(COLOUR_DICT[:len(tmp_anno_series.value_counts())]).values
            if data_val.sample_id.startswith("GSM483813") or data_val.sample_id == "zebrafish_tumor":
                scatter_size = 0.3
            else:
                scatter_size = 0.1
            if adata.obsm["spatial"].shape[1] == 2:
                plt.scatter(scatter_df["x"], scatter_df["y"], c=scatter_df[f"{annotation_key}"], s=scatter_size)
            elif adata.obsm["spatial"].shape[1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(scatter_df["x"], scatter_df["y"], scatter_df["z"], c=scatter_df[f"{annotation_key}"], s=scatter_size)
            plt.axis("equal")
            if data_type == "10x":
                plt.gca().invert_yaxis()
        sc_fig = plt.gcf()
        sc_fig.savefig(gt_tissue_graph_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        gt_tissue_graph_path = None

    graph_type="pred_tissue"
    relative_path = "{}_{}.png".format(graph_type, current_epoch)
    pred_tissue_graph_path = osp.join(log_path, relative_path)
    adata.obs["plot_pred_labels"] = pd.Categorical(pred_labels)
    if data_type == "10x" and data_val.sample_id != "all" and (not data_val.sample_id.startswith("GSM483813")) and data_val.sample_id != "zebrafish_tumor" and adata.obsm["spatial"].shape[1] == 2:
        # ax_dict = sc.pl.spatial(adata, img_key="hires", color="plot_pred_labels", show=False)
        # ax_dict = sc.pl.spatial(adata, img_key="lowres", color="plot_pred_labels", alpha_img=0., show=False)
        ax_dict = sc.pl.spatial(adata, img_key="lowres", color="plot_pred_labels", title=f"Epoch {current_epoch}", show=False)
    else:
        if adata.obsm["spatial"].shape[1] == 2:
            scatter_df_cols = ["x", "y"]
        elif adata.obsm["spatial"].shape[1] == 3:
            scatter_df_cols = ["x", "y", "z"]
        scatter_df = pd.DataFrame(adata.obsm["spatial"], columns=scatter_df_cols)
        pred_cluster_num = len(np.unique(pred_labels))
        if "annotation_colors" in adata.uns:
            try:
                scatter_df[f"{annotation_key}"] = adata.obs["plot_pred_labels"].cat.rename_categories(adata.uns["annotation_colors"][:pred_cluster_num]).values
            except:
                # scatter_df[f"{annotation_key}"] = adata.obs["plot_pred_labels"].map(dict(zip(adata.obs["plot_pred_labels"].cat.categories, adata.uns["annotation_colors"])))
                unique_color = set(adata.uns["annotation_colors"].tolist())
                if len(unique_color) < len(adata.obs[f"{annotation_key}"].cat.categories):
                    if len(adata.obs[f"{annotation_key}"].cat.categories) - len(unique_color) == 1:
                        if "#ff000000" not in unique_color: # balck
                            unique_color.add("#ff000000")
                        elif "#ffffffff" not in unique_color: # white
                            unique_color.add("#ffffffff")
                        else:
                            raise NotImplementedError
                    elif len(adata.obs[f"{annotation_key}"].cat.categories) - len(unique_color) == 2:
                        assert "#ff000000" not in unique_color and "#ffffffff" not in unique_color
                        unique_color.add("#ff000000")
                        unique_color.add("#ffffffff")
                    else:
                        raise NotImplementedError
                scatter_df[f"{annotation_key}"] = adata.obs["plot_pred_labels"].cat.rename_categories(list(unique_color)).values
        else:
            scatter_df[f"{annotation_key}"] = adata.obs["plot_pred_labels"].cat.rename_categories(COLOUR_DICT[:pred_cluster_num]).values
        if data_val.sample_id.startswith("GSM483813") or data_val.sample_id == "zebrafish_tumor":
            scatter_size = 0.3
        else:
            scatter_size = 0.1
        if adata.obsm["spatial"].shape[1] == 2:
            plt.scatter(scatter_df["x"], scatter_df["y"], c=scatter_df[f"{annotation_key}"], s=scatter_size)
        elif adata.obsm["spatial"].shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(scatter_df["x"], scatter_df["y"], scatter_df["z"], c=scatter_df[f"{annotation_key}"], s=scatter_size)
        plt.axis("equal")
        if data_type == "10x":
            plt.gca().invert_yaxis()
    sc_fig = plt.gcf()
    sc_fig.savefig(pred_tissue_graph_path, dpi=300, bbox_inches="tight")
    plt.close()

    return gt_tissue_graph_path, pred_tissue_graph_path

def get_neighbor_domain(adata, this_domain_unit_indices, domain_key="pred_labels", neighbor_key="sp_k"):
    return adata.obs[domain_key].values[adata[this_domain_unit_indices].obsm[neighbor_key].reshape(-1)].reshape(adata[this_domain_unit_indices].obsm[neighbor_key].shape)

def get_domain_boundary_unit_indices(adata, domain_idx, domain_key="pred_labels", neighbor_key="sp_k"):
    this_domain_unit_indices = adata.obs[domain_key] == domain_idx
    this_domain_unit_num_indices = np.where(this_domain_unit_indices)[0]
    # A x K -> A
    domain_boundary_unit_indices = np.where((get_neighbor_domain(adata, this_domain_unit_indices) ==
                                             adata[this_domain_unit_indices].obs[domain_key].values.reshape(-1, 1)
                                             ).sum(1) != adata.obsm[neighbor_key].shape[1])[0]
    domain_boundary_unit_indices = this_domain_unit_num_indices[domain_boundary_unit_indices]
    return domain_boundary_unit_indices

# https://stackoverflow.com/questions/40591754/vectorizing-numpy-bincount
def batched_bincount_for_row(x):
    m = x.shape[0]
    n = x.max() + 1
    if isinstance(x, torch.Tensor):
        x1 = x + (n * torch.arange(m).reshape(m, 1).to(x.device))
        return torch.bincount(x1.ravel(), minlength=n * m).reshape(-1, n)
    elif isinstance(x, np.ndarray):
        m = x.shape[0]
        n = x.max() + 1
        x1 = x + (n * np.arange(m).reshape(m, 1))
        return np.bincount(x1.ravel(), minlength=n * m).reshape(-1, n)

def batched_bincount_for_col(x):
    m = x.shape[1]
    n = x.max() + 1
    if isinstance(x, torch.Tensor):
        x1 = x + (n * torch.arange(m).to(x.device))
        return torch.bincount(x1.ravel(), minlength=n * m).reshape(m, -1).T
    elif isinstance(x, np.ndarray):
        m = x.shape[1]
        n = x.max() + 1
        x1 = x + (n * np.arange(m))
        return np.bincount(x1.ravel(), minlength=n * m).reshape(m, -1).T

def int2onehot(x):
    onehot_arr = np.zeros((x.size, x.max() + 1))
    onehot_arr[np.arange(x.size), x] = 1
    return onehot_arr

# def _downsample_array(
#     one_row: np.ndarray,
#     target: int,
#     rng=None,
#     replace: bool = True,
# ):
#     """\
#     Evenly reduce/increase counts in cell to target amount.
#     """
#     cumcounts = np.cumsum(one_row)
#     new_row = np.zeros_like(one_row)
#     total = np.int_(cumcounts[-1])
#     sample = rng.choice(total, target, replace=replace)
#     sample.sort()
#     geneptr = 0
#     for count in sample:
#         while count >= cumcounts[geneptr]:
#             geneptr += 1
#         new_row[geneptr] += 1
#     return new_row

# @numba.njit(parallel=True, cache=True)
# @numba.jit(forceobj=True)
# @numba.jit
@numba.jit(cache=True)
def downsample_array(
    one_row: np.ndarray,
    target: int,
    rng=None,
    replace: bool = True,
):
    """\
    Evenly reduce/increase counts in cell to target amount.
    """
    cumcounts = np.cumsum(one_row)
    new_row = np.zeros_like(one_row)
    total = np.int_(cumcounts[-1])
    sample = np.random.choice(total, target, replace=replace)
    sample.sort()
    geneptr = 0
    for count in sample:
        while count >= cumcounts[geneptr]:
            geneptr += 1
        new_row[geneptr] += 1
    return new_row

def resample_per_cell(X, target, rng, replace=True):
    new_X = np.empty_like(X)
    for i in range(X.shape[0]):
        new_X[i] = downsample_array(X[i], target, rng, replace)
    return new_X

def describe_graph(graph_type: str, graph: nx.Graph):
    graph_nodes = graph.number_of_nodes()
    graph_edges = graph.number_of_edges()
    print("{} graph details: nodes: {}; edges: {}".format(
        graph_type, graph_nodes, graph_edges
    ))

def get_mul_mask_matrix(adj_matrix):
    """Get multiplication mask matrix from adjacency matrix;
    M_ij = {
        0 if A_ij = 0;
        1 if A_ij != 0;
    }

    Args:
        adj_matrix (np.array): adjacency matrix from graph.

    Returns:
        mask_matrix (np.array): mask matrix.
    """
    mask_matrix = np.zeros_like(adj_matrix)
    mask_matrix[adj_matrix != 0] = 1

    return mask_matrix

def get_add_mask_matrix(adj_matrix):
    """Get addition mask matrix from adjacency matrix;
    M_ij = {
        -inf if A_ij = 0;
        0 if A_ij != 0;
    }

    Args:
        adj_matrix (np.array): adjacency matrix from graph.

    Returns:
        mask_matrix (np.array): mask matrix.
    """
    mask_matrix = np.zeros_like(adj_matrix)
    mask_matrix[adj_matrix == 0] = np.NINF

    return mask_matrix

def normalize_adjacency_matrix(adj_matrix):
    """Normalize adjacency matrix using the formulation from GCN;
    A_norm = D(-1/2) @ A @ D(-1/2).

    Args:
        adj_matrix (scipy.csr_matrix): adjacency matrix from graph.

    Returns:
        adj_normalized_matrix (np.array single precision): normalized adjacency
            matrix could cause out of memory issue.
    """
    D = np.diag(np.sum(adj_matrix, axis=1))
    D_half_norm = fractional_matrix_power(D, -0.5)
    normalized_adj_matrix = D_half_norm.dot(adj_matrix).dot(D_half_norm)
    normalized_adj_matrix = normalized_adj_matrix.astype(np.single)

    return normalized_adj_matrix

def add_self_loop(adj_matrix):
    loop_adj_matrix = adj_matrix.copy()
    np.fill_diagonal(loop_adj_matrix, 1 - np.diag(loop_adj_matrix))

    return loop_adj_matrix.astype(np.single)

def cut_adjacency_matrix(adj_matrix):
    cut_adj_matrix = adj_matrix.copy()
    assert (adj_matrix < 0).sum() == (adj_matrix > 1).sum() == 0
    cut_adj_matrix[adj_matrix > 0.9] = 0.9
    cut_adj_matrix[adj_matrix < 0.1] = 0.1

    return cut_adj_matrix.astype(np.single)
