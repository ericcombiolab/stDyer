import logging
import os
import random
import torch
import dgl
import pytorch_lightning as pl
from src.datamodules.datasets.graph_dataset import GraphDataset
from src.datamodules.datasets.dgl_dataset import MyDGLDataset
from src.datamodules.samplers.dgl_sampler import MyKNNMultiLayerNeighborSampler
from dgl.dataloading import DataLoader as DGLDataLoader
from torch.utils.data import DataLoader
from src.utils.states import Dynamic_neigh_level

class GMVAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        dataset_dir,
        data_type,
        in_type,
        out_type,
        compared_type,
        train_val_test_split,
        data_file_name=None,
        count_key=None,
        num_classes="auto",
        num_hvg=2048,
        lib_norm=True,
        n_pc=50,
        batch_size=256,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=False,
        rec_neigh_num=None,
        rec_mask_neigh_threshold=None,
        use_cell_mask=False,
        max_dynamic_neigh=False,
        dynamic_neigh_level=Dynamic_neigh_level.unit,
        unit_fix_num=None,
        unit_dynamic_num=None,
        k=18,
        test_with_gt_sp=False,
        forward_neigh_num=None,
        gat_layer_num=0,
        exchange_forward_neighbor_order=False,
        sample_id="sample_id",
        multi_slides=False,
        weighted_neigh=False,
        keep_tiles=False,
        supervise_cat=False,
        seed=42,
        device="auto",
        load_whole_graph_on_gpu=False,
        z_scale=2.,
        recreate_dgl_dataset=False,
        n_jobs="mean",
        use_ddp=False,
        **kwargs,
    ):
        """DataModule of GMGATModel, specify the dataloaders of

        Args:
            graph_dataset_roots (list): list of the graph datasets path.
        """
        super().__init__()
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.data_type = data_type
        self.in_type = in_type
        self.out_type = out_type
        self.compared_type = compared_type
        self.count_key = count_key
        self.train_val_test_split = train_val_test_split
        self.data_file_name = data_file_name
        self.num_classes = num_classes
        self.num_hvg = num_hvg
        self.lib_norm = lib_norm
        self.n_pc = n_pc
        self.batch_size = batch_size
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.max_dynamic_neigh = max_dynamic_neigh
        self.dynamic_neigh_level = Dynamic_neigh_level[dynamic_neigh_level]
        self.unit_fix_num = unit_fix_num
        self.unit_dynamic_num = unit_dynamic_num
        self.start_use_domain_neigh = False
        self.kept_val_dataloader = None
        self.kept_test_dataloader = None
        self.k = k
        if rec_neigh_num is not None:
            self.rec_neigh_num = rec_neigh_num
        else:
            self.rec_neigh_num = k
        self.rec_mask_neigh_threshold = rec_mask_neigh_threshold
        if forward_neigh_num is not None:
            self.forward_neigh_num = forward_neigh_num
        else:
            self.forward_neigh_num = k
        self.gat_layer_num = gat_layer_num
        self.use_cell_mask = use_cell_mask
        self.test_with_gt_sp = test_with_gt_sp
        self.exchange_forward_neighbor_order = exchange_forward_neighbor_order
        self.sample_id = str(sample_id)
        if self.sample_id.startswith("c_"):
            self.sample_id = self.sample_id[2:]
        self.multi_slides = multi_slides
        self.weighted_neigh = weighted_neigh
        self.keep_tiles = keep_tiles
        self.supervise_cat = supervise_cat
        self.seed = seed if seed is not None else random.randint(0, 100000)
        self.load_whole_graph_on_gpu = load_whole_graph_on_gpu
        self.use_ddp = use_ddp
        self.z_scale = z_scale
        self.recreate_dgl_dataset = recreate_dgl_dataset
        if n_jobs == "mean":
            try:
                num_gpu = int(os.popen("nvidia-smi|grep Default|wc -l").read().strip())
                self.n_jobs = os.cpu_count() // num_gpu
            except:
                self.n_jobs = os.cpu_count()
        elif n_jobs == "all":
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        if num_workers == "mean":
            try:
                num_gpu = int(os.popen("nvidia-smi|grep Default|wc -l").read().strip())
                self.num_workers = os.cpu_count() // num_gpu
            except:
                self.num_workers = os.cpu_count()
        elif num_workers == "all":
            self.num_workers = os.cpu_count()
        else:
            self.num_workers = num_workers
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.max_dynamic_neigh:
            if self.device.type.startswith("cuda"):
                self.num_workers = 0
                self.dgl_device = torch.device("cuda")
                self.persistent_workers = False
                if not self.load_whole_graph_on_gpu:
                    self.use_uva = True
                else:
                    self.use_uva = False
            else:
                self.dgl_device = torch.device("cpu")
                self.use_uva = False

    def prepare_data(self):
        pass

    def create_dgl_dataset(self):
        self.dgl_train_dataset = MyDGLDataset(
            in_type=self.in_type,
            out_type=self.out_type,
            count_key=self.data_val.count_key,
            sample_id=self.sample_id,
            dynamic_neigh_nums=self.data_val.dynamic_neigh_nums,
            dynamic_neigh_level=self.dynamic_neigh_level,
            unit_fix_num=self.unit_fix_num,
            unit_dynamic_num=self.unit_dynamic_num,
            start_use_domain_neigh=self.start_use_domain_neigh,
            adata=self.data_val.adata,
            load_whole_graph_on_gpu=self.load_whole_graph_on_gpu,
            seed=self.seed,
        )

    def setup(self, stage=None):
        train_dataset = GraphDataset(
            data_dir=self.data_dir,
            dataset_dir=self.dataset_dir,
            data_type=self.data_type,
            in_type=self.in_type,
            out_type=self.out_type,
            compared_type=self.compared_type,
            count_key=self.count_key,
            num_classes=self.num_classes,
            data_file_name=self.data_file_name,
            num_hvg=self.num_hvg,
            lib_norm=self.lib_norm,
            n_pc=self.n_pc,
            max_dynamic_neigh=self.max_dynamic_neigh,
            dynamic_neigh_level=self.dynamic_neigh_level,
            unit_fix_num=self.unit_fix_num,
            unit_dynamic_num=self.unit_dynamic_num,
            k=self.k,
            rec_neigh_num=self.rec_neigh_num,
            rec_mask_neigh_threshold=self.rec_mask_neigh_threshold,
            use_cell_mask=self.use_cell_mask,
            test_with_gt_sp=self.test_with_gt_sp,
            forward_neigh_num=self.forward_neigh_num,
            exchange_forward_neighbor_order=self.exchange_forward_neighbor_order,
            sample_id=self.sample_id,
            multi_slides=self.multi_slides,
            weighted_neigh=self.weighted_neigh,
            keep_tiles=self.keep_tiles,
            supervise_cat=self.supervise_cat,
            z_scale=self.z_scale,
            use_ddp=self.use_ddp,
            device=self.device,
            seed=self.seed,
        )
        self.data_train = train_dataset
        test_dataset = train_dataset
        self.data_val = test_dataset
        self.data_test = test_dataset
        if self.max_dynamic_neigh:
            self.create_dgl_dataset()
            self.dgl_indices = torch.arange(self.dgl_train_dataset[0].num_nodes(), device=self.dgl_device)
            if self.forward_neigh_num:
                # assert self.forward_neigh_num == self.rec_neigh_num
                self.train_sampler = MyKNNMultiLayerNeighborSampler(fanouts=[self.rec_neigh_num, self.forward_neigh_num])
                self.val_test_sampler = MyKNNMultiLayerNeighborSampler(fanouts=[self.forward_neigh_num])
            else:
                self.train_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                self.val_test_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        if self.batch_size == "all":
            self.batch_size = len(self.data_val.adata)
        self.train_drop_last = True if self.batch_size <= len(self.data_val.adata) else False
        if self.trainer.is_global_zero:
            logging.info(self.data_train.adata)

    def train_dataloader(self):
        if self.max_dynamic_neigh:
            if self.recreate_dgl_dataset:
                self.create_dgl_dataset()
            return DGLDataLoader(
                self.dgl_train_dataset[0],
                self.dgl_indices,
                self.train_sampler,
                device=self.dgl_device,
                use_uva=self.use_uva,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=self.train_drop_last,
                persistent_workers=False,
                use_prefetch_thread=False,
                shuffle=True,
                use_ddp=self.use_ddp,
            )
        else:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.pin_memory,
                drop_last=self.train_drop_last,
                shuffle=True,
            )

    def val_dataloader(self):
        # print("start val_dataloader")
        if self.max_dynamic_neigh:
            if self.forward_neigh_num:
                return DGLDataLoader(
                    self.dgl_train_dataset[0],
                    self.dgl_indices,
                    self.val_test_sampler,
                    device=self.dgl_device,
                    use_uva=self.use_uva,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=False,
                    use_prefetch_thread=False,
                    shuffle=False,
                    use_ddp=self.use_ddp,
                )
            else:
                if self.kept_val_dataloader is None:
                    self.kept_val_dataloader = DGLDataLoader(
                        self.dgl_train_dataset[0],
                        self.dgl_indices,
                        self.val_test_sampler,
                        device=self.dgl_device,
                        use_uva=self.use_uva,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=self.persistent_workers,
                        shuffle=False,
                        use_ddp=self.use_ddp,
                    )
        else:
            self.kept_val_dataloader = DataLoader(
                dataset=self.data_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        # print("stop val_dataloader")
        return self.kept_val_dataloader

    def test_dataloader(self):
        if self.max_dynamic_neigh:
            if self.forward_neigh_num:
                return DGLDataLoader(
                    self.dgl_train_dataset[0],
                    self.dgl_indices,
                    self.val_test_sampler,
                    device=self.dgl_device,
                    use_uva=self.use_uva,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=False,
                    use_prefetch_thread=False,
                    shuffle=False,
                    use_ddp=self.use_ddp,
                )
            else:
                if self.kept_test_dataloader is None:
                    self.kept_test_dataloader = DGLDataLoader(
                        self.dgl_train_dataset[0],
                        self.dgl_indices,
                        self.val_test_sampler,
                        device=self.dgl_device,
                        use_uva=self.use_uva,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=self.persistent_workers,
                        shuffle=False,
                        use_ddp=self.use_ddp,
                    )
        else:
            self.kept_test_dataloader = DataLoader(
                dataset=self.data_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        return self.kept_test_dataloader
