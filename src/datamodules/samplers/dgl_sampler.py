import dgl
from dgl import NID, EID, to_block

class MyKNNMultiLayerNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts):
        super().__init__()
        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        previous_g_device = g.device
        # print("sample_frontier device", previous_g_device)
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            # frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            frontier = dgl.sampling.select_topk(g.cpu(), k=fanout, weight="sp_dist", nodes=seed_nodes.cpu(), ascending=True).to(previous_g_device)
        return frontier

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        previous_g_device = g.device
        # print("sample_blocks device start", previous_g_device)
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            # frontier = g.sample_neighbors(
            #     seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
            #     replace=self.replace, output_device=self.output_device,
            #     exclude_edges=exclude_eids)
            # print("fanout", fanout)
            # print("seed_nodes", seed_nodes)
            frontier = dgl.sampling.select_topk(g.cpu(), k=fanout, weight="sp_dist", nodes=seed_nodes.cpu(), ascending=True).to(previous_g_device)
            # print("sample_blocks device in iter", previous_g_device)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        # print("sample_blocks device stop", previous_g_device)

        return seed_nodes, output_nodes, blocks