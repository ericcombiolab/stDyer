import torch
from torch import nn
from torch.nn import functional as F
from src.models.modules.layers import (
    GumbelSoftmax,
    Gaussian,
    DenseBlock,
)
from dgl.heterograph import DGLBlock
import dgl.function as fn

# import dgl

class BoundedAct(nn.Module):
    def __init__(self, activation_func, lower_bound=None, upper_bound=None, symmetric="auto"):
        super(BoundedAct, self).__init__()
        self.activation_func = activation_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if symmetric != "auto":
            self.symmetric = symmetric
        elif activation_func is nn.Sigmoid:
            self.symmetric = True
        elif activation_func is nn.Softplus:
            self.symmetric = False
        else:
            raise ValueError("activation_func must be either nn.Sigmoid or nn.Softplus")

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        if self.activation_func is nn.Sigmoid:
            if self.symmetric:
                # print("x.sum()", x.sum())
                # raise
                return self.lower_bound + (1 - 2 * self.lower_bound) * self.activation_func()(x)
            else:
                return self.lower_bound + (self.upper_bound - self.lower_bound) * self.activation_func()(x)
        elif self.activation_func is nn.Softplus:
            if self.symmetric:
                raise ValueError("Softplus is not symmetric")
            else:
                return self.lower_bound + self.activation_func()(x)

class Encoder(nn.Module):
    """Encoder model which encodes the logits y and latent vector z.
    Currently support 'Dense' block during message passing step.

    Args:
        y_dim (int): dimension of the gaussian mixture.
        z_dim (int): dimension of feature latent vector.
        in_channels (list): encoder input channels list.
        out_channels (list): encoder output channels list.
        y_block_type (string): encoder_y block type, support 'Dense'.
        z_block_type (string): encoder_z block type, support 'Dense'.
        use_bias (boolean): whether use bias in models.
        num_heads (int): number of heads in graph transformer block.
    """
    def __init__(
        self,
        y_dim,
        z_dim,
        attention_dim=128,
        onehot_embed_size=None,
        in_channels=[],
        out_channels=[],
        y_block_type="Dense",
        z_block_type="Dense",
        use_bias=True,
        num_heads=1,
        activation=nn.ReLU(),
        max_mu=10.,
        max_logvar=5.,
        min_logvar=-5.,
        forward_neigh_num=False,
        dropout=0.,
        use_batch_norm=False,
        add_cat_bias=False,
        GMM_model_name="VVI",
        use_kl_bn=False,
        kl_bn_gamma=32.,
        adata=None,
        device=None,
        legacy=False,
    ):
        super(Encoder, self).__init__()
        self.device = device
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.attention_dim = attention_dim
        self.onehot_embed_size = out_channels[-1] if onehot_embed_size is None else onehot_embed_size
        self.scale_factor = torch.sqrt(torch.tensor(attention_dim).float().to(self.device))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.y_block_type = y_block_type
        self.z_block_type = z_block_type
        self.use_bias = use_bias
        self.num_heads = num_heads
        self.activation = activation
        self.max_mu = max_mu
        self.max_logvar = max_logvar
        self.min_logvar = min_logvar
        self.forward_neigh_num = forward_neigh_num
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.add_cat_bias = add_cat_bias
        self.GMM_model_name = GMM_model_name
        self.use_kl_bn = use_kl_bn
        self.kl_bn_gamma = kl_bn_gamma
        self.adata = adata
        self.legacy = legacy
        self.ig_pc = None
        if self.forward_neigh_num:
            self.multiplying_power = 2
        else:
            self.multiplying_power = 1
        assert self.y_block_type in ["Dense", "STAGATE_v3_improved", "STAGATE_v2_improved3"]
        if self.z_block_type == "shared":
            assert self.forward_neigh_num > 0
        self.embed_input = torch.arange(self.y_dim, dtype=torch.long).reshape(-1, 1).to(self.device)
        self.encoder_y = torch.nn.ModuleList()
        self.encoder_z = torch.nn.ModuleList()
        # assert self.z_dim == self.out_channels[-1]
        assert len(self.in_channels) == len(self.out_channels)

        for index, item in enumerate(zip(self.in_channels, self.out_channels)):
            if index == 0:
                if self.y_block_type != "Dense":
                    dense_block = DenseBlock(
                                        in_features=item[0],
                                        out_features=item[1],
                                        use_bias=self.use_bias,
                                        use_batch_norm=self.use_batch_norm,
                                        activation=self.activation
                                    )
                    q_module = DenseBlock(
                                    in_features=item[1],
                                    out_features=self.attention_dim * self.num_heads,
                                    use_bias=self.use_bias,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=None
                                )
                    k_module = DenseBlock(
                                    in_features=item[1],
                                    out_features=self.attention_dim * self.num_heads,
                                    use_bias=self.use_bias,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=None
                                )
                    self.encoder_y.append(
                        torch.nn.ModuleList([
                            dense_block,
                            q_module,
                            k_module,
                        ])
                    )
                    if self.y_block_type == "STAGATE_v3_improved":
                        v_module = DenseBlock(
                                        in_features=item[1],
                                        out_features=item[1],
                                        use_bias=self.use_bias,
                                        use_batch_norm=self.use_batch_norm,
                                        activation=None
                                    )
                        self.encoder_y[-1].append(v_module)
                elif self.y_block_type == "Dense":
                    self.encoder_y.append(
                            DenseBlock(
                                in_features=item[0] * self.multiplying_power,
                                out_features=item[1],
                                use_bias=self.use_bias,
                                use_batch_norm=self.use_batch_norm,
                                activation=self.activation
                            )
                    )
            else:
                self.encoder_y.append(
                    DenseBlock(
                        in_features=item[0],
                        out_features=item[1],
                        use_bias=self.use_bias,
                        use_batch_norm=self.use_batch_norm,
                        activation=self.activation
                    )
                )
        # self.encoder_y.append(
        #     DenseBlock(
        #         in_features=item[1],
        #         out_features=item[1],
        #         use_bias=self.use_bias,
        #         use_batch_norm=self.use_batch_norm,
        #         activation=None
        #     )
        # )
        # add softmax function at last layer.
        self.encoder_y.append(
            GumbelSoftmax(self.out_channels[-1], self.y_dim, self.activation)
        )

        # encode latent vectors.
        # self.encoder_z.append(nn.BatchNorm1d(self.in_channels[0]+self.y_dim))
        # self.encoder_z.append(nn.BatchNorm1d(self.in_channels[0]))

        # for encoder_z: len(encoder_z) == len(self.in_channels) + 1(Dense) / 2(att)
        self.is_z_att = (((self.z_block_type != "Dense") and (self.z_block_type != "shared")) or \
                        ((self.y_block_type != "Dense") and (self.z_block_type == "shared")))
        for index, item in enumerate(zip(self.in_channels, self.out_channels)):
            if index == 0:
                if self.z_block_type == "shared":
                    self.encoder_z.append(torch.nn.ModuleList([]))
                elif self.z_block_type == "Dense":
                    self.encoder_z.append(
                        DenseBlock(
                            in_features=item[0] * self.multiplying_power + self.y_dim,
                            out_features=item[1],
                            use_bias=self.use_bias,
                            use_batch_norm=self.use_batch_norm,
                            activation=self.activation
                        )
                    )
                else:
                    dense_block = DenseBlock(
                                    in_features=item[0],
                                    out_features=item[1],
                                    use_bias=self.use_bias,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=self.activation
                                )
                    q_module = DenseBlock(
                                    in_features=item[1],
                                    out_features=self.attention_dim * self.num_heads,
                                    use_bias=self.use_bias,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=None
                                )
                    k_module = DenseBlock(
                                    in_features=item[1],
                                    out_features=self.attention_dim * self.num_heads,
                                    use_bias=self.use_bias,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=None
                                )
                    self.encoder_z.append(
                        torch.nn.ModuleList([
                            dense_block,
                            q_module,
                            k_module,
                        ])
                    )
                    if self.z_block_type.startswith("STAGATE_v3_improved"):
                        v_module = DenseBlock(
                                        in_features=item[1],
                                        out_features=item[1],
                                        use_bias=self.use_bias,
                                        use_batch_norm=self.use_batch_norm,
                                        activation=None
                                    )
                        self.encoder_z[-1].append(v_module)
            else:
                self.encoder_z.append(
                    DenseBlock(
                        in_features=item[0],
                        out_features=item[1],
                        use_bias=self.use_bias,
                        use_batch_norm=self.use_batch_norm,
                        activation=self.activation
                    )
                )
                if index == len(self.in_channels) - 1:
                    if self.is_z_att:
                        assert item[1] // 2 * 2 == item[1]
                        self.encoder_z.append(
                            torch.nn.ModuleList([
                                DenseBlock(
                                    in_features=item[1],
                                    out_features=item[1] // 2,
                                    use_bias=False,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=None
                                ),
                                DenseBlock(
                                    in_features=self.y_dim,
                                    out_features=item[1] // 2,
                                    use_bias=self.use_bias,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=None
                                ),
                                self.activation,
                                DenseBlock(
                                    in_features=item[1],
                                    out_features=item[1],
                                    use_bias=self.use_bias,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=None
                                ),
                            ])
                        )
        # add reparameterize.
        self.encoder_z.append(
            Gaussian(self.out_channels[-1], self.z_dim, self.max_mu, self.max_logvar, self.min_logvar, self.activation, kind=self.GMM_model_name, use_kl_bn=self.use_kl_bn, kl_bn_gamma=self.kl_bn_gamma, device=self.device)
        )

    def src_dot_dst(self, src_field, dst_field, out_field):
        def func(edges):
            return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
        return func

    def scaled_exp(self, field, scale_constant):
        def func(edges):
            # clamp for softmax numerical stability
            return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
        return func

    def propagate_attention(self, g, q="query", k="key", v="value"):
        # Compute attention score
        g.apply_edges(self.src_dot_dst(k, q, 'score'))
        g.apply_edges(self.scaled_exp('score', self.scale_factor))
        g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'score_sum'))
        g.apply_edges(lambda edges: {'score': edges.data['score'] / edges.dst['score_sum']})
        g.update_all(fn.u_mul_e(v, 'score', v), fn.sum(v, 'wv'))

    # @profile
    def encode_y(self, x, neighbor_x=None):
        common_embed = None
        if neighbor_x is not None and isinstance(neighbor_x, torch.Tensor):
            B, K, G = neighbor_x.shape
        num_layers = len(self.encoder_y)
        att_x = None
        for i, layer in enumerate(self.encoder_y):
            if i == num_layers - 1:
                logits, prob_cat, y = layer(concat_x)
                if self.add_cat_bias:
                    prob_cat = prob_cat + self.add_cat_bias
                    # prob_cat = torch.softmax(prob_cat, dim=1)
                # print(layer, logits.shape, prob.shape, y.shape)
            elif i == 0:
                if isinstance(x, torch.Tensor):
                    if neighbor_x is not None:
                        if self.y_block_type != "Dense":
                            # B x (K + 1) x G
                            neighbor_self_x = torch.cat([x.unsqueeze(1), neighbor_x], dim=1)
                            # B x (K + 1) x G
                            h_x_neighbor = layer[0](neighbor_self_x)
                            # # B x (K + 1) x G -> B x (K + 1) x N x D
                            # h_x_neighbor = layer[0](neighbor_self_x).view(B, K + 1, self.num_heads, -1)
                            # B x 1 x G -> B x 1 x N x D
                            hh_x = layer[1](h_x_neighbor[:, [0], :]).view(B, 1, self.num_heads, -1)
                            # B x (K + 1) x G -> B x (K + 1) x N x D
                            hh_x_neighbor = layer[2](h_x_neighbor).view(B, K + 1, self.num_heads, -1)
                            # print("hh_x.shape", hh_x.shape)
                            # print("hh_x_neighbor.shape", hh_x_neighbor.shape)
                            # B x 1 x N x D -> B x N x 1 x D
                            # B x (K + 1) x N x D -> B x N x D x (K + 1)
                            # (B x N x 1 x D, B x N x D x (K + 1)) -> B x N x 1 x (K + 1) -> B x (K + 1) x N x 1
                            e_x_neighbor = (torch.matmul(hh_x.transpose(1, 2), hh_x_neighbor.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) / self.scale_factor
                            # B x (K + 1) x N x 1
                            att_x = nn.Softmax(1)(e_x_neighbor)
                            # attn: B x (K + 1) x N x 1 -> B x N x 1 x (K + 1)
                            # value: B x (K + 1) x G -> B x (K + 1) x N x D -> B x N x (K + 1) x D
                            # (B x N x 1 x (K + 1), B x N x (K + 1) x D) -> B x N x 1 x D -> B x ND
                            # print("att_x.shape", att_x.shape, att_x.permute(0, 2, 3, 1).shape)
                            if self.y_block_type == "STAGATE_v3_improved":
                                concat_x = torch.matmul(att_x.permute(0, 2, 3, 1), (layer[3](h_x_neighbor)).reshape(B, K + 1, self.num_heads, -1).transpose(1, 2)).view(B, -1)
                            else:
                                concat_x = torch.matmul(att_x.permute(0, 2, 3, 1), h_x_neighbor.reshape(B, K + 1, self.num_heads, -1).transpose(1, 2)).view(B, -1)
                        else:
                            raise NotImplementedError
                        if self.z_block_type == "shared":
                            common_embed = concat_x.clone()
                    else:
                        concat_x = layer(x)
                elif isinstance(x, DGLBlock):
                    assert self.forward_neigh_num > 0 and self.y_block_type == "STAGATE_v2_improved3"
                    with x.local_scope():
                        # A x G
                        x.srcdata.update({'embed': layer[0](x.srcdata['exp_feature'])})
                        x.dstdata.update({'embed': layer[0](x.dstdata['exp_feature'])})
                        # B x G
                        x.dstdata.update({'query': layer[1](x.dstdata['embed'])})
                        x.srcdata.update({'key': layer[2](x.srcdata['embed'])})
                        # x.apply_edges(fn.v_dot_u('query', 'key', 'att'))
                        # x.edata["att"] = x.edata["att"] / self.scale_factor
                        self.propagate_attention(x, q="query", k="key", v="embed")
                        # x.apply_edges(fn.u_add_v('exp_feature', 'exp_feature', 'out'))
                        # x.apply_edges(lambda edges: {'neighbor_self_x': torch.cat([edges.src['exp_feature'], edges.dst['exp_feature']], -1)})
                        # x.edata["embed"] = layer[0](x.edata["neighbor_self_x"])
                        # print(x.ndata)
                        # print(x.edata)
                        # att_x = x.srcdata["score"]
                        att_x = None
                        concat_x = x.dstdata["wv"]
            else:
                concat_x = layer(concat_x)
        return logits, prob_cat, y, att_x, common_embed

    def encode_z(self, x, onehot_idx):
        C = self.y_dim
        if x.ndim == 3:
            # B x N x C
            B, N, G = x.shape
            onehot_y = torch.zeros((B, N, C), device=x.device, dtype=x.dtype)
            onehot_y[:, :, onehot_idx] = 1
            # B x N x (G + C) <- cat(B x N x G, B x N x C)
            concat = torch.cat((x, onehot_y), dim=-1) # i.e. dim=2
        elif x.ndim == 2:
            # B x C
            B, G = x.shape
            onehot_y = torch.zeros((B, C), device=x.device, dtype=x.dtype)
            onehot_y[:, onehot_idx] = 1
            # B x (G + C) <- cat(B x G, B x C)
            concat = torch.cat((x, onehot_y), dim=-1)

        num_layers = len(self.encoder_z)

        # print("concat.shape", concat.shape)
        for i, layer in enumerate(self.encoder_z):
            if i == num_layers - 1:
                mu, logvar, z = layer(concat)
            else:
                concat = layer(concat)

        # print("[mu.sum() for mu in mus]", [mu.sum() for mu in mus])
        # print("[logvar.sum() for logvar in logvars]", [logvar.sum() for logvar in logvars])
        # print("[z.sum() for z in zs]", [z.sum() for z in zs])
        # raise
        # B x N x D [B x D]
        return mu, logvar, z

    # @profile
    def encode_zs(self, x, neighbor_x=None, common_embed=None):
        if isinstance(neighbor_x, torch.Tensor):
            if neighbor_x is not None:
                B, K, G = neighbor_x.shape

        if isinstance(x, DGLBlock):
            onehot_ys = torch.zeros(self.y_dim, len(x.dstnodes()), self.y_dim, device=x.device, dtype=x.dstdata["exp_feature"].dtype)
        elif x.ndim == 2:
            # B x C -> C x B x C
            onehot_ys = torch.zeros(self.y_dim, x.shape[0], self.y_dim, device=x.device, dtype=x.dtype)
        elif x.ndim == 3:
            # B x N x C -> B x C x N x C
            onehot_ys = torch.zeros(x.shape[0], self.y_dim, x.shape[1], self.y_dim, device=x.device, dtype=x.dtype)

        num_layers = len(self.encoder_z)
        mus = []
        logvars = []
        zs = []
        var_invs = []
        for c in range(self.y_dim):
            if isinstance(x, DGLBlock) or x.ndim == 2:
                onehot_ys[c, :, c] = 1
                if not isinstance(x, DGLBlock):
                    # B x (G + C) <- cat(B x G, B x C)
                    concat_x = torch.cat((x, onehot_ys[c, :, :]), dim=-1)
            elif x.ndim == 3:
                raise NotImplementedError
                onehot_ys[:, c, :, c] = 1
                # B x N x (G + C) <- cat(B x N x G, B x N x C)
                concat = torch.cat((x, onehot_ys[:, c, :, :]), dim=-1) # i.e. dim=2
            # print("concat.shape", concat.shape)
            for i, layer in enumerate(self.encoder_z):
                if i == num_layers - 1:
                    if i == 1 and self.z_block_type != "Dense":
                        mu, logvar, z, var_inv = layer(reuse_concat_x)
                    else:
                        mu, logvar, z, var_inv = layer(concat_x)
                elif i == 0:
                    if neighbor_x is not None:
                        if self.z_block_type == "shared":
                            reuse_concat_x = common_embed
                        else:
                            if c == 0:
                                # B x (K + 1) x G
                                neighbor_self_x = torch.cat([x.unsqueeze(1), neighbor_x], dim=1)
                                # B x (K + 1) x G
                                h_x_neighbor = layer[0](neighbor_self_x)
                                # # B x (K + 1) x G -> B x (K + 1) x N x D
                                # h_x_neighbor = layer[0](neighbor_self_x).view(B, K + 1, self.num_heads, -1)
                                # B x 1 x G -> B x 1 x N x D
                                hh_x = layer[1](h_x_neighbor[:, [0], :]).view(B, 1, self.num_heads, -1)
                                # B x (K + 1) x G -> B x (K + 1) x N x D
                                hh_x_neighbor = layer[2](h_x_neighbor).view(B, K + 1, self.num_heads, -1)
                                # B x 1 x N x D2 -> B x N x 1 x D2
                                # B x (K + 1) x N x D2 -> B x N x D2 x (K + 1)
                                # (B x N x 1 x D2, B x N x D2 x (K + 1)) -> B x N x 1 x (K + 1) -> B x (K + 1) x N x 1
                                e_x_neighbor = (torch.matmul(hh_x.transpose(1, 2), hh_x_neighbor.permute(0, 2, 3, 1))).permute(0, 3, 1, 2) / self.scale_factor
                                # B x (K + 1) x N x 1
                                att_x = nn.Softmax(1)(e_x_neighbor)
                                # attn: B x (K + 1) x N x 1 -> B x N x 1 x (K + 1)
                                # value: B x (K + 1) x G -> B x (K + 1) x N x D -> B x N x (K + 1) x D
                                # (B x N x 1 x (K + 1), B x N x (K + 1) x D) -> B x N x 1 x D -> B x ND
                                if self.y_block_type == "STAGATE_v3_improved":
                                    concat_x = torch.matmul(att_x.permute(0, 2, 3, 1), (layer[3](h_x_neighbor)).reshape(B, K + 1, self.num_heads, -1).transpose(1, 2)).view(B, -1)
                                else:
                                    concat_x = torch.matmul(att_x.permute(0, 2, 3, 1), h_x_neighbor.reshape(B, K + 1, self.num_heads, -1).transpose(1, 2)).view(B, -1)
                                reuse_concat_x = concat_x.clone()
                    else:
                        if self.forward_neigh_num > 0:
                            if isinstance(x, DGLBlock):
                                assert self.forward_neigh_num > 0 and self.y_block_type == "STAGATE_v2_improved3"
                                with x.local_scope():
                                    # A x G
                                    x.srcdata.update({'embed': layer[0](x.srcdata['exp_feature'])})
                                    x.dstdata.update({'embed': layer[0](x.dstdata['exp_feature'])})
                                    # B x G
                                    x.dstdata.update({'query': layer[1](x.dstdata['embed'])})
                                    x.srcdata.update({'key': layer[2](x.srcdata['embed'])})
                                    # x.apply_edges(fn.v_dot_u('query', 'key', 'att'))
                                    # x.edata["att"] = x.edata["att"] / self.scale_factor
                                    self.propagate_attention(x, q="query", k="key", v="embed")
                                    # x.apply_edges(fn.u_add_v('exp_feature', 'exp_feature', 'out'))
                                    # x.apply_edges(lambda edges: {'neighbor_self_x': torch.cat([edges.src['exp_feature'], edges.dst['exp_feature']], -1)})
                                    # x.edata["embed"] = layer[0](x.edata["neighbor_self_x"])
                                    # print(x.ndata)
                                    # print(x.edata)
                                    # att_x = x.srcdata["score"]
                                    att_x = None
                                    concat_x = x.dstdata["wv"]
                                    reuse_concat_x = concat_x.clone()
                        else:
                            concat_x = layer(concat_x)
                else:
                    # combine onehot cluster information before last layer
                    if self.is_z_att and (i == num_layers - 2):
                        if i == 1:
                            concat_x = layer[3](layer[2](torch.cat((layer[0](reuse_concat_x), layer[1](onehot_ys[c, :, :].clone())), dim=1)))
                        else:
                            concat_x = layer[3](layer[2](torch.cat((layer[0](concat_x), layer[1](onehot_ys[c, :, :].clone())), dim=1)))
                    elif self.is_z_att:
                        if i == 1:
                            concat_x = layer(reuse_concat_x)
                        else:
                            concat_x = layer(concat_x)
                    else:
                        concat_x = layer(concat_x)
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)
            var_invs.append(var_inv)
        # print("[mu.sum() for mu in mus]", [mu.sum() for mu in mus])
        # print("[logvar.sum() for logvar in logvars]", [logvar.sum() for logvar in logvars])
        # print("[z.sum() for z in zs]", [z.sum() for z in zs])
        # raise
        # return torch.stack(mus, dim=0).transpose(0, 1), torch.stack(logvars, dim=0).transpose(0, 1), torch.stack(zs, dim=0).transpose(0, 1)

        if isinstance(x, DGLBlock) or x.ndim == 2:
            if var_invs[0] is not None:
                # C x B x D
                return torch.stack(mus, dim=0), torch.stack(logvars, dim=0), torch.stack(zs, dim=0), torch.stack(var_invs, dim=0)
            else:
                # C x B x D
                return torch.stack(mus, dim=0), torch.stack(logvars, dim=0), torch.stack(zs, dim=0), None
        elif x.ndim == 3:
            if var_invs[0] is not None:
                # B x C x N x D
                return torch.stack(mus, dim=1), torch.stack(logvars, dim=1), torch.stack(zs, dim=1), torch.stack(var_invs, dim=1)
            else:
                # B x C x N x D
                return torch.stack(mus, dim=1), torch.stack(logvars, dim=1), torch.stack(zs, dim=1), None
    # @profile
    def forward(self, x, neighbor_x=None, perform_ig=False):
        if perform_ig is False:
            logits, prob, y, att_x, common_embed = self.encode_y(
                x=x,
                neighbor_x=neighbor_x,
            )
            mus, logvars, zs, var_invs = self.encode_zs(
                x=x,
                neighbor_x=neighbor_x,
                common_embed=common_embed,
            )
        else:
            assert neighbor_x is None
            # if "pca" in self.adata.uns:
            #     if self.ig_pc is None:
            #         self.ig_pc = torch.from_numpy(self.adata.uns["pca"].components_.T.copy()).float().requires_grad_().to(x.device)
            #     x_pca = torch.matmul(x[:, 0, :], self.ig_pc)
            #     neighbor_x_pca = torch.matmul(x[:, 1:, :], self.ig_pc)
            #     # print("x_pca.shape", x_pca.shape)
            #     # print("neighbor_x_pca.shape", neighbor_x_pca.shape)
            #     logits, prob, y, att_x, common_embed = self.encode_y(
            #         x=x_pca,
            #         neighbor_x=neighbor_x_pca,
            #     )
            # else:
            x_extracted = x[:, 0, :]
            neighbor_x_extracted = x[:, 1:, :]
            logits, prob, y, att_x, common_embed = self.encode_y(
                x=x_extracted,
                neighbor_x=neighbor_x_extracted,
            )

        # print("y.sum(), mus.sum(), logvars.sum(), zs.sum(), logits.sum(), prob.sum()",
        #        y.sum(), mus.sum(), logvars.sum(), zs.sum(), logits.sum(), prob.sum())
        # raise
        if perform_ig is False:
            return y, mus, logvars, zs, var_invs, logits, prob, att_x
        else:
            return prob

class Decoder(nn.Module):
    """Decoder estimates distribution parameters based
    on latent vector z and estimate the mean and variance of y.

    Args:
        y_dim (int): gaussain mixture number.
        z_dim (int): latent vector dimension.
    """
    def __init__(self,
                 y_dim,
                 z_dim,
                 in_channels,
                 out_channels,
                 rec_type,
                 r_type='gene-wise',
                 activation=nn.ReLU(),
                 use_batch_norm=False,
                 prior_generator="fc",
                 GMM_model_name="VVI",
                 g_dim=50,
                 device=None,
                 legacy=False):
        super(Decoder, self).__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        # g_dim = self.out_channels[-1]

        self.rec_type = rec_type
        self.r_type = r_type
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.prior_generator = prior_generator
        self.GMM_model_name = GMM_model_name

        self.device = device
        self.legacy = legacy
        self.onehot_y = F.one_hot(torch.arange(self.y_dim), self.y_dim).float().to(self.device)

        # p(z|y)s
        if self.prior_generator.startswith("fc"):
            self.y_mus_1 = DenseBlock(y_dim, z_dim, activation=activation)
            self.y_mus_2 = nn.Linear(z_dim, z_dim)
            if self.GMM_model_name == "VVI":
                self.y_logvars_1 = DenseBlock(y_dim, z_dim, activation=activation)
                self.y_logvars_2 = nn.Linear(z_dim, z_dim)
        elif self.prior_generator.startswith("tensor"):
            self.mu_prior = nn.Parameter(torch.randn((y_dim, z_dim), device=self.device, requires_grad=True))
            if self.GMM_model_name == "VVI":
                self.logvar_prior = nn.Parameter(torch.randn((y_dim, z_dim), device=self.device, requires_grad=True))
            elif self.GMM_model_name == "EEE":
                self.logvar_prior = None

        # p(x|z)
        if self.rec_type == 'Bernoulli':
            self.generative_pxz = [
                nn.Linear(z_dim, self.in_channels[0]),
                nn.BatchNorm1d(self.in_channels[0]) if self.use_batch_norm else nn.Identity(),
                activation,
                nn.Linear(self.in_channels[0], self.out_channels[0]),
                nn.BatchNorm1d(self.out_channels[0]) if self.use_batch_norm else nn.Identity(),
                activation,
            ]
            if len(self.in_channels) > 1:
                self.generative_pxz += [
                    nn.Linear(self.in_channels[1], self.out_channels[1]),
                    nn.BatchNorm1d(self.out_channels[1]) if self.use_batch_norm else nn.Identity(),
                ]
            self.generative_pxz.append(nn.Linear(self.out_channels[1], g_dim))
            self.generative_pxz = nn.ModuleList(self.generative_pxz)
        elif self.rec_type == 'Gaussian':
            self.generative_pxz = []
            for i_layer in range(len(self.in_channels)):
                self.generative_pxz += [
                    nn.Linear(self.in_channels[i_layer], self.out_channels[i_layer]),
                    nn.BatchNorm1d(self.out_channels[i_layer]) if self.use_batch_norm else nn.Identity(),
                    activation,
                ]
            self.generative_pxz.append(nn.ModuleList([DenseBlock(self.out_channels[-1], g_dim, activation=activation), DenseBlock(self.out_channels[-1], g_dim, activation=activation)]))
            self.generative_pxz.append(nn.ModuleList([nn.Linear(g_dim, g_dim), nn.Linear(g_dim, g_dim)]))
            # self.generative_pxz.append(nn.ModuleList([nn.Linear(self.out_channels[-1], g_dim), nn.Linear(self.out_channels[-1], g_dim)]))
            self.generative_pxz = nn.ModuleList(self.generative_pxz)
        elif self.rec_type == 'Poisson':
            # raise NotImplementedError
            # λ: rate
            self.generative_pxz = nn.ModuleList([
                nn.Linear(z_dim, self.in_channels[0]),
                nn.BatchNorm1d(self.in_channels[0]) if self.use_batch_norm else nn.Identity(),
                activation,
                nn.Linear(self.in_channels[0], self.out_channels[0]),
                nn.BatchNorm1d(self.out_channels[0]) if self.use_batch_norm else nn.Identity(),
                activation,
                nn.Linear(self.in_channels[1], self.out_channels[1]),
                nn.BatchNorm1d(self.out_channels[1]) if self.use_batch_norm else nn.Identity(),
                nn.Linear(self.out_channels[1], g_dim),
                nn.Softplus()
            ])
        elif self.rec_type == 'NegativeBinomial':
            # raise NotImplementedError
            if r_type == 'gene-wise':
                # r: number of failures until the experiment is stopped; m: mean of the negative binomial distribution
                # self.r_para = nn.parameter.Parameter(torch.randn(g_dim)) # bug
                # self.r_para = torch.randn(g_dim, requires_grad=True)
                # self.r = torch.exp(self.r_para) # slow
                # self.r = nn.ParameterList([torch.randn(g_dim, requires_grad=True),
                self.r = nn.ParameterList([nn.parameter.Parameter(torch.randn(g_dim, requires_grad=True)),])
                self.r_exp = torch.exp
                self.p = nn.ParameterList([nn.parameter.Parameter(torch.randn(g_dim, requires_grad=True)),])
                self.p_sigmoid = nn.Sigmoid()
                self.k_exp = torch.exp
                # self.r = torch.exp(self.r_para).cuda() # bug
                # self.r = torch.exp(nn.Parameter(torch.cuda.FloatTensor(g_dim).normal_())) # hung
                self.generative_pxz = [
                    nn.Linear(z_dim, self.in_channels[0]),
                    nn.BatchNorm1d(self.in_channels[0]) if self.use_batch_norm else nn.Identity(),
                    activation,
                ]
                for i_layer in range(len(self.in_channels)):
                    self.generative_pxz += [
                        nn.Linear(self.in_channels[i_layer], self.out_channels[i_layer]),
                        nn.BatchNorm1d(self.out_channels[i_layer]) if self.use_batch_norm else nn.Identity(),
                        activation,
                    ]
                self.generative_pxz.append(nn.Linear(self.out_channels[-1], g_dim))
                self.generative_pxz = nn.ModuleList(self.generative_pxz)
            elif r_type == 'element-wise':
                # r: number of failures until the experiment is stopped; p: probability of success
                self.generative_pxz = nn.ModuleList([
                    nn.Linear(z_dim, self.in_channels[0]),
                    nn.BatchNorm1d(self.in_channels[0]) if self.use_batch_norm else nn.Identity(),
                    activation,
                    nn.Linear(self.in_channels[0], self.out_channels[0]),
                    nn.BatchNorm1d(self.out_channels[0]) if self.use_batch_norm else nn.Identity(),
                    activation,
                    nn.Linear(self.in_channels[1], self.out_channels[1]),
                    nn.BatchNorm1d(self.out_channels[1]) if self.use_batch_norm else nn.Identity(),
                    nn.ModuleList([nn.Linear(self.out_channels[1], g_dim), nn.Linear(self.out_channels[1], g_dim)]),
                    # nn.ModuleList([nn.ReLU(), nn.Sigmoid()])
                    # nn.ModuleList([nn.Softplus(), nn.Sigmoid()])
                    nn.ModuleList([nn.Softplus(), BoundedAct(nn.Sigmoid, 0.001)])
                ])
        elif self.rec_type == 'MSE':
            self.generative_pxz = nn.ModuleList([
                nn.Linear(z_dim, self.in_channels[0]),
                nn.BatchNorm1d(self.in_channels[0]) if self.use_batch_norm else nn.Identity(),
                activation,
                nn.Linear(self.in_channels[0], self.out_channels[0]),
                nn.BatchNorm1d(self.out_channels[0]) if self.use_batch_norm else nn.Identity(),
                activation,
                nn.Linear(self.in_channels[1], self.out_channels[1]),
                nn.BatchNorm1d(self.out_channels[1]) if self.use_batch_norm else nn.Identity(),
                nn.Linear(self.out_channels[1], g_dim)
            ])
        # elif self.rec_type == 'ZeroInflatedNegativeBinomial':
        #     # r: number of failures until the experiment is stopped; p: probability of success; π: dropout probability
        #     # raise NotImplementedError
        #     self.generative_pxz = nn.ModuleList([
        #         nn.Linear(z_dim, 512),
        #         activation,
        #         nn.Linear(512, 512),
        #         activation,
        #         nn.Linear(512, 512),
        #         nn.ModuleList([nn.Linear(512, g_dim), nn.Linear(512, g_dim), nn.Linear(512, g_dim)]),
        #         nn.ModuleList([nn.Softplus(), nn.Sigmoid(), nn.Sigmoid()])
        #     ])
        else:
            raise NotImplementedError('No implementation of the generative layers for %s distribution outputs.' % rec_type)

    # p(x|z)s
    # @profile
    def pxzs(self, zs):
        z_dict = {}
        # print('zs.shape:', zs.shape)
        z_dict['z'] = zs
        for layers in self.generative_pxz:
            if isinstance(layers, nn.ModuleList):
                for j_layer, layer in enumerate(layers, 1):
                    z_dict['z_{}'.format(j_layer)] = layer(z_dict.get('z_{}'.format(j_layer), z_dict['z']))
                    # print(layer, z_dict['z_{}'.format(j_layer)].shape)
            else:
                if isinstance(layers, nn.BatchNorm1d):
                    C, B, D = z_dict['z'].shape
                    z_dict['z'] = layers(z_dict['z'].view(-1, D)).view(C, B, D)
                else:
                    z_dict['z'] = layers(z_dict['z'])
                # print(layers, z_dict['z'].shape)
        if self.rec_type == 'NegativeBinomial':
            z_dict['z_r_type'] = self.r_type
            # if self.r_type == "gene-wise":
                # z_dict['z_r'] = self.r_exp(self.r[0]).type_as(zs)
                # z_dict['z_p'] = self.p_sigmoid(self.p[0]).type_as(zs)
            z_dict['z_k'] = self.k_exp(z_dict['z']).type_as(zs)
        return z_dict

    # @profile
    def get_prior_mu_logvar(self, onehot_idx=None):
        if onehot_idx is None:
            onehot_y = self.onehot_y
        else:
            onehot_y = torch.zeros((1, self.y_dim), device=self.device, dtype=torch.float32)
            onehot_y[0, onehot_idx] = 1
        y_mus = self.y_mus_2(self.y_mus_1(onehot_y))
        # ret_dict = {"prior_mu": y_mus}
        if self.GMM_model_name == "VVI":
            y_logvars = self.y_logvars_2(self.y_logvars_1(onehot_y))
        elif self.GMM_model_name == "EEE":
            y_logvars = None
        # ret_dict["prior_logvar"] = y_logvars
        # return ret_dict
        return y_mus, y_logvars

    # @profile
    def forward(self, y, zs, onehot_idx=None):
        if self.prior_generator.startswith("fc"):
            y_mus, y_logvars = self.get_prior_mu_logvar()
        elif self.prior_generator.startswith("tensor"):
            y_mus, y_logvars = self.mu_prior, self.logvar_prior
        x_recs = self.pxzs(zs)
        # onehot_idx is None & zs: C x D, C x D, B x C x N x G [C x B x G]
        # onehot_idx is None & z: 1 x D, 1 x D, B x 1 x N x G [1 x B x G]
        # onehot_idx is not None & zs: 1 x D, 1 x D, B x 1 x N x G [1 x B x G]
        # onehot_idx is not None & z: 1 x D, 1 x D, B x N x G [B x G]
        return y_mus, y_logvars, x_recs
