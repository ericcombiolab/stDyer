import torch
from torch import nn

# import torch.nn.init as init
# import numpy as np
# from torch.nn import functional as F
# from torch.nn.parameter import Parameter

class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, y_dim, activation=nn.ELU(), connect_prior=False, device=None):
    super(GumbelSoftmax, self).__init__()
    self.device = device
    self.connect_prior = connect_prior
    self.softmax = nn.Softmax(dim=-1)
    self.before_logits = nn.Linear(f_dim, f_dim)
    self.before_logits_act = activation
    self.logits = nn.Linear(f_dim, y_dim) # do not consider spatial information
    # self.before_logits = nn.Linear(f_dim, 2 * f_dim)
    # self.before_logits_act = nn.ReLU()
    # self.logits = nn.Linear(2 * f_dim, y_dim) # do not consider spatial information

    self.f_dim = f_dim
    self.y_dim = y_dim

  def sample_gumbel(self, shape, is_cuda=False, eps=torch.nextafter(torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))):
    U = torch.rand(shape)
    if is_cuda:
      U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    # return F.softmax(y / temperature, dim=-1)
    return self.softmax(y / temperature)

  def gumbel_softmax(self, logits, temperature, hard_gumbel=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard_gumbel:
        return y
    else:
        raise NotImplementedError
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

  def forward(self, x):
    # logits = self.logits(x).view(-1, self.y_dim)
    logits = self.logits(self.before_logits_act(self.before_logits(x)))
    logits = self.logits(x)
    # logits = self.before_logits_act(self.before_logits(x))
    # logits = self.logits(logits)
    # prob = F.softmax(logits, dim=-1)
    prob = self.softmax(logits)
    # y = self.gumbel_softmax(self.log_softmax(logits, dim=-1), temperature, hard_gumbel)
    # y = self.gumbel_softmax(self.log_softmax(logits), temperature, hard_gumbel)
    y = torch.empty_like(prob)
    # print("logits:", logits[0, :3])
    # print("prob:", prob[0, :3])
    # print("y:", y[0, :3])
    if len(x.shape) == 3:
        raise NotImplementedError
        # B x N x C
        return logits.view(x.shape[0], x.shape[1], self.y_dim), prob.view(x.shape[0], x.shape[1], self.y_dim), y.view(x.shape[0], x.shape[1], self.y_dim)
    elif len(x.shape) == 2:
        # B x C
        return logits.view(x.shape[0], self.y_dim), prob.view(x.shape[0], self.y_dim), y.view(x.shape[0], self.y_dim)

class Gaussian(nn.Module):
  def __init__(self, in_dim, z_dim, max_mu=10., max_logvar=5., min_logvar=-5., activation=nn.ELU(), skip_connection=False, kind="VVI", use_kl_bn=False, kl_bn_gamma=32., device=None):
    super(Gaussian, self).__init__()
    self.kind = kind
    self.use_kl_bn = use_kl_bn
    self.kl_bn_gamma = kl_bn_gamma
    self.mu_1 = DenseBlock(in_dim, z_dim, activation=activation, skip_connection=skip_connection)
    self.mu_2 = nn.Linear(z_dim, z_dim)
    if self.use_kl_bn:
        self.bn = nn.BatchNorm1d(z_dim)
        self.bn.weight.requires_grad_(False)
        self.bn.weight.fill_(self.kl_bn_gamma)
        self.bn.bias.requires_grad_(True)
    if kind == "VVI":
        self.logvar_1 = DenseBlock(in_dim, z_dim, activation=activation, skip_connection=skip_connection)
        self.logvar_2 = nn.Linear(z_dim, z_dim)
    elif kind == "EEE":
        from torch.nn.utils.parametrizations import orthogonal
        self.var_posi_defi_act = nn.Softplus()
        self.var_A = nn.ParameterList([nn.parameter.Parameter(torch.randn(z_dim, requires_grad=True)),])
        # self.var_A = nn.Parameter(torch.randn(z_dim, requires_grad=True))
        self.var_eye = nn.ParameterList([nn.parameter.Parameter(torch.eye(z_dim, requires_grad=False), requires_grad=False),])
        self.var_orth_layer = nn.ModuleList([orthogonal(nn.Linear(z_dim, z_dim, bias=False))])
        # self.var_orth_layer = nn.ModuleList([nn.Linear(z_dim, z_dim, bias=False)])
        # print("linear", type(nn.Linear(z_dim, z_dim, bias=False)))
        # print("orth", type(orthogonal(nn.Linear(z_dim, z_dim, bias=False))))
    self.max_mu = max_mu
    self.max_logvar = max_logvar
    self.min_logvar = min_logvar

  # def reparameterize(self, mu, var):
    # std = torch.sqrt(var + 1e-10)
  def reparameterize(self, mu, logvar):
    # std = torch.exp(logvar / 2.0) + 1e-5
    noise = torch.randn_like(mu)
    if self.kind == "VVI":
        std = torch.exp(logvar / 2.0)
        z = mu + noise * std
    elif self.kind == "EEE":
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        # L = torch.linalg.cholesky(torch.exp(logvar))
        # L, _ = torch.linalg.cholesky_ex(torch.exp(logvar))
        L, _ = torch.linalg.cholesky_ex(logvar.float())
        L = L.to(logvar.dtype)
        # print("L@L.T L.T@L", L@L.T, L.T@L)
        # print("torch.dist(L @ L.T, logvar)", torch.dist(L @ L.T, logvar))
        # mu, noise: B x D, L: D x D
        # L @ noise: (D x D, B x D) -> (B x D x D, B x D x 1) -> (B x D x 1) -> (B x D)
        z = mu + torch.matmul(L.unsqueeze(0), noise.unsqueeze(-1)).squeeze(-1)
        # z = mu + torch.matmul(noise, L.T)
        # print("mu.shape, logvar.shape, L.shape", mu.shape, logvar.shape, L.shape)
        # raise
    # z = mu
    # print("std.sum()", std.sum())
    # print("std.max()", std.max())
    # print("z.sum()", z.sum())
    # raise
    return z

  def forward(self, x):
    mu = self.mu_2(self.mu_1(x))
    var_inv = None
    if self.kind == "VVI":
        logvar = self.logvar_2(self.logvar_1(x))
        # for increasing the stability of the model
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
    elif self.kind == "EEE":
        # actually we compute var instead of logvar because lots of zeros exist in var and will lead to nan after log
        logvar =  self.var_orth_layer[0](self.var_eye[0]) @ torch.diag(self.var_posi_defi_act(self.var_A[0])) @ self.var_orth_layer[0](self.var_eye[0]).T
        var_inv = self.var_orth_layer[0](self.var_eye[0]) @ (torch.diag(1. / self.var_posi_defi_act(self.var_A[0]))) @ self.var_orth_layer[0](self.var_eye[0]).T
        # print("var", self.var_orth_layer.weight.T @ self.var_orth_layer.weight)
        # print("logvar", logvar)
        # print("var_inv", var_inv)
        # print("var_A", self.var_posi_defi_act(self.var_A[0]))
    mu = torch.clamp(mu, min=-self.max_mu, max=self.max_mu)
    if self.use_kl_bn:
        mu = self.bn(mu)
    z = self.reparameterize(mu, logvar)
    # return mu, logvar, z # (B x N x D) [B x D]
    return mu, logvar, z, var_inv

class DenseBlock(nn.Module):
    """DenseBlock Block which performs linear transformation when encoding.

    Args:
        in_features (int): input feature dimension.
        out_features (int): output feature dimension.
        use_bias (boolean): whether use bias in when calculating.
    """
    def __init__(
        self,
        in_features,
        out_features,
        use_bias=True,
        use_batch_norm=False,
        skip_connection=False,
        activation=nn.ReLU(),
        dropout=0,
    ):
        super(DenseBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        self.skip_connection = skip_connection
        self.activation = activation
        self.dropout_rate = dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation=True):
        output = self.linear(input)
        if self.use_batch_norm:
            if output.ndim == 2:
                output = self.batch_norm(output)
            elif output.ndim == 3:
                C, B, D = output.shape # or B x K x D
                output = self.batch_norm(output.view(-1, D)).view(C, B, D)
        if self.skip_connection:
            output = output + input
        if activation and self.activation is not None:
            output = self.activation(output)
        if self.dropout_rate > 0:
            output = self.dropout(output)
        return output
