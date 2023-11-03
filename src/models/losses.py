import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
# from pytorch_memlab import profile

class LossFunctions:
    def __init__(self, num_classes, GMM_model_name="VVI", out_log_sigma=0., nb_r=None, device="auto"):
      self.eps = 1e-8
      if device == "auto":
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      else:
         device = torch.device(device)
      self.GMM_model_name = GMM_model_name
      try:
         self.entropy_diff = OneHotCategorical(torch.tensor([1. / num_classes] * num_classes)).entropy() - \
            OneHotCategorical(torch.tensor([1. / (num_classes - 1)] * (num_classes - 1))).entropy()
      except:
         self.entropy_diff = 0.
      if isinstance(out_log_sigma, float):
        self.out_log_simga = out_log_sigma
      else:
         self.out_log_simga = torch.tensor(out_log_sigma).to(device)
      self.nb_r = torch.tensor(nb_r).to(device)

   #  @profile
    def batch_reconstruction_loss(self, real, predictions, prob_cat, rec_mask=1., rec_type='Bernoulli', real_origin=None, affi_weights=None, verbose=False):
      """batch reconstruction loss
         loss = -Σi Σj [(A_ij * log(reconstruct_ij)) + (1 - A_ij) * log(1 - reconstruct_ij)]

      Args:
         #  real: (array) corresponding array containing the true labels B x D
         #  predictions: (array) corresponding array containing the predicted labels C x B x D
         #  prob_cat: (array) array containing the probability for the categorical latent variable B x C

         #  adj_matrix (tensor): original normalized adjacency matrix. B x B
         #  reconstruct_graphs (tensor): reconstruct graphs by dot product. C x B x B
         #  mul_mask (tensor): mask matrix for multiplication. B x B
         #  prob_cat: (array) array containing the probability for the categorical latent variable. B x C

          real: (array) corresponding array containing the true labels B x D
          predictions: (array) corresponding array containing the predicted labels C x B x D
          prob_cat: (array) array containing the probability for the categorical latent variable. B x C


      Returns:
          loss (tensor): loss
      """
      if rec_type == "Bernoulli":
         loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * F.binary_cross_entropy_with_logits(predictions['z'], real.unsqueeze(0).expand(predictions['z'].shape), reduction='none')).sum((0, -1))
         # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * F.binary_cross_entropy_with_logits(predictions['z'], real.unsqueeze(0).expand(predictions['z'].shape), reduction='none')).sum(-1)
      elif rec_type == "NegativeBinomial":
         if real_origin is None:
            if predictions['z_r_type'] == 'gene-wise':
               # print("predictions['z'].shape", predictions['z'].shape)
               C, B, D = predictions['z_k'].shape

               # (use z_p)
               # predictions['z'] * (1 - predictions['z_p']) / predictions['z_p']
               # predictions['z'] / predictions['z_p']

               # predictions['z']
               # predictions['z'] / predictions['z_p']

               # rec_x = predictions['z']
               # z_p = predictions['z_p']
               # z_r = rec_x * (1 - z_p) / z_p
               # loss = -torch.nan_to_num(prob_cat.transpose(0, 1).unsqueeze(-1) * (
               #    ((real.unsqueeze(0) + z_r) != 0).float() * torch.lgamma(real.unsqueeze(0) + z_r) -
               #    torch.lgamma(real + 1).unsqueeze(0) -
               #    (z_r != 0).float() * torch.lgamma(z_r) +
               #    z_r * torch.log(1 - z_p) +
               #    # z_r * torch.log(z_r / (z_r + rec_x)) +
               #    real.unsqueeze(0) * torch.log(z_p)
               #    # real.unsqueeze(0) * torch.log(rec_x / (z_r + rec_x))
               #    # ), nan=0.0, posinf=0.0, neginf=0.0).mean(1).sum((0, 1))
               #    # ), nan=0.0, posinf=0.0, neginf=0.0).sum((0, 1))
               #    ), nan=0.0, posinf=0.0, neginf=0.0).sum((0, -1))

               rec_x = predictions['z_k']
               loss = -torch.nan_to_num(prob_cat.transpose(0, 1).unsqueeze(-1) * (
                  # ((rec_x + self.nb_r) != 0).float() * torch.lgamma(rec_x + self.nb_r) -
                  torch.lgamma(rec_x + self.nb_r) -
                  torch.lgamma(rec_x + 1) -
                  # (self.nb_r != 0).float() * torch.lgamma(self.nb_r) +
                  torch.lgamma(self.nb_r) +
                  self.nb_r * torch.log(self.nb_r / (self.nb_r + real.unsqueeze(0))) +
                  rec_x * torch.log(real.unsqueeze(0) / (self.nb_r + real.unsqueeze(0)))
                  ), nan=0.0, posinf=0.0, neginf=0.0).sum((0, -1))

               # (use z_p)
               # l1 = ((real.unsqueeze(0) + z_r) != 0).float() * torch.lgamma(real.unsqueeze(0) + z_r)
               # l2 = torch.lgamma(real + 1).unsqueeze(0)
               # l3 = (z_r != 0).float() * torch.lgamma(z_r)
               # l4_log = torch.log(1 - z_p)
               # l4 = z_r * l4_log
               # l5_log = torch.log(z_p)
               # l5 = real.unsqueeze(0) * l5_log
               # l1 = ((real.unsqueeze(0) + z_r) != 0).float() * torch.lgamma(real.unsqueeze(0) + z_r)
               # l2 = torch.lgamma(real + 1).unsqueeze(0)
               # l3 = (z_r != 0).float() * torch.lgamma(z_r)
               # l4 = z_r * torch.log(1 - z_p)
               # l5 = real.unsqueeze(0) * torch.log(z_p)
               # print("l1", l1.max(), l1.min())
               # print("l2", l2.max(), l2.min())
               # print("l3", l3.max(), l3.min())
               # print("l4", l4.max(), l4.min())
               # print("l5", l5.max(), l5.min())
               # loss = -torch.nan_to_num(prob_cat.transpose(0, 1).unsqueeze(-1) * (
               #    l1 -
               #    l2 -
               #    l3 +
               #    l4 +
               #    l5
               #    ), nan=0.0, posinf=0.0, neginf=0.0).sum((0, -1))

               # nan_bl_idx = torch.isnan(prob_cat.transpose(0, 1).unsqueeze(-1) * (
               #    ((real.unsqueeze(0) + z_r) != 0).float() * torch.lgamma(real.unsqueeze(0) + z_r) -
               #    torch.lgamma(real + 1).unsqueeze(0) -
               #    (z_r != 0).float() * torch.lgamma(z_r) +
               #    z_r * torch.log(1 - z_p) +
               #    real.unsqueeze(0) * torch.log(z_p)
               #    ))
               # if nan_bl_idx.sum() != 0:
               #    # C x B x D
               #    print("nan_bl_idx.shape", nan_bl_idx.shape)
               #    # C x B x D
               #    print((((real.unsqueeze(0) + z_r) != 0).float() * torch.lgamma(real.unsqueeze(0) + z_r))[nan_bl_idx])
               #    # 1 x B x D -> C x B x D
               #    print(torch.lgamma(real + 1).unsqueeze(0).expand((C, B, D))[nan_bl_idx])
               #    # C x B x D
               #    print(((z_r != 0).float() * torch.lgamma(z_r))[nan_bl_idx])
               #    # C x B x D
               #    print((z_r * torch.log(1 - z_p))[nan_bl_idx])
               #    # 1 x B x D -> C x B x D
               #    print((real.unsqueeze(0) * torch.log(z_p)).expand((C, B, D))[nan_bl_idx])
               #    print("z_r", z_r.min(), z_r.max())
               #    print("torch.lgamma(z_r)", torch.lgamma(z_r).min(), torch.lgamma(z_r).max())

               #    print(torch.log(predictions['z'])[nan_bl_idx])
               #    # print(predictions['z'][nan_bl_idx])
               #    log_nan_bl_idx = torch.isnan(torch.log(predictions['z']))
               #    print(torch.log(predictions['z'])[log_nan_bl_idx])
               #    print(predictions['z'][log_nan_bl_idx])

               # inf_bl_idx = torch.isinf((prob_cat.transpose(0, 1).unsqueeze(-1) * (
               #    ((real.unsqueeze(0) + z_r) != 0).float() * torch.lgamma(real.unsqueeze(0) + z_r) -
               #    torch.lgamma(real + 1).unsqueeze(0) -
               #    (z_r != 0).float() * torch.lgamma(z_r) +
               #    z_r * torch.log(1 - z_p) +
               #    real.unsqueeze(0) * torch.log(z_p)
               #    )).mean(1))

               # if inf_bl_idx.sum() != 0:
               #    # C x D
               #    print("inf_bl_idx.shape", inf_bl_idx.shape)
               #    # C x B x D -> C x D
               #    print((((real.unsqueeze(0) + z_r) != 0).float() * torch.lgamma(real.unsqueeze(0) + z_r)).mean(1)[inf_bl_idx])
               #    # B x 1 x D -> 1 X D -> C x D
               #    print(torch.lgamma(real + 1).unsqueeze(0).mean(0).expand((C, D))[inf_bl_idx])
               #    # C x B x D -> C x D
               #    print(((z_r != 0).float() * torch.lgamma(z_r)).mean(1)[inf_bl_idx])
               #    # C x B x D -> C X D
               #    print((z_r * torch.log(1 - z_p)).mean(1)[inf_bl_idx])
               #    # 1 x B x D -> 1 X D -> C X D
               #    print((real.unsqueeze(0) * torch.log(z_p)).mean(1).expand((C, D))[inf_bl_idx])

               #    print(torch.log(predictions['z']).mean(1)[inf_bl_idx])
               #    # print(predictions['z'].mean(1)[inf_bl_idx])
               #    log_inf_bl_idx = torch.isinf(torch.log(predictions['z']))
               #    print(torch.log(predictions['z'])[log_inf_bl_idx])
               #    print(predictions['z'][log_inf_bl_idx])

            elif predictions['z_r_type'] == 'element-wise':
               loss = -torch.nan_to_num(prob_cat.transpose(0, 1).unsqueeze(-1) * (
                  ((real.unsqueeze(0) + predictions['z_1']) != 0).float() * torch.lgamma(real.unsqueeze(0) + predictions['z_1']) -
                  torch.lgamma(real + 1).unsqueeze(0) -
                  (predictions['z_1'] != 0).float() * torch.lgamma(predictions['z_1']) +
                  real.unsqueeze(0) * torch.log(1 - predictions['z_2']) +
                  predictions['z_1'] * torch.log(predictions['z_2'])
                  ), nan=0.0, posinf=0.0, neginf=0.0).sum((0, -1))
         else:
            if predictions['z_r_type'] == 'gene-wise':
               # C, B, G = predictions['z'].shape
               # # print("C, B, G", C, B, G)
               # # B_, C_ = prob_cat.shape
               # # prob_cat B x C -> C x B x G
               # # real_origin B x G -> C x B x G
               # rec_x = predictions['z']
               # z_p = predictions['z_p']
               # z_r = rec_x * (1 - z_p) / z_p
               # # C x B x G
               # loss = -torch.nan_to_num(prob_cat.transpose(0, -1).unsqueeze(-1) * (
               #    ((real_origin.unsqueeze(0) + z_r) != 0).float() * torch.lgamma(real_origin.unsqueeze(0) + z_r) -
               #    torch.lgamma(real_origin + 1).unsqueeze(0) -
               #    (z_r != 0).float() * torch.lgamma(z_r) +
               #    z_r * torch.log(1 - z_p) +
               #    real_origin.unsqueeze(0) * torch.log(z_p)
               #    ), nan=0.0, posinf=0.0, neginf=0.0).sum((0, -1))

               C, B, G = predictions['z_k'].shape
               rec_x = predictions['z_k']
               loss = -torch.nan_to_num(prob_cat.transpose(0, 1).unsqueeze(-1) * (
                  torch.lgamma(rec_x + self.nb_r) -
                  torch.lgamma(rec_x + 1) -
                  torch.lgamma(self.nb_r) +
                  self.nb_r * torch.log(self.nb_r / (self.nb_r + real_origin.unsqueeze(0))) +
                  rec_x * torch.log(real_origin.unsqueeze(0) / (self.nb_r + real_origin.unsqueeze(0)))
                  ), nan=0.0, posinf=0.0, neginf=0.0).sum((0, -1))

            elif predictions['z_r_type'] == 'element-wise':
               # generated by Copilot, not checked carefully
               loss = -torch.nan_to_num(prob_cat.transpose(0, -1).unsqueeze(-1) * (
                  ((real_origin.unsqueeze(0) + predictions['z_1']) != 0).float() * torch.lgamma(real_origin.unsqueeze(0) + predictions['z_1']) -
                  torch.lgamma(real_origin + 1).unsqueeze(0) -
                  (predictions['z_1'] != 0).float() * torch.lgamma(predictions['z_1']) +
                  real_origin.unsqueeze(0) * torch.log(1 - predictions['z_2']) +
                  predictions['z_1'] * torch.log(predictions['z_2'])
                  ), nan=0.0, posinf=0.0, neginf=0.0).sum((0, -1))
            else:
               raise NotImplementedError
      elif rec_type == "Gaussian":
         if real_origin is None:
            # print("prob_cat.shape", prob_cat.shape)
            # print("predictions['z_1'].shape", predictions['z_1'].shape)
            # print("real.shape", real.shape)
            if not verbose:
               # C x B x D -> B x D -> B
               # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * (((predictions['z_1'] - real.unsqueeze(0)).square()) + np.log(2. * np.pi))).sum((0, -1))
               # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * ((((predictions['z_1'] - real.unsqueeze(0)) / predictions['z_2'].exp()).square()) + np.log(2. * np.pi) + 2 * predictions['z_2'])).sum((0, -1))
               # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * ((((predictions['z_1'] - real.unsqueeze(0)) / np.exp(self.out_log_simga)).square()) + np.log(2. * np.pi) + 2 * self.out_log_simga)).sum((0, -1))
               # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * ((((predictions['z_1'] - real.unsqueeze(0)) / np.exp(self.out_log_simga)).square()))).sum((0, -1))
               # loss = ((prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * (((predictions['z_1'] - real.unsqueeze(0)) / np.exp(self.out_log_simga)).square())).sum(0) * rec_mask).sum(-1)
               if isinstance(rec_mask, float):
                  loss = (prob_cat.transpose(0, 1) * 0.5 * (((predictions['z_1'] - real.unsqueeze(0)) / np.exp(self.out_log_simga)).square()).sum(-1)).sum(0)
               else:
                  loss = (prob_cat.transpose(0, 1) * 0.5 * (((predictions['z_1'] - real.unsqueeze(0)) / np.exp(self.out_log_simga)).square() * rec_mask.unsqueeze(0)).sum(-1)).sum(0)
            else:
               # C x B x D -> C x B
               # cluster_loss = (((predictions['z_1'] - real.unsqueeze(0)).square()) + np.log(2. * np.pi)).sum(-1)
               if isinstance(rec_mask, float):
                  cluster_loss = 0.5 * (((predictions['z_1'] - real.unsqueeze(0)).square())).sum(-1)
               else:
                  cluster_loss = (0.5 * (((predictions['z_1'] - real.unsqueeze(0)).square())) * rec_mask.unsqueeze(0)).sum(-1)
               weighted_cluster_loss = prob_cat.transpose(0, 1) * cluster_loss
               # C x B -> B
               loss = weighted_cluster_loss.sum(0)
         else:
            if not verbose:
               # C x B x D -> B x D -> B
               # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * (((predictions['z_1'] - real_origin.unsqueeze(0)).square()) + np.log(2. * np.pi))).sum((0, -1))
               # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * ((((predictions['z_1'] - real_origin.unsqueeze(0)) / predictions['z_2'].exp()).square()) + np.log(2. * np.pi) + 2 * predictions['z_2'])).sum((0, -1))
               # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * ((((predictions['z_1'] - real_origin.unsqueeze(0)) / np.exp(self.out_log_simga)).square()) + np.log(2. * np.pi) + 2 * self.out_log_simga)).sum((0, -1))
               # loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * ((((predictions['z_1'] - real_origin.unsqueeze(0)) / np.exp(self.out_log_simga)).square()))).sum((0, -1))
               # loss = ((prob_cat.transpose(0, 1).unsqueeze(-1) * 0.5 * ((((predictions['z_1'] - real_origin.unsqueeze(0)) / np.exp(self.out_log_simga)).square()))).sum(0) * rec_mask).sum(-1)
               if isinstance(rec_mask, float):
                  loss = (prob_cat.transpose(0, 1) * 0.5 * (((predictions['z_1'] - real_origin.unsqueeze(0)) / np.exp(self.out_log_simga)).square()).sum(-1)).sum(0)
               else:
                  loss = (prob_cat.transpose(0, 1) * 0.5 * (((predictions['z_1'] - real_origin.unsqueeze(0)) / np.exp(self.out_log_simga)).square() * rec_mask.unsqueeze(0)).sum(-1)).sum(0)
               # # prob_cat: B x K x C -> K x C x B x D
               # loss = (prob_cat.transpose(0, 1).transpose(1, 2).unsqueeze(-1) * 0.5 * (predictions['z'] ** 2 + np.log(2. * np.pi)).unsqueeze(0)).mean((0, 2)).sum((0, 1))
            else:
               # C x B x D -> C x B
               # cluster_loss = (((predictions['z_1'] - real_origin.unsqueeze(0)).square()) + np.log(2. * np.pi)).sum(-1)
               if isinstance(rec_mask, float):
                  cluster_loss = 0.5 * (((predictions['z_1'] - real_origin.unsqueeze(0)).square())).sum(-1)
               else:
                  cluster_loss = (0.5 * (((predictions['z_1'] - real_origin.unsqueeze(0)).square())) * rec_mask.unsqueeze(0)).sum(-1)
               weighted_cluster_loss = prob_cat.transpose(0, 1) * cluster_loss
               # C x B -> B
               loss = weighted_cluster_loss.sum(0)
      elif rec_type == "MSE":
         if real_origin is None:
            loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * (predictions['z'] - real.unsqueeze(0)) ** 2).sum((0, -1))
         else:
            loss = (prob_cat.transpose(0, 1).unsqueeze(-1) * (predictions['z'] - real_origin.unsqueeze(0)) ** 2).sum((0, -1))
      # return loss
      if affi_weights is not None:
         loss = loss * affi_weights
      if not verbose:
         return {"loss": loss}
      else:
         return {"loss": loss, "cluster_loss": cluster_loss, "weighted_cluster_loss": weighted_cluster_loss}
         # loss.retain_grad()
         # l1.retain_grad()
         # l3.retain_grad()
         # l4.retain_grad()
         # l4_log.retain_grad()
         # l5.retain_grad()
         # l5_log.retain_grad()
         # z_r.retain_grad()
         # z_p.retain_grad()
         # debug_dict = {"loss": loss,
         #               "l1": l1,
         #               "l3": l3,
         #               "l4": l4,
         #               "l4_log": l4_log,
         #               "l5": l5,
         #               "l5_log": l5_log,
         #               "z_r": z_r,
         #               "z_p": z_p}

   #  @profile
    def batch_expected_gaussian_loss(self, prob_cat, z_mus, z_logvars, z_mu_priors, z_logvar_priors, affi_weights=None, y_var_invs=None, verbose=False, kind="element-wise"):
      """The expected gaussian loss for each y.
         loss = KL[q(z|x,y)||p(z|y)]
              = 0.5 * [tr(Σ_1^-1 Σ_0) + (μ_1-μ_0)^T Σ_1^-1 (μ_1-μ_0) - d + log(detΣ_1/detΣ_0)]

      Args:
         prob_cat: (array) array containing the probability for the categorical latent variable B x C
         z_mus: (array) array containing the mean of the inference model C x B x D
         z_logvars: (array) array containing the log(variance) of the inference model C x B x D
         z_mu_priors: (array) array containing the prior mean of the generative model C x D
         z_logvar_priors: (array) array containing the prior log(variance) of the generative mode C x D
      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      if not verbose:
         if kind == "element-wise":
            if self.GMM_model_name == "VVI":
               # # (μ_1-μ_0)^T Σ_1^−1 (μ_1-μ_0)
               # B x C -> C x B
               # C x B x D -> C x B
               # C x B x D -> C x B
               # 1
               # C x B x D -> C x B
               # C x B -> B
               loss = (prob_cat.transpose(0, 1) * 0.5 * (
                     (z_logvars - z_logvar_priors.unsqueeze(1)).exp().sum(-1) +
                     ((z_mu_priors.unsqueeze(1) - z_mus).square() / z_logvar_priors.unsqueeze(1).exp()).sum(-1) -
                     z_mus.shape[-1] +
                     (z_logvar_priors.unsqueeze(1) - z_logvars).sum(-1)
                  )).sum(0)
            elif self.GMM_model_name == "EEE":
               # B x C -> C x B
               # C x B x D, C x D^2, C x D x B -> C x B x 1 x D, C x B x D x D, C x B x D x 1 -> C x B x 1 x 1 -> C x B
               # C x B -> B
               # TODO: simplify computation if possible.
               # print("z_logvar_priors", torch.isfinite(z_logvar_priors).sum())
               # print("y_var_invs", torch.isfinite(y_var_invs).sum())

               loss = (prob_cat.transpose(0, 1) * 0.5 * (
                     # z_mus.shape[-1] + # because prior and post share the same covirance matrix
                     torch.matmul(
                        torch.matmul((z_mu_priors.unsqueeze(1) - z_mus).unsqueeze(-2),
                                     torch.broadcast_to(y_var_invs.view(z_mus.shape[0], 1, z_mus.shape[-1], z_mus.shape[-1]), (z_mus.shape[0], z_mus.shape[1], z_mus.shape[-1], z_mus.shape[-1]))
                                    ),
                        (z_mu_priors.unsqueeze(1) - z_mus).unsqueeze(-1)).squeeze(-1).squeeze(-1) # there will be a minus sign in the end if we want to add more terms
                     # z_mus.shape[-1] +
                     # 0 # because prior and post share the same covirance matrix
                  )).sum(0)
         elif kind == "batch":
            if self.GMM_model_name == "EEE":
               raise NotImplementedError
            # C x B x D -> C x D
            z_logvars = 2 * torch.log(torch.std(z_mus, 1))
            # B x C -> C x B
            # C x D -> C x 1
            # C x B x D -> C x B
            # 1
            # C x D -> C x 1
            # C x B -> B
            loss = (prob_cat.transpose(0, 1) * 0.5 * (
                  (z_logvars - z_logvar_priors).exp().sum(-1, keepdim=True) +
                  ((z_mu_priors.unsqueeze(1) - z_mus).square() / z_logvar_priors.unsqueeze(1).exp()).sum(-1) -
                  z_mus.shape[-1] +
                  (z_logvar_priors - z_logvars).sum(-1, keepdim=True)
               )).sum(0)
         if affi_weights is not None:
            loss = loss * affi_weights
         return {"loss": loss}
      else:
         # C x B x D -> C x B
         cluster_loss = 0.5 * ((z_logvars - z_logvar_priors.unsqueeze(1)).exp().sum(-1) + \
            ((z_mu_priors.unsqueeze(1) - z_mus).square() / z_logvar_priors.unsqueeze(1).exp()).sum(-1) - \
            z_mus.shape[-1] + \
            (z_logvar_priors.unsqueeze(1) - z_logvars).sum(-1))
         weighted_cluster_loss = prob_cat.transpose(0, 1) * cluster_loss
         if affi_weights is not None:
            weighted_cluster_loss = weighted_cluster_loss * affi_weights.unsqueeze(0)
         # C x B -> B
         loss = weighted_cluster_loss.sum(0)
         return {"loss": loss, "cluster_loss": cluster_loss, "weighted_cluster_loss": weighted_cluster_loss}

   #  @profile
    def batch_expected_categorical_loss(self, prob_cat=None, affi_weights=None, prior='average_uniform', learned_prior=None):
      """The expected categorical loss.
         loss = KL[q(y|x)||p(y)]

      Args:
         prob_cat: (array) array containing the probability for the categorical latent variable B x C
         y: (array) array containing the categorical latent variable B x C
         logits: (array) array containing the logtis for the categorical latent variable B x C

      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      # strange torch implementation https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
      if isinstance(prior, str):
         if prior.startswith('average_uniform'):
            if True:
            # if affi_weights is not None:
            #    prob_cat_mean = (prob_cat * affi_weights.unsqueeze(-1)).sum(0) / affi_weights.sum(0)
            # else:
               prob_cat_mean = prob_cat.mean(0)
               # print("prob_cat_mean", prob_cat_mean)
               # print("learned_prior", learned_prior)
            if prior.startswith('average_uniform'):
               if learned_prior is None:
                  loss = F.kl_div(-torch.log(torch.ones_like(prob_cat_mean) * prob_cat.shape[-1]), prob_cat_mean, reduction='none').sum(-1)
               else:
                  loss = F.kl_div(torch.log(torch.ones_like(prob_cat_mean) * torch.from_numpy(learned_prior).to(prob_cat_mean.device)), prob_cat_mean, reduction='none').sum(-1)
            if prior.endswith('batch'):
               loss = len(prob_cat) * loss
         else:
            raise NotImplementedError
      elif isinstance(prior, torch.Tensor):
         # prior_cat = torch.ones_like(prob_cat)
         # prior_cat[:] = torch.tensor(prior, device=prob_cat.device, dtype=prob_cat.dtype)
         # loss = F.kl_div(torch.log(prior_cat), prob_cat, reduction='none').sum(-1)
         # loss = F.kl_div(torch.log(prior), prob_cat, reduction='none').sum(-1)
         # print(prior)
         # print(prob_cat.mean(0))
         loss = F.kl_div(torch.log(prior), prob_cat.mean(0), reduction='none').sum(-1)
         # loss = F.kl_div(torch.log(prior.reshape(1, -1)), prob_cat.mean(0).reshape(1, -1), reduction='none').sum(-1) # the same as above
         # print(loss)
         # print(loss.shape)
         # raise
      else:
         print(type(prior))
         print(prior)
         raise NotImplementedError
      return loss.mean()
