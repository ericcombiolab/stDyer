import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
# from pytorch_memlab import profile

class LossFunctions:
    def __init__(self, num_classes, GMM_model_name="VVI", out_log_sigma=0., nb_r=None, out_type="scaled", device="auto"):
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
      self.out_type = out_type

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

               rec_x = predictions['z_k']
               real_mean = (self.nb_r * (2 * real + 1)) / (2 * (self.nb_r - 1)) # accurate but unstable at all
               loss = -(prob_cat.transpose(0, 1).unsqueeze(-1) * (
                  torch.lgamma(rec_x + self.nb_r) -
                  torch.lgamma(rec_x + 1) -
                  torch.lgamma(self.nb_r) +
                  self.nb_r * torch.log(self.nb_r / (self.nb_r + real_mean.unsqueeze(0))) +
                  (rec_x == 0.) * (real_mean.unsqueeze(0) == 0.) + rec_x * torch.nan_to_num(torch.log(real_mean.unsqueeze(0) / (self.nb_r + real_mean.unsqueeze(0))), nan=0.0, posinf=0.0, neginf=0.0)
                  )).sum((0, -1))

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
               C, B, G = predictions['z_k'].shape
               rec_x = predictions['z_k']
               real_origin_mean = (self.nb_r * (2 * real_origin + 1)) / (2 * (self.nb_r - 1)) # stable but inaccurate according to formula
               loss = -(prob_cat.transpose(0, 1).unsqueeze(-1) * (
                  torch.lgamma(rec_x + self.nb_r) -
                  torch.lgamma(rec_x + 1) -
                  torch.lgamma(self.nb_r) +
                  self.nb_r * torch.log(self.nb_r / (self.nb_r + real_origin_mean.unsqueeze(0))) +
                  (rec_x == 0.) * (real_origin_mean.unsqueeze(0) == 0.) + rec_x * torch.nan_to_num(torch.log(real_origin_mean.unsqueeze(0) / (self.nb_r + real_origin_mean.unsqueeze(0))), nan=0.0, posinf=0.0, neginf=0.0)
                  )).sum((0, -1))
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
         clamp_factor = 6
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
                  if not self.out_type.startswith("pca"):
                     loss = (prob_cat.transpose(0, 1) * 0.5 * (((predictions['z_1'] - real.unsqueeze(0)) / np.exp(self.out_log_simga)).square()).sum(-1)).sum(0)
                  else:
                     loss_c = 0.5 * np.log(2. * np.pi) * real.shape[-1]
                     loss = (prob_cat.transpose(0, 1) * 0.5 * (predictions['z_2'].sum(-1).clamp(-clamp_factor * loss_c, clamp_factor * loss_c) + ((predictions['z_1'] - real.unsqueeze(0)).square() / predictions['z_2'].exp()).sum(-1).clamp(-clamp_factor * loss_c, clamp_factor * loss_c))).sum(0)
                     loss = loss / loss.detach() * (loss.detach() + loss_c)
               else:
                  loss = (prob_cat.transpose(0, 1) * 0.5 * (((predictions['z_1'] - real.unsqueeze(0)) / np.exp(self.out_log_simga)).square() * rec_mask.unsqueeze(0)).sum(-1)).sum(0)
            else:
               raise NotImplementedError
         else:
            if not verbose:
               if isinstance(rec_mask, float):
                  if not self.out_type.startswith("pca"):
                     loss = (prob_cat.transpose(0, 1) * 0.5 * (((predictions['z_1'] - real_origin.unsqueeze(0)) / np.exp(self.out_log_simga)).square()).sum(-1)).sum(0)
                  else:
                     loss_c = 0.5 * np.log(2. * np.pi) * real_origin.shape[-1]
                     loss = (prob_cat.transpose(0, 1) * 0.5 * (predictions['z_2'].sum(-1).clamp(-clamp_factor * loss_c, clamp_factor * loss_c) + ((predictions['z_1'] - real_origin.unsqueeze(0)).square() / predictions['z_2'].exp()).sum(-1).clamp(-clamp_factor * loss_c, clamp_factor * loss_c))).sum(0)
                     loss = loss / loss.detach() * (loss.detach() + loss_c)
               else:
                  loss = (prob_cat.transpose(0, 1) * 0.5 * (((predictions['z_1'] - real_origin.unsqueeze(0)) / np.exp(self.out_log_simga)).square() * rec_mask.unsqueeze(0)).sum(-1)).sum(0)
            else:
               raise NotImplementedError
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
               prob_cat_mean = prob_cat.mean(0)
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
         loss = F.kl_div(torch.log(prior), prob_cat.mean(0), reduction='none').sum(-1)
      else:
         print(type(prior))
         print(prior)
         raise NotImplementedError
      return loss.mean()
