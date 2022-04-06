import numpy as np
from scipy.special import logsumexp
import torch

from .lbfgsnew import lbfgsnew

LOG2PI = 1.837877066409345483560659472811235279722794947275566825634

# TODO: move specific functions to each class

######################


def _gaussian_condparams_cov(a, b, datum, mean, cov):
  # a, b    : python list
  # mean    : ndarray
  # cov     : ndarray
  # datum_b : ndarray

  assert datum.ndim == 1
  if len(a)>0 and len(b)>0:
    mean_a = mean[a]
    mean_b = mean[b]
    cov_aa = cov[np.ix_(a, a)]
    cov_ab = cov[np.ix_(a, b)]
    cov_ba = cov[np.ix_(b, a)]
    cov_bb = cov[np.ix_(b, b)]
    tmp = np.linalg.solve(cov_bb.T, cov_ab.T).T
    mean_a_given_b = mean_a + tmp @ (datum[b] - mean_b)
    cov_a_given_b = cov_aa - tmp @ cov_ba
    return mean_a_given_b, cov_a_given_b
  elif len(a)>0:
    return mean, cov
  else:
    return [], []


def _compset(S, n):
  if type(S) is list:
    nonS = list(set(range(n)) - set(S))
  elif type(S) is tuple:
    nonS = tuple(set(range(n)) - set(S))
  elif type(S) is np.ndarray:
    nonS = np.array(list(set(range(n)) - set(S)))
  elif type(S) is set:
    nonS = set(range(n)) - S
  else:
    raise ValueError('unknown S type: type(S) = %s' % type(S))
  return nonS


######################


class F:
  def __init__(self, e):
    # e : python function, e(*args)
    #     This function can be defined for batch x.
    self.e = e
    self.count = 0

  def __call__(self, *args):
    e_ = self.e(*args)
    self.count+=e_.shape[0]
    return e_

  def zero_count(self):
    self.count=0


######################


def general_baseline(S, x, r, dim_x, ef):
  x = x.reshape(-1, dim_x)
  r = r.reshape(-1, dim_x)

  S = list(S)
  nonS = _compset(S, dim_x)

  ef_avg = np.zeros(x.shape[0])
  for j in range(r.shape[0]):
    x_new = np.copy(x)
    x_new[:,nonS] = r[j,nonS]
    ef_avg += ef(x_new)
  ef_avg /= r.shape[0]
  return ef_avg


def general_optbaseline(S, x, dim_x, ef, kwargs_to_bline):
  x = x.reshape(-1, dim_x)

  S = list(S)
  nonS = _compset(S, dim_x)
  if len(nonS)<1:
    return ef(x)

  r = np.zeros_like(x)
  for i in range(x.shape[0]):
    r[i] = optbaseline_lbfgs(S, x[i], ef, **kwargs_to_bline)
  #print(S, r, ef(r))
  return general_baseline(S, x, r, dim_x, ef)


# class general_optbaseline_approx1():
#   def __init__(self, x, dim_x, ef, kwargs_to_bline):
#     x = x.reshape(-1, dim_x)

#     self.x = x
#     self.dim_x = dim_x
#     self.ef = ef

#     self.efvalue_each = np.zeros((x.shape[0], dim_x))
#     for i in range(dim_x):
#       self.efvalue_each[:,i] = general_optbaseline([i,], x, dim_x, ef, kwargs_to_bline)
#     self.efvalue_empty = general_optbaseline([], x, dim_x, ef, kwargs_to_bline)

#   def __call__(self, S):
#     S = list(S)
#     nonS = _compset(S, self.dim_x)
#     if len(nonS)==self.dim_x:
#       return self.efvalue_empty

#     ef_avg = np.mean(self.efvalue_each[:,S], axis=1)
#     #print(S, ef_avg)
#     return ef_avg


class general_optbaseline_approx2():
  def __init__(self, x, dim_x, ef, kwargs_to_bline):
    assert x.ndim==1
    assert x.size==dim_x
    # currently for only single x

    self.x = x
    self.dim_x = dim_x
    self.ef = ef

    self.x_each = np.zeros((dim_x,dim_x))
    for i in range(dim_x):
      self.x_each[:,i] = optbaseline_lbfgs([i,], x, ef, **kwargs_to_bline)
    self.x_empty = optbaseline_lbfgs([], x, ef, **kwargs_to_bline)

  def __call__(self, S):
    S = list(S)
    nonS = _compset(S, self.dim_x)
    if len(nonS)<1:
      return self.ef(self.x)
    elif len(nonS)==self.dim_x:
      return self.ef(self.x_empty)

    r = np.sum(self.x_each[:,S], axis=1) + self.x_empty
    r /= len(S)+1
    x_new = np.copy(self.x)
    x_new[nonS] = r[nonS]
    ef_ = self.ef(x_new)
    #print(S, ef_)
    return ef_


def optbaseline_lbfgs(S, x, ef, gaussian=True, regparam=1e-2, margin=0.0,
  itermax=500, learnrate=0.1, tolreldif=1e-3):
  assert x.ndim == 1

  # TODO consider device properly

  if type(x) is np.ndarray:
    x_org = torch.from_numpy(x).float().clone().requires_grad_(False)
  elif type(x) is torch.Tensor:
    x_org = x.clone().requires_grad_(False)
  dim_x = x_org.size(0)

  S = list(S)
  nonS = _compset(S, dim_x)
  dim_nonS = len(nonS)

  if not gaussian:
    x_org_binary = x_org.clone()
    x_org = x_org*10.0 - 5.0

  x_S = x_org[S] # fixed
  x_nonS = x_org[nonS].clone().requires_grad_(True)

  def loss_function(x_now):
    torch.manual_seed(42); np.random.seed(42)
    if gaussian:
      loss = ef(x_now)[0] / dim_x
      tiny = 1e-4
      normalizer = torch.where(torch.abs(x_org)>tiny, x_org, torch.ones_like(x_org)*tiny)
      reldist = torch.pow((x_now-x_org)/normalizer, 2).sum() / dim_nonS
    else:
      loss = ef(torch.sigmoid(x_now))[0] / dim_x
      reldist = torch.nn.functional.binary_cross_entropy(
        torch.sigmoid(x_now), x_org_binary, reduction='sum') / dim_nonS
    loss += regparam*torch.nn.functional.relu(reldist-margin)
    return loss

  def get_x_now(x_nonS_):
    x_now = torch.zeros(dim_x)
    x_now[S] = x_S
    x_now[nonS] = x_nonS_
    return x_now

  optimizer = lbfgsnew.LBFGSNew([x_nonS,], lr=learnrate)

  # try LBFGS first
  LBFGS_failed = False
  LBFGS_converged = False
  loss_value = 1e10
  for i in range(itermax):
    # save to recover from it
    x_nonS_save = x_nonS.clone()

    # try several times till we avoid nan
    for j in range(50):
      def closure():
        if torch.is_grad_enabled():
          optimizer.zero_grad()
        loss = loss_function(get_x_now(x_nonS))
        if loss.requires_grad:
          loss.backward()
        return loss
      loss = optimizer.step(closure)
      if torch.isnan(x_nonS).any():
        # print('nan detected; roll back to previous step')
        x_nonS = x_nonS_save.clone().requires_grad_(True)
      else:
        break

    # if still nan, we give up
    if torch.isnan(x_nonS).any():
      # print('give up LBFGS; roll back to original')
      x_nonS = x_org[nonS].clone().requires_grad_(True)
      LBFGS_failed = True
      break

    # check convergence
    loss_value_new = loss.detach().item()
    reldif = abs(loss_value - loss_value_new) / abs(loss_value)
    loss_value = loss_value_new
    if reldif<tolreldif:
      LBFGS_converged = True
      break

  # if not LBFGS_converged:
  #   print('LBFGS did not converged well')

  # check divergence
  torch.manual_seed(42); np.random.seed(42)
  x_now = get_x_now(x_nonS).detach()
  if gaussian:
    eval_init = ef(x_org)[0].detach().item()
    eval_now = ef(x_now)[0].detach().item()
  else:
    eval_init = ef(x_org_binary)[0].detach().item()
    eval_now = ef(torch.sigmoid(x_now))[0].detach().item()

  if eval_now != eval_now:
    # print('nan detected; roll back to original')
    x_nonS = x_org[nonS].clone().requires_grad_(True)
    LBFGS_failed = True

  if eval_now > eval_init:
    # print('e(x) did not decreased (now=%e, init=%e); roll back to original'\
    #   % (eval_now, eval_init))
    x_nonS = x_org[nonS].clone().requires_grad_(True)
    LBFGS_failed = True

  # if LBFGS fails, try Adam
  if LBFGS_failed or not LBFGS_converged:
    # print('LBFGS faild or not converged; try adam')
    optimizer = torch.optim.Adam([x_nonS,], lr=1e-2)

    loss_value = 1e10
    for i in range(5000):
      # update
      optimizer.zero_grad()
      loss = loss_function(get_x_now(x_nonS))
      loss.backward()
      optimizer.step()

      # check convergence
      loss_value_new = loss.detach().item()
      reldif = abs(loss_value - loss_value_new) / abs(loss_value)
      loss_value = loss_value_new
      if reldif<tolreldif:
        break

  # return result
  x_last = torch.zeros(dim_x)
  x_last[S] = x_S.detach()
  x_last[nonS] = x_nonS.detach()

  if not gaussian:
    x_last = torch.sigmoid(x_last)

  if type(x) is np.ndarray:
    return x_last.numpy()
  else:
    return x_last


######################


def gaussian_energy(x, dim_x, mean, prec, logdet_cov):
  '''
  Energy of x under Gaussian distribution
  '''

  x = x.reshape(-1, dim_x)

  if type(x) is np.ndarray:
    sum_ = lambda x: np.sum(x, axis=1)
  else:
    sum_ = lambda x: torch.sum(x, dim=1)
  answer = 0.5*sum_((x-mean) * (prec@(x-mean).T).T)
  answer += 0.5*dim_x*LOG2PI + 0.5*logdet_cov
  return answer


def gaussian_marginenergy(S, x, dim_x, mean, cov, prec, logdet_cov):
  '''
  Energy of x[S] under marginal of Gaussian distribution
  '''

  x = x.reshape(-1, dim_x)

  if len(S)==dim_x:
    return gaussian_energy(x, dim_x, mean, prec, logdet_cov)

  new_mean = mean[S]
  new_cov = cov[np.ix_(S, S)]
  new_prec = prec[np.ix_(S, S)]
  _, new_logdet_cov = np.linalg.slogdet(new_cov)
  return gaussian_energy(x[:,S], len(S), new_mean, new_prec, new_logdet_cov)


def gaussian_allmarginenergy(x, dim_x, mean, prec):
  '''
  Energies of each feature of x under its marginal Gaussian distribution
  '''

  x = x.reshape(-1, dim_x)

  diag_prec = np.diag(prec)[np.newaxis,:]
  answer = 0.5*diag_prec * np.power(x-mean, 2)
  answer += 0.5*dim_x*LOG2PI - 0.5*np.log(diag_prec)
  return answer


def gaussian_energy_condexpt(S, x, r, dim_x, mean, cov, prec, logdet_cov):
  '''
  Energy of x[S] under conditional Gaussian distribution, conditioned on x[nonS]
  '''

  x = x.reshape(-1, dim_x)
  r = r.reshape(-1, dim_x)
  assert x.shape==r.shape

  nonS = _compset(S, dim_x)
  mu = mean
  Sigma = cov
  L = prec

  x_S = x[:,S]
  x_nonS = x[:,nonS]
  mu_S = mu[S]
  mu_nonS = mu[nonS]
  L_S_S = L[np.ix_(S, S)]
  L_S_nonS = L[np.ix_(S, nonS)]
  L_nonS_nonS = L[np.ix_(nonS, nonS)]

  # constant term
  answer = np.zeros(x.shape[0]) + 0.5*dim_x*LOG2PI + 0.5*logdet_cov

  # TODO below can be improved because e.g. Sigma_cond_nonS_S is same for every x_S
  for i in range(x.shape[0]):
    mu_cond_nonS_S, Sigma_cond_nonS_S = \
      _gaussian_condparams_cov(nonS, S, r[i], mu, Sigma)

    if len(S)>0:
      tmp1 = L_S_S @ np.tensordot(x_S[i]-mu_S, x_S[i]-mu_S, 0)
      answer[i] += 0.5*np.trace(tmp1)
    if len(nonS)>0 and len(S)>0:
      tmp2 = L_S_nonS @ np.tensordot(mu_cond_nonS_S-mu_nonS, x_S[i]-mu_S, 0)
      answer[i] += np.trace(tmp2)
    if len(nonS)>0:
      # constant term
      tmp3 = L_nonS_nonS @ (Sigma_cond_nonS_S +
        np.tensordot(mu_cond_nonS_S-mu_nonS, mu_cond_nonS_S-mu_nonS, 0))
      answer[i] += 0.5*np.trace(tmp3)

  return answer


######################


def gmm_energy(x, dim_x, gmm_model):
  '''
  Energy of x under GMM
  '''

  return gmm_marginenergy(range(dim_x), x, dim_x, gmm_model)


def gmm_marginenergy(S, x, dim_x, gmm_model):
  '''
  Energy of x[S] under marginal of GMM
  '''

  x = x.reshape(-1, dim_x)

  K = gmm_model.weights_.shape[0]
  N = x.shape[0]

  if type(x) is np.ndarray:
    answer = np.zeros((K, N))
    for i in range(K):
      answer[i] = -gaussian_marginenergy(S, x, dim_x,
        gmm_model.means_[i], gmm_model.covariances_[i], gmm_model.precisions_[i],
        gmm_model.logdet_covs_[i])
      answer[i] += np.log(gmm_model.weights_[i])
    return -logsumexp(answer, axis=0, keepdims=False, return_sign=False)
  else:
    answer = torch.zeros(K, N)
    for i in range(K):
      answer[i] = -gaussian_marginenergy(S, x, dim_x,
        torch.from_numpy(gmm_model.means_[i]).float(),
        torch.from_numpy(gmm_model.covariances_[i]).float(),
        torch.from_numpy(gmm_model.precisions_[i]).float(),
        torch.from_numpy(np.array([gmm_model.logdet_covs_[i]])).float())
      answer[i] += torch.log( torch.tensor([gmm_model.weights_[i]]) )
    return -torch.logsumexp(answer, dim=0, keepdim=False)


def gmm_allmarginenergy(x, dim_x, gmm_model):
  '''
  Energies of each feature of x under its marginal of GMM
  '''

  x = x.reshape(-1, dim_x)

  K = gmm_model.weights_.shape[0]

  answer = np.zeros((K, x.shape[0], dim_x))
  for i in range(K):
    answer[i] = -gaussian_allmarginenergy(x, dim_x,
      gmm_model.means_[i], gmm_model.precisions_[i])
    answer[i] += np.log(gmm_model.weights_[i])
  return -logsumexp(answer, axis=0, keepdims=False, return_sign=False)
