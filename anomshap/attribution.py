import copy
import time
import warnings

import numpy as np
import shap

from . import vfunc, efunc, shval

class Attributer():
  def __init__(self, dim_x, model, model_type, score_type,
    trdata_example=None, trdata_mean=None):

    self.dim_x = dim_x
    self.model = model
    self.model_type = model_type
    self.score_type = score_type
    self.trdata_example = trdata_example
    self.trdata_mean = trdata_mean

    if self.trdata_example is not None:
      self.trdata_example = self.trdata_example.reshape(-1, self.dim_x)

    self.allmarginscorer = None
    self.gaussian = True

    if model_type=='gmm' and score_type=='energy':
      self.ef = efunc.F(lambda x: efunc.gmm_energy(x, dim_x, model))
      self.allmarginscorer = lambda x: efunc.gmm_allmarginenergy(x, dim_x, model)[0]
    else:
      raise ValueError('Unknown setting: model_type=%s, score_type=%s' % (model_type, score_type))


  def evalue(self, data):
    data = data.reshape(-1, self.dim_x)
    return self.ef(data)


  def attribute(
    self, datum, e_th, methods,
    seed=1234567890,
    kernshap_num_sample='auto',
    anomshap_num_sample='auto',
    anomshap_bl_regparam=1e-2,
    anomshap_bl_learnrate=0.1,
    anomshap_bl_maxiter=500,
  ):

    assert datum.ndim==1, 'only a single data point can be treated'
    assert datum.size==self.dim_x

    if not (type(methods) is list or type(methods) is tuple):
      methods = [methods,]

    if anomshap_num_sample=='auto':
      anomshap_num_sample = self.dim_x*2 + 2048

    attr = {}
    info = {}

    # attribution by margin score
    if 'margscore' in methods:
      if self.allmarginscorer is not None:
        self.ef.zero_count(); start = time.time()
        attr['margscore'] = self.allmarginscorer(datum)
        info['margscore_duration'] = time.time()-start
        info['margscore_funcount'] = self.ef.count
      else:
        warnings.warn('margscore is specified in methods, but no allmarginscorer is prepared')

    # attribution by KernelSHAP
    if 'kernshap' in methods:
      if self.trdata_example is not None:
        np.random.seed(seed)
        self.ef.zero_count(); start = time.time()
        elr = shap.KernelExplainer(lambda x: np.fmin(self.ef(x),e_th), self.trdata_example,
          link='identity' if self.gaussian else 'logit')
        attr['kernshap'] = elr.shap_values(datum, nsamples=kernshap_num_sample, l1_reg=0)
        info['kernshap_duration'] = time.time()-start
        info['kernshap_funcount'] = self.ef.count
      else:
        warnings.warn('kernshap is specified in methods, but no trdata_example is provided')

    # attribution by AnomSHAP (Monte Carlo)
    if 'anomshap' in methods:
      kwargs_to_bline = {
        'gaussian':self.gaussian,
        'regparam':anomshap_bl_regparam,
        'margin':0.0,
        'itermax':anomshap_bl_maxiter,
        'learnrate':anomshap_bl_learnrate,
        'tolreldif':1e-3,
      }

      self.ef.zero_count(); start = time.time()
      optbaseliner = efunc.general_optbaseline_approx2(datum, self.dim_x, self.ef, kwargs_to_bline)
      vf = vfunc.VF(optbaseliner, e_th)
      elr = shval.wls(self.dim_x, anomshap_num_sample, seed=seed)
      attr['anomshap'], _ = elr(vf, do_bound=False)
      info['anomshap_duration'] = time.time()-start
      info['anomshap_funcount_e'] = self.ef.count
      info['anomshap_funcount_v'] = vf.count

    # attribution by AnomSHAP (Monte Carlo) without the relaxation
    if 'anomshap_norelax' in methods:
      kwargs_to_bline = {
        'gaussian':self.gaussian,
        'regparam':anomshap_bl_regparam,
        'margin':0.0,
        'itermax':anomshap_bl_maxiter,
        'learnrate':anomshap_bl_learnrate,
        'tolreldif':1e-3,
      }

      self.ef.zero_count(); start = time.time()
      optbaseliner = lambda S: efunc.general_optbaseline(S, datum, self.dim_x, self.ef, kwargs_to_bline)
      vf = vfunc.VF(optbaseliner, e_th)
      elr = shval.wls(self.dim_x, anomshap_num_sample, seed=seed)
      attr['anomshap_norelax'], _ = elr(vf, do_bound=False)
      info['anomshap_norelax_duration'] = time.time()-start
      info['anomshap_norelax_funcount_e'] = self.ef.count
      info['anomshap_norelax_funcount_v'] = vf.count

    return attr, info
