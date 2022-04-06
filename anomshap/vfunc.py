import numpy as np

class VF:
  def __init__(self, e_hat, e_th):
    self.e_hat = e_hat
    self.e_th = e_th
    self.count = 0

  def __call__(self, S):
    self.count += 1
    e_hat_ = self.e_hat(S)
    return np.fmin(e_hat_, self.e_th), e_hat_>=self.e_th

  def zero_count(self):
    self.count=0
