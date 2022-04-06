import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics


def plot_attr(attr, anofeats=None):
  for i, key in enumerate(attr):
    tmp = attr[key]
    #tmp = attr[key] / np.max(np.abs(attr[key]))
    ax = plt.subplot(len(attr),1,i+1)
    ax.bar(range(tmp.size), tmp, label=key)
    if anofeats is not None:
      ax.bar(anofeats, tmp[anofeats])
    ax.legend()
  plt.show()


def hitsatk(attr_all, feat_all, k):
  assert feat_all[0][0].size==1

  keys = list(attr_all[0][0].keys())
  correct = {}; total = {}
  for key in keys:
    correct[key] = 0.0
    total[key] = 0.0

  for i in range(len(attr_all)):
    for j in range(len(attr_all[i])):
      truth = feat_all[i][j].item()
      for key in keys:
        if attr_all[i][j][key] is None:
          continue
        score = np.nan_to_num(attr_all[i][j][key], nan=1.0)
        detected = np.argsort(score)[::-1][:k]
        if truth in detected:
          correct[key] += 1.0
        total[key] += 1.0
  return correct, total


def score_to_rank(score):
  o = np.argsort(score)[::-1]
  rank = np.zeros(score.size)
  c = 1
  for i in range(score.size):
      rank[o[i]] = c
      c += 1
  return rank


def recrank(attr_all, feat_all):
  assert feat_all[0][0].size==1

  keys = list(attr_all[0][0].keys())
  recrank = {}
  for key in keys:
    recrank[key] = []

  for i in range(len(attr_all)):
    for j in range(len(attr_all[i])):
      truth = feat_all[i][j].item()
      for key in keys:
        if attr_all[i][j][key] is None:
          continue
        score = np.nan_to_num(attr_all[i][j][key], nan=1.0)
        rank = score_to_rank(score)
        recrank[key].append(1.0/rank[truth])
  return recrank


def auc(attr_all, feat_all):
  keys = list(attr_all[0][0].keys())
  if type(attr_all[0][0][keys[0]]) is np.ndarray:
    num_feat = attr_all[0][0][keys[0]].size
  else:
    raise ValueError('unknown attr container type')
  auc = {}
  for key in keys:
    auc[key] = []

  for i in range(len(attr_all)):
    for j in range(len(attr_all[i])):
      truth = np.zeros(num_feat)
      truth[feat_all[i][j]] = 1
      for key in keys:
        if attr_all[i][j][key] is None: continue
        score = attr_all[i][j][key]
        auc[key].append( sklearn.metrics.roc_auc_score(
          truth, np.nan_to_num(score, nan=1.0), average='macro') )
  return auc
