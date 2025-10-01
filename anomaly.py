import numpy as np

class AnomalyScore():
  def __init__(self, lift_weight=0.2, kl_weight=0.2, gini_weight=0.2, support_weight=0.2, tfidf_weight=0.2, min_score=0.0):
    tot_weight = lift_weight + kl_weight + gini_weight + support_weight + tfidf_weight
    if tot_weight > 0:
      self.lift_weight = lift_weight / tot_weight
      self.kl_weight = kl_weight / tot_weight
      self.gini_weight = gini_weight / tot_weight
      self.support_weight = support_weight / tot_weight
      self.tfidf_weight = tfidf_weight / tot_weight
      self.min_score = min_score
    else:
      print("The sum of weights has to be > 0")
      raise ValueError

  def _calculate_base_score_for_binned_array(self, x, y):
  
    global_pos_rate = np.mean(y)
    mean_support = len(x) / len(list(set(x)))
    score = {}
    for bin in list(set(x)):
      mask = x == bin
      support = sum((mask).astype(int)) / mean_support
  
      pos = sum(((mask) & (y == 1)).astype(int))
      neg = sum(((mask) & (y == 0)).astype(int))
  
      p1 = pos / (pos + neg)
      p0 = 1 - p1
      q1 = global_pos_rate
      q0 = 1 - q1
      gini = 1 - (p1**2 + p0**2)
  
      eps = 1e-12
      kl_div = 0
      if p1 > 0:
        kl_div += p1 * np.log((p1 + eps) / (q1 + eps))
      if p0 > 0:
        kl_div += p0 * np.log((p0 + eps) / (q0 + eps))
      lift = p1 / q1 if q1 > 0 else np.nan
      score[bin] = support * self.support_weight + lift * self.lift_weight + gini * self.gini_weight + kl_div * self.kl_weight
    return score

  def _calculate_tfidf_for_binned_array(self, x, dates):
    order = np.argsort(dates)
    x_sorted = x[order]
    
    tfidf_sorted = np.zeros_like(x, dtype=float)
    bin_counts = {}
    
    for idx, bin_val in enumerate(x_sorted, start=1):
      bin_counts[bin_val] = bin_counts.get(bin_val, 0) + 1
      freq = bin_counts[bin_val]
      tfidf_sorted[idx - 1] = np.log(idx / freq)
    
    tfidf = np.empty_like(tfidf_sorted)
    tfidf[order] = tfidf_sorted
    
    return tfidf

  def _calcualte_full_score_for_binned_array(self, x, y, dates):
    score = self._calculate_base_score_for_binned_array(x, y)
    x_score = [score.get(z, 0) for z in x]
    x_score = np.array(x_score, dtype=float)
    tfidf = self._calculate_tfidf_for_binned_array(x, dates)
    tfidf *= self.tfidf_weight
    final_score = x_score + tfidf
    final_score = np.where(final_score >= self.min_score, final_score, 0.0)
    return final_score

def fit_transform(self, X, y, dates):
  X = np.array(X)
  X_transformed = []
  for i in range(X.shape[1]):
    x = X[:,i]
    X_transformed.append(self._calculate_full_score_for_binned_array(x, y, dates))
  X_transformed = np.array(X_transformed).T
  return X_transformed
    

