import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor

class AnomalyScorer():
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
    

class AnomalyEncoder():
  def __init__(self, n_svd_components, anomaly_scorer, hidden_layer_sizes=(128, 16, 128)):
    self.n_svd_components = n_svd_components
    self.anomaly_scorer = anomaly_scorer
    self.hidden_layer_sizes = hidden_layer_sizes
    
  def preprocess(self, X, default_num=0.0, default_cat='', exclusion_columns=[]):
    num_cols = []
    cat_cols = []
    cols = [col for col in X.columns if col not in exclusion_columns]
    for col in cols:
      try:
        X[col] = X[col].astype(float)
        num_cols.append(col)
        X[col] = X[col].fillna(default_num)
      except:
        X[col] = X[col].fillna(default_cat)
        cat_cols.append(col)
        X[col] = X[col].astype(str)
    if len(num_cols) > 0:
      if not hasattr(self, 'scaler'):
        scaler = StandardScaler()
        x_num = scaler.fit_transform(X[num_cols])
        self.scaler = scaler
      else:
        x_num = self.scaler.transform(X[self.scaler.feature_names_in_])
    if len(cat_cols) > 0:
      if not hasattr(self, 'ohe'):
        ohe = OneHotEncoder(sparse_output = False, min_frequency=0.001, handle_unknown='infrequent_if_exist')
        x_cat = ohe.fit_transform(X[cat_cols])
        self.ohe = ohe
      else:
        x_cat = self.ohe.transform(X[self.ohe.feature_names_in_])
    if hasattr(self, 'scaler'):
      if hasattr(self, 'ohe'):
        x_prep = np.hstack((x_num, x_cat))
      else:
        x_prep = x_num
    else:
      if hasattr(self, 'ohe'):
        x_prep = x_cat
      else:
        print("Could not continue preprocessing")
        raise
    if not hasattr(self, 'svd'):
      svd = TruncatedSVD(self.n_svd_components)
      x_svd = svd.fit_transform(x_prep)
      self.svd = svd
    else:
      x_svd = svd.transform(x_prep)
    return x_svd

  def train_encoder(self, X, y, dates):
    x_prep = self.preprocess(X)
    x_anom = self.anomaly_scorer.fit_transform(X, y, dates)
    encoder = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes)
    encoder.fit(x_prep, x_anom)
    self.encoder = encoder
    return 

  def forward_to_nth_layer(self, X, n=2):
    x_prep = self.preprocess(X)
    k = 0
    while k < n:
      x_prep = np.matmul(x_prep, self.encoder.coefs_[k]) + self.encoder.intercepts_[k]
      if self.encoder.activation == 'identity':
        x_prep = x_prep
      elif self.encoder.activation == 'logistic':
        x_prep = 1 / (1 + np.exp(x_prep))
      elif self.encoder.actiation == 'tanh':
        x_prep = np.tanh(x_prep)
      else:
        x_prep = np.maximum(x_prep, 0)
      k += 1
    return x_prep

class AnomalyClassifier():
  def __init__(self, anomaly_encoder):
    self.classifier = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    self.anomaly_encoder = anomaly_encoder

  def train(self, X, y):
    #train the classifier model on the latent representations of the encoder
    x_bottleneck = self.anomaly_encoder.forward_to_nth_layer(X, 2)
    self.classifier.fit(x_bottleneck, y)

  def predict(self, X):
    x_bottleneck = self.anomaly_encoder.forward_to_nth_layer(X, 2)
    y_proba = self.classifier.predict_proba(x_bottleneck)[:, 1]
    return y_proba
