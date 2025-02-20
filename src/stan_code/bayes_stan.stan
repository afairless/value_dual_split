data {
  int<lower=0> N;       // number of cases/data points
  int<lower=0> G;       // number of groups
  int<lower=0> I;       // number of individuals
  matrix[N, G] g_x;     // matrix of group identifiers
  matrix[N, I] i_x;     // matrix of individual identifiers
  vector[N] y;          // outcome/response variable
}
parameters {
  real<lower=0, upper=1> g_prob;
  vector<lower=0, upper=1>[G] g_v;
  vector<lower=0, upper=1>[I] i_v;
}
transformed parameters {
  vector[2] ps;
  ps[1] = g_prob;
  ps[2] = 1 - g_prob;
}
model {
  g_prob ~ normal(0.5, 0.3);
  g_v ~ normal(0.5, 0.3);
  i_v ~ normal(0.5, 0.3);
  y ~ normal(append_col(g_x * g_v, i_x * i_v) * ps, 0.1);
}
