data {
  int<lower=0> N;       // number of cases/data points
  int<lower=0> G;       // number of groups
  int<lower=0> I;       // number of individuals
  matrix[N, G+I] x;     // matrix of predictor variables
  vector[N] y;          // outcome/response variable
}
parameters {
  real<lower=0, upper=1> g_prob;
  real<lower=0> sigma;
}
transformed parameters {
  vector[G+I] g_i_prob;
  g_i_prob = to_vector(append_array(rep_array(g_prob, G), rep_array(1-g_prob, I)));
}
model {
  g_prob ~ uniform(0, 1);
  sigma ~ uniform(0, 5);
  y ~ normal(x * g_i_prob, sigma);
}
