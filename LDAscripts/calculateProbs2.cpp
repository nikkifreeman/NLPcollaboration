//[[Rcpp::depends(RcppArmadillo)]]

# include <RcppArmadilloExtensions/sample.h>
# include <RcppArmadilloExtensions/fixprob.h>

using namespace Rcpp;

// [[Rcpp::export]]
double calculate_a_over_b( NumericMatrix Z, NumericMatrix W, int m, int n, int k_tilde, NumericVector alpha){
  
  // Calculate a
  NumericVector Zm = Z(m - 1, _);
  NumericVector Wm = W(m - 1, _);
  LogicalVector WmTimesZmEqTok = (Wm == 1) & (Zm == k_tilde);
  WmTimesZmEqTok(n - 1) = NA_LOGICAL;
  double a = alpha(k_tilde - 1) + sum(na_omit(WmTimesZmEqTok));
  
  // Calculate b
  NumericVector Wm_n = Wm;
  Wm_n(n - 1) = 0;
  double b = sum(alpha) + sum(na_omit(Wm_n));
  
  return a/b;
}

// [[Rcpp::export]]
double calculate_c_over_d(NumericMatrix W, NumericMatrix Z, int m, int n, int k_tilde, NumericVector delta){
  
  // Calculate c
  NumericVector Wn = W(_, n - 1);
  NumericVector Zn = Z(_, n - 1);
  LogicalVector WnTimesZnEqTok = (Wn == 1) & (Zn == k_tilde);
  WnTimesZnEqTok(m - 1) = NA_LOGICAL;
  double c = delta(k_tilde - 1) + sum(na_omit(WnTimesZnEqTok));
  
  // Calculate d
  arma::mat Wmat = as<arma::mat>(W);
  arma::mat Zmat = as<arma::mat>(Z);
  arma::mat ZTimesWmat = (Zmat == k_tilde) % Wmat;
  ZTimesWmat(m - 1, n - 1) = NA_REAL;
  NumericMatrix ZTimesW = wrap(ZTimesWmat);
  double d = sum(delta) + sum(na_omit(ZTimesW));
  
  return c/d;
}



// [[Rcpp::export]]
NumericVector calculate_prob_vec(NumericMatrix Z, NumericMatrix W, int m, int n, int K, 
                          NumericVector alpha, NumericVector delta){
  NumericVector probs(K);
  for(int k =0; k<K; k++){
    
    int k_tilde = k + 1;
    
    double first = calculate_a_over_b(Z, W,  m, n, k_tilde, alpha);
    double second = calculate_c_over_d(W, Z, m, n, k_tilde, delta);
    probs(k) = first*second;
  }
  
  probs = probs/sum(probs);
  
  return probs;
}





  
   
  