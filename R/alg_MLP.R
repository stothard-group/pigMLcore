# -----------------------------------------------------------------------------
# R FILE             --Tianfu Yang
# Type:              Functions
# Subtype/Project:   pigMLcore
# Descriptions:
#   The file provides functions that enable xgboost prediction
# -----------------------------------------------------------------------------
# Contents:
# -----------------------------------------------------------------------------
# To-do
# -----------------------------------------------------------------------------
# Pre-load
# -----------------------------------------------------------------------------
# rm(list = ls())

# -----------------------------------------------------------------------------
# FUNCTION:     calculate_search_grid_mlpL2
#' DESCRIPTION
#' @rdname      calculate_search_grid_mlpL2
#' @title       calculate the whole grid for MLP with L2 penalty
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       n_size       number of size in grid search,
#' @param       n_lambda     number of lambda in grid search,
#' @return      a data frame
# -----------------------------------------------------------------------------
# @export
#' @note        A search grid calculated based on the input data
# @examples    \
# -----------------------------------------------------------------------------
calculate_search_grid_mlpL2 = function(y, X, n_size, n_lambda){
  res_grid = expand.grid(
    size        = floor(seq(3, ncol(X), length.out = n_size)),
    lambda      = c(0, 10 ^ seq(0, -4, length = n_lambda - 1)),
    batch_size  = 2*4^(0:(n_size -1)),
    lr          = 10^(-4),
    rho         = 0.9,
    decay       = 0,
    activation  = "relu"
  )
  return(res_grid)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # reg_mlpL2
#' DESCRIPTION
#' @rdname      reg_mlpL2
#' @title       Prediction with XGBTree and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       mlpL2_TC trControl that pass to caret
#' @param       mlpL2_TG tuneGrid that pass to caret
#' @param       n_size   number of size in grid search,
#' @param       n_lambda number of lambda in grid search,
#' @param       epoch    epoch in the training
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using xgboost through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_mlpL2 = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  mlpL2_TC  = trainControl(method = "cv", number = 2, verboseIter = F),
  mlpL2_TG  = NULL,
  n_size    = 5,
  n_lambda  = 3,
  epoch     = 200
){
  # Checking data
  seed      = check_input_seed(seed = seed)
  idx_valid = check_input_data(y =y, X = X)
  y = y[idx_valid]; X = X[idx_valid,]
  foldset   = check_input_foldset(foldset = foldset, y = y, K = K)
  
  # Cross-validation
  if (pb) pbi = txtProgressBar(min = 0, max=K, style = 3)
  hat_y = rep(NA, length(y))
  model_fit_list = vector("list", K)

  for (i in 1:K){
    idx_test = foldset[[i]]
    transformed = pre_prosess_in_CV(X = X, idx_test = idx_test, n_pca = n_pca)
    if (is.null(mlpL2_TG)){
      mlpL2_TG = calculate_search_grid_mlpL2(
        y = y[-idx_test],
        X = transformed$training,
        n_size    = n_size,
        n_lambda  = n_lambda
      )
    }
    model_fit_list[[i]] = train(
      x = transformed$training, y = y[-idx_test],
      method = "mlpKerasDecay", tuneGrid  = mlpL2_TG, trControl = mlpL2_TC,
      verbose = 0,
      epoch = epoch
    )
    if (pb) {setTxtProgressBar(pbi, i)} # Predict
    hat_y[idx_test] = predict(model_fit_list[[i]], newdata = transformed$testing)
  }

  # Results
  res = list(
    meta = list(
      job_name   = job_name,
      seed       = seed,
      foldset    = foldset,
      n_pca      = n_pca
    ),
    hyper = list(
      TC = mlpL2_TC,
      TG = mlpL2_TG,
      search_par = list(
        n_size  ,
        n_lambda
      )
    ),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}
