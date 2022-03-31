# -----------------------------------------------------------------------------
# R FILE             --Tianfu Yang
# Type:              Functions
# Subtype/Project:   pigMLcore
# Descriptions:
#   The file provides functions that enable Elastic Net prediction
# -----------------------------------------------------------------------------
# Contents:
# -----------------------------------------------------------------------------
# To-do
# -----------------------------------------------------------------------------
# Pre-load
# -----------------------------------------------------------------------------
# rm(list = ls())
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# FUNCTION:     calculate_lambda_max
#' DESCRIPTION
#' @rdname      calculate_lambda_max
#' @title       calculate data-based lambda.max
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       alpha the alpha hyperparameter
#' @return      \item{name a}{description}
# -----------------------------------------------------------------------------
# @export      \
#' @note       The function calculate the minimal lambda that shrinks all
#'             regression coefficients into zero (lambda.max) based on the
#'             data:
# L = \max_j\frac{1}{\alpha n} \sum_{i=1}^n [Y_i-\bar Y(1-\bar Y)] X_{ij}
# @examples    \
# -----------------------------------------------------------------------------
calculate_lambda_max = function(y, X, alpha){
  if (!is.numeric(y)) y = as.numeric(y)
  if (alpha > 1) stop("alpha cannot be greater than 1.")
  if (alpha < 0) stop("alpha cannot be negative.")
  alpha = max(alpha, 0.001) ## alpha cannot be zero in the calculation.
  lambda_max = max(abs(t(y - mean(y)*(1-mean(y))) %*% X)) / (alpha*nrow(X))
  return(lambda_max)
}

# -----------------------------------------------------------------------------
# FUNCTION:     calculate_search_grid_en
#' DESCRIPTION
#' @rdname      calculate_search_grid_en
#' @title       calculate the whole grid for lasso
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       n_alpha number of alpha tested in the grid
#' @param       n_lambda number of lambda tested in the grid
#' @param       lambda_ratio Smallest value for lambda, as a fraction of lambda.max,
#' @return      a data frame
# -----------------------------------------------------------------------------
# @export
#' @note        A search grid calculated based on the input data
# @examples    \
# -----------------------------------------------------------------------------
calculate_search_grid_en = function(y, X, n_alpha, n_lambda, lambda_ratio = NULL){
  if (is.null(lambda_ratio)){
    lambda_ratio = ifelse(ncol(X) > nrow(X), 0.0001, 0.01)
  }
  res_grid = data.frame(alpha = numeric(0), lambda = numeric(0))
  alpha_list = seq(from = 0, to = 1, length.out = n_alpha)
  for (i in 1:n_alpha){
    temp_alpha = alpha_list[i]
    temp_lambda_max = calculate_lambda_max(y, X, temp_alpha)
    temp_log2_lambda_max = log2(temp_lambda_max)
    temp_log2_lambda_min = log2(temp_lambda_max * lambda_ratio)
    temp_log2_lambda_list =
      2^(seq(from = temp_log2_lambda_min,
             to = temp_log2_lambda_max,
             length.out = n_lambda))
    res_grid = rbind(res_grid,
      data.frame(alpha = temp_alpha, lambda = temp_log2_lambda_list))
  }
  return(res_grid)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # regression_en
#' DESCRIPTION
#' @rdname      reg_en
#' @title       Prediction with EN and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       lasso_TC trControl that pass to caret
#' @param       lasso_TG tuneGrid that pass to caret
#' @param       n_alpha number of alpha tested in the grid
#' @param       n_lambda number of lambda tested in the grid
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using EN through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_en = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  lasso_TC  = trainControl(method = "cv", number = 5, verboseIter = F),
  lasso_TG  = NULL,
  n_alpha   = 10,
  n_lambda  = 100,
  nc        = 1
){
  # Checking data
  seed      = check_input_seed(seed = seed)
  idx_valid = check_input_data(y =y, X = X)
  y = y[idx_valid]; X = X[idx_valid,]
  foldset   = check_input_foldset(foldset = foldset, y = y, K = K)

  # Cross-validation - preparation
  if (pb) pbi = txtProgressBar(min = 0, max=K, style = 3)
  hat_y = rep(NA, length(y))
  model_fit_list = vector("list", K)

  # Cross-validation
  for (i in 1:K){
    idx_test = foldset[[i]]
    transformed = pre_prosess_in_CV(X = X, idx_test = idx_test, n_pca = n_pca)
    if (is.null(lasso_TG)){
      lasso_TG = calculate_search_grid_en(
        y = y[-idx_test], X = transformed$training,
        n_alpha = n_alpha, n_lambda = n_lambda)
    }

    # Cross-validation - training
    cl = parallel::makePSOCKcluster(nc)
    doParallel::registerDoParallel(cl)
    model_fit_list[[i]] = train(
      x = transformed$training, y = y[-idx_test],
      method = "glmnet", tuneGrid  = lasso_TG, trControl = lasso_TC
    )
    parallel::stopCluster(cl)

    # Cross-validation - Testing
    if (pb) {setTxtProgressBar(pbi, i)} # Predict
    hat_y[idx_test] = predict(model_fit_list[[i]], newdata = transformed$testing)
  }

  # Results
  res = list(
    meta = list(
      job_name = job_name, seed = seed, foldset = foldset, n_pca = n_pca
    ),
    hyper = list(TC = lasso_TC, TG = lasso_TG, search_par = list(n_alpha, n_lambda)),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )

  return(res)
}

## The following function tries to use flexible lambda based on alpha. ========
## NOT FINISHED
# get_optimal_alpha = function(y, X, n_alpha, K){
#   alphas = seq(from = 0, to = 1, length.out= n_alpha)
#   res_prediction = numeric(n_alpha)
#   partitions = caret::createFolds(y, K)
#   for (i in 1:n_alpha){
#     temp_alpha = alpha[i]
#     for (fold in 1:K){
#       temp_training_X = X[-(partitions[[fold]]),]
#       temp_testing_X  = X[partitions[[fold]],]
#     }
#   }
# }
#==============================================================================
# -----------------------------------------------------------------------------
# FUNCTION:     # cla_en
#' DESCRIPTION
#' @rdname      cla_en
#' @title       Prediction with EN and evaluation through CV - classification
#' @param       y the responding variable - binary
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       lasso_TC trControl that pass to caret
#' @param       lasso_TG tuneGrid that pass to caret
#' @param       n_alpha number of alpha tested in the grid
#' @param       n_lambda number of lambda tested in the grid
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using EN through CV.
# @examples    \
# -----------------------------------------------------------------------------
cla_en = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  lasso_TC  = trainControl(method = "cv", number = 5, verboseIter = F),
  lasso_TG  = NULL,
  n_alpha   = 10,
  n_lambda  = 100,
  nc        = 1
){
  # Checking data
  seed      = check_input_seed(seed = seed)
  idx_valid = check_input_data(y =y, X = X)
  y = y[idx_valid]; X = X[idx_valid,]
  foldset   = check_input_foldset(foldset = foldset, y = y, K = K)

  # Cross-validation - preparation
  if (pb) pbi = txtProgressBar(min = 0, max=K, style = 3)
  hat_y = y
  model_fit_list = vector("list", K)

  # Cross-validation
  for (i in 1:K){
    idx_test = foldset[[i]]
    transformed = pre_prosess_in_CV(X = X, idx_test = idx_test, n_pca = n_pca)
    if (is.null(lasso_TG)){
      lasso_TG = calculate_search_grid_en(
        y = y[-idx_test], X = transformed$training,
        n_alpha = n_alpha, n_lambda = n_lambda)
    }

    # Cross-validation - training
    cl = parallel::makePSOCKcluster(nc)
    doParallel::registerDoParallel(cl)
    model_fit_list[[i]] = train(
      x = transformed$training, y = y[-idx_test],
      method = "glmnet", tuneGrid  = lasso_TG, trControl = lasso_TC
    )
    parallel::stopCluster(cl)

    # Cross-validation - Testing
    if (pb) {setTxtProgressBar(pbi, i)} # Predict
    hat_y[idx_test] = predict(model_fit_list[[i]], newdata = transformed$testing)
  }

  # Results
  res = list(
    meta = list(
      job_name = job_name, seed = seed, foldset = foldset, n_pca = n_pca
    ),
    hyper = list(TC = lasso_TC, TG = lasso_TG, search_par = list(n_alpha, n_lambda)),
    model = model_fit_list,
    accuracy   = caret::confusionMatrix(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )

  return(res)
}