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
# FUNCTION:     calculate_search_grid_xgbl
#' DESCRIPTION
#' @rdname      calculate_search_grid_xgbl
#' @title       calculate the whole grid for xgblinear
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       n_rounds number of # boosting iterations
#' @param       n_reg   number of tested regularization parameter
#' @return      a data frame
# -----------------------------------------------------------------------------
# @export
#' @note        A search grid calculated based on the input data
# @examples    \
# -----------------------------------------------------------------------------
calculate_search_grid_xgbl = function(y, X, n_rounds, n_reg){
  if (n_rounds <= 5) {stop("n_round should be larger than 5")}
  res_grid = expand.grid(
    nrounds = c(seq(from = 4, to = 20, by = 4),   ## length of
                seq(from = 40, by = 40, length.out = n_rounds - 5)),
    lambda  = c(0, 2 ^ seq(0, -12, length = n_reg)),
    alpha   = c(0, 2 ^ seq(0, -12, length = n_reg)),
    eta     = seq(from = 0.01, to = 0.3, length.out = 5)
  )
  return(res_grid)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # reg_xgbl
#' DESCRIPTION
#' @rdname      reg_xgbl
#' @title       Prediction with XGBlinear and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       xgbl_TC trControl that pass to caret
#' @param       xgbl_TG tuneGrid that pass to caret
#' @param       n_rounds number of # boosting iterations
#' @param       n_reg   number of tested regularization parameter
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using xgboost through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_xgbl = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  xgbl_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  xgbl_TG   = NULL,
  n_rounds  = 10,
  n_reg     = 5,
  nc        = 1
){
  # Checking data
  seed      = check_input_seed(seed = seed)
  idx_valid = check_input_data(y =y, X = X)
  y = y[idx_valid]; X = X[idx_valid,]
  foldset   = check_input_foldset(foldset = foldset, y = y, K = K)

  if (pb) pbi = txtProgressBar(min = 0, max=K, style = 3)
  hat_y = rep(NA, length(y))
  model_fit_list = vector("list", K)

  for (i in 1:K){
    idx_test = foldset[[i]]
    transformed = pre_prosess_in_CV(X = X, idx_test = idx_test, n_pca = n_pca)
    if (is.null(xgbl_TG)){
      xgbl_TG = calculate_search_grid_xgbl(
        y = y[-idx_test], X = transformed$training,
        n_rounds = n_rounds, n_reg = n_reg)
    }

    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "xgbLinear", tuneGrid  = xgbl_TG, trControl = xgbl_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "xgbLinear", tuneGrid  = xgbl_TG, trControl = xgbl_TC
      )
    }
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
      TC = xgbl_TC,
      TG = xgbl_TG,
      search_par = list(
        n_rounds,
        n_reg
      )
    ),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # cla_xgbl
#' DESCRIPTION
#' @rdname      cla_xgbl
#' @title       Prediction with XGBlinear and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       xgbl_TC trControl that pass to caret
#' @param       xgbl_TG tuneGrid that pass to caret
#' @param       n_rounds number of # boosting iterations
#' @param       n_reg   number of tested regularization parameter
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using xgboost through CV.
# @examples    \
# -----------------------------------------------------------------------------
cla_xgbl = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  xgbl_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  xgbl_TG   = NULL,
  n_rounds  = 10,
  n_reg     = 5,
  nc        = 1
){
  # Checking data
  seed      = check_input_seed(seed = seed)
  idx_valid = check_input_data(y =y, X = X)
  y = y[idx_valid]; X = X[idx_valid,]
  foldset   = check_input_foldset(foldset = foldset, y = y, K = K)

  if (pb) pbi = txtProgressBar(min = 0, max=K, style = 3)
  hat_y = y
  model_fit_list = vector("list", K)

  for (i in 1:K){
    idx_test = foldset[[i]]
    transformed = pre_prosess_in_CV(X = X, idx_test = idx_test, n_pca = n_pca)
    if (is.null(xgbl_TG)){
      xgbl_TG = calculate_search_grid_xgbl(
        y = y[-idx_test], X = transformed$training,
        n_rounds = n_rounds, n_reg = n_reg)
    }

    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "xgbLinear", tuneGrid  = xgbl_TG, trControl = xgbl_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "xgbLinear", tuneGrid  = xgbl_TG, trControl = xgbl_TC
      )
    }
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
      TC = xgbl_TC,
      TG = xgbl_TG,
      search_par = list(
        n_rounds,
        n_reg
      )
    ),
    model = model_fit_list,
    accuracy   = caret::confusionMatrix(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}