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
# FUNCTION:     calculate_search_grid_xgbt
#' DESCRIPTION
#' @rdname      calculate_search_grid_xgbt
#' @title       calculate the whole grid for xgbtree
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       n_rounds  number of nrounds in grid search,
#' @param       n_depth   number of depth in grid search,
#' @param       n_seq     number of gamma/min_child_weight in grid search,
#' @return      a data frame
# -----------------------------------------------------------------------------
# @export
#' @note        A search grid calculated based on the input data
# @examples    \
# -----------------------------------------------------------------------------
calculate_search_grid_xgbt = function(y, X, n_rounds, n_depth, n_seq){
  if (n_rounds <= 5) {
    stop("n_round should be larger than 5")
  }
  res_grid = expand.grid(
    nrounds          = c(seq(from = 4, to = 20, by = 4),
                         seq(from = 40, by = 40, length.out = n_rounds - 5)),
    max_depth        = seq(2, by = 3, length.out = n_depth),
    gamma            = seq(0, 10,     length.out = n_seq),
    min_child_weight = seq(0, 20,     length.out = n_seq),
    eta              = c(.01, .1, .2, .3),
    colsample_bytree = c(.5, .7, .9),
    subsample        = 0.9
  )
  return(res_grid)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # reg_xgbt
#' DESCRIPTION
#' @rdname      reg_xgbt
#' @title       Prediction with XGBTree and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       xgbt_TC trControl that pass to caret
#' @param       xgbt_TG tuneGrid that pass to caret
#' @param       n_rounds  number of nrounds in grid search,
#' @param       n_depth   number of depth in grid search,
#' @param       n_seq     number of gamma/min_child_weight in grid search,
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using xgboost through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_xgbt = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  xgbt_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  xgbt_TG   = NULL,
  n_rounds  = 8,
  n_depth   = 5,
  n_seq     = 5,
  nc        = 1
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
    if (is.null(xgbt_TG)){
      xgbt_TG = calculate_search_grid_xgbt(
        y = y[-idx_test],
        X = transformed$training,
        n_rounds  = n_rounds,
        n_depth   = n_depth,
        n_seq     = n_seq
      )
    }

    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "xgbTree", tuneGrid  = xgbt_TG, trControl = xgbt_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "xgbTree", tuneGrid  = xgbt_TG, trControl = xgbt_TC
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
      TC = xgbt_TC,
      TG = xgbt_TG,
      search_par = list(
        n_rounds,
        n_depth,
        n_seq
      )
    ),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # cla_xgbt
#' DESCRIPTION
#' @rdname      cla_xgbt
#' @title       Prediction with XGBTree and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       xgbt_TC trControl that pass to caret
#' @param       xgbt_TG tuneGrid that pass to caret
#' @param       n_rounds  number of nrounds in grid search,
#' @param       n_depth   number of depth in grid search,
#' @param       n_seq     number of gamma/min_child_weight in grid search,
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using xgboost through CV.
# @examples    \
# -----------------------------------------------------------------------------
cla_xgbt = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  xgbt_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  xgbt_TG   = NULL,
  n_rounds  = 8,
  n_depth   = 5,
  n_seq     = 5,
  nc        = 1
){
  # Checking data
  seed      = check_input_seed(seed = seed)
  idx_valid = check_input_data(y =y, X = X)
  y = y[idx_valid]; X = X[idx_valid,]
  foldset   = check_input_foldset(foldset = foldset, y = y, K = K)

  # Cross-validation
  if (pb) pbi = txtProgressBar(min = 0, max=K, style = 3)
  hat_y = y
  model_fit_list = vector("list", K)

  for (i in 1:K){
    idx_test = foldset[[i]]
    transformed = pre_prosess_in_CV(X = X, idx_test = idx_test, n_pca = n_pca)
    if (is.null(xgbt_TG)){
      xgbt_TG = calculate_search_grid_xgbt(
        y = y[-idx_test],
        X = transformed$training,
        n_rounds  = n_rounds,
        n_depth   = n_depth,
        n_seq     = n_seq
      )
    }

    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "xgbTree", tuneGrid  = xgbt_TG, trControl = xgbt_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "xgbTree", tuneGrid  = xgbt_TG, trControl = xgbt_TC
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
      TC = xgbt_TC,
      TG = xgbt_TG,
      search_par = list(
        n_rounds,
        n_depth,
        n_seq
      )
    ),
    model = model_fit_list,
    accuracy   = caret::confusionMatrix(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}