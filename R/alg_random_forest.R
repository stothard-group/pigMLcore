# -----------------------------------------------------------------------------
# R FILE             --Tianfu Yang
# Type:              Functions
# Subtype/Project:   pigMLcore
# Descriptions:
#   The file provides functions that enable Random Forest prediction
# -----------------------------------------------------------------------------
# Contents:
# -----------------------------------------------------------------------------
# To-do
# -----------------------------------------------------------------------------
# Pre-load
# -----------------------------------------------------------------------------
# rm(list = ls())

# -----------------------------------------------------------------------------
# FUNCTION:     calculate_search_grid_rf
#' DESCRIPTION
#' @rdname      calculate_search_grid_rf
#' @title       calculate the whole grid for rf
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       n_mtry number of mtry tested in the grid
#' @param       n_MNS number of min.node.size tested in the grid
#' @return      a data frame
# -----------------------------------------------------------------------------
# @export
#' @note        A search grid calculated based on the input data
# @examples    \
# -----------------------------------------------------------------------------
calculate_search_grid_rf = function(y, X, n_mtry, n_MNS){
  d = ncol(X)
  mtry_list = var_seq(p = d, len = n_mtry)
  if (is.factor(y)){
    split_rule_list = c("gini", "extratrees", "hellinger")
  } else {
    split_rule_list = c("variance", "extratrees", "maxstat")
  }
  MNS_list = seq(from = 5, to = 25, length.out = n_MNS)
  res_grid = expand.grid(mtry = mtry_list,
                         min.node.size = MNS_list,
                         splitrule = split_rule_list)
  return(res_grid)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # reg_rf
#' DESCRIPTION
#' @rdname      reg_rf
#' @title       Prediction with RF and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       verbose output log or not
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       rf_TC trControl that pass to caret
#' @param       rf_TG tuneGrid that pass to caret
#' @param       n_mtry number of ntry tested in the grid
#' @param       n_MNS number of min.node.size tested in the grid
#' @param       nc number of cpus to use in parallel- should be 1
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using RF through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_rf = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  verbose   = FALSE,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  rf_TC  = trainControl(method = "cv", number = 5, verboseIter = F),
  rf_TG  = NULL,
  n_mtry    = 10,
  n_MNS     = 5,
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
    if (is.null(rf_TG)){
      rf_TG = calculate_search_grid_rf(
        y = y[-idx_test], X = transformed$training,
        n_mtry = n_mtry, n_MNS = n_MNS)
    }
    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "ranger", tuneGrid  = rf_TG, trControl = rf_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "ranger", tuneGrid  = rf_TG, trControl = rf_TC
      )
    }
    if (pb) {setTxtProgressBar(pbi, i)}
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
      TC = rf_TC,
      TG = rf_TG,
      search_par = list(
      	n_mtry,
      	n_MNS
      )
    ),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # cla_rf
#' DESCRIPTION
#' @rdname      cla_rf
#' @title       Prediction with RF and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       verbose output log or not
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       rf_TC trControl that pass to caret
#' @param       rf_TG tuneGrid that pass to caret
#' @param       n_mtry number of ntry tested in the grid
#' @param       n_MNS number of min.node.size tested in the grid
#' @param       nc number of cpus to use in parallel- should be 1
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using RF through CV.
# @examples    \
# -----------------------------------------------------------------------------
cla_rf = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  verbose   = FALSE,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  rf_TC     = trainControl(method = "cv", number = 5, verboseIter = F),
  rf_TG     = NULL,
  n_mtry    = 10,
  n_MNS     = 5,
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
    if (is.null(rf_TG)){
      rf_TG = calculate_search_grid_rf(
        y = y[-idx_test], X = transformed$training,
        n_mtry = n_mtry, n_MNS = n_MNS)
    }
    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "ranger", tuneGrid  = rf_TG, trControl = rf_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "ranger", tuneGrid  = rf_TG, trControl = rf_TC
      )
    }
    if (pb) {setTxtProgressBar(pbi, i)}
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
      TC = rf_TC,
      TG = rf_TG,
      search_par = list(
        n_mtry,
        n_MNS
      )
    ),
    model = model_fit_list,
    accuracy   = caret::confusionMatrix(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}