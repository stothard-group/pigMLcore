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
# FUNCTION:     # regression_rr
#' DESCRIPTION
#' @rdname      reg_rr
#' @title       Prediction with EN and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       verbose output log or not
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       rr_TC trControl that pass to caret
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using EN through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_rr = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  verbose   = FALSE,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  rr_TC  = trainControl(method = "cv", number = 5, verboseIter = F),
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

    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]]  = train(
        x = transformed$training, y = y[-idx_test],
        method = "ridge", trControl = rr_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]]  = train(
        x = transformed$training, y = y[-idx_test],
        method = "ridge", trControl = rr_TC
      )
    }

    if (pb) {setTxtProgressBar(pbi, i)}
    hat_y[idx_test] = predict(model_fit_list[[i]] , newdata = transformed$testing)
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
      TC = rr_TC,
      TG = NULL,
      search_par = NULL
    ),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  if (verbose){
    cat("- ============================================ Analysis Completed!\n")
  }
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
