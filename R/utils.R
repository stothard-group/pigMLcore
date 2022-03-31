# -----------------------------------------------------------------------------
# R FILE             --Tianfu Yang
# Type:              Functions/Analysis/Visualization/Script
# Subtype/Project:
# Descriptions:
#
# -----------------------------------------------------------------------------
# Contents:
# -----------------------------------------------------------------------------
# To-do
# -----------------------------------------------------------------------------
# Pre-load
# -----------------------------------------------------------------------------
# rm(list = ls())

# FUNCTION:     # the function name
#' DESCRIPTION
#' @rdname      check_input_data
#' @title       check the data for the prediction
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       verbose output level: True for more logs.
#' @return      a idx of data records available for the prediction (no missing)
# -----------------------------------------------------------------------------
# @export
#' @note        The function checks the missing values and dimension of the
#'              dataset
# @examples
# -----------------------------------------------------------------------------
check_input_data = function(y, X, verbose = FALSE){
  report_header     = function(){
    cat(
      "- ==================================================================\n",
      "- Starting the classification analysis...                           \n",
      "- ==================================================================\n",
      "- Checking the data ==============================================\n"
    )
  }
  report_data_input = function(){
    if (any(is.na(y))){
      cat("*", sum(is.na(y)),
          "NAs were found in y. Corresponding records has been removed", "\n")
    }
    cat(
      "- From the input data, we detected: ",                             "\n",
      "- \t Number of records: ",       "\t\t", n_rec_y,                  "\n",
      "- \t Number of features: ",      "\t\t", n_feature_x,              "\n")
  }

  # - Write to report
  if (verbose) report_header()
  # - Stop if the feature matrix include any NA.
  if (any(is.na(X))){
    stop("NAs were found in the X matrix. Please consider imputation.")
  }
  # - Output valid y records
  is_not_na_y = !is.na(y)
  # - Check the dimensions of the datasets
  n_rec_y = length(y[is_not_na_y])
  n_feature_x  = ncol(X[is_not_na_y,])
  if (n_rec_y != nrow(X[is_not_na_y,])) {
    stop("Dimensions do not match for y and X, Please Check")
  }
  # - Write to report
  if (verbose) report_data_input()
  return(is_not_na_y)
}

# -----------------------------------------------------------------------------
# FUNCTION:     check_input_foldset
#' DESCRIPTION
#' @rdname      check_input_foldset
#' @title       check foldset
#' @param       foldset provide a foldset to control random sampling error.
#' @param       y the responding variable. The function assume it is non-missing
#' @param       K integer, the number of folds in the CV
#' @param       verbose output level: True for more logs.
#' @return      Checked foldset - the same structure as caret::createFolds()
# -----------------------------------------------------------------------------
# @export
#' @note        check the foldset. If there is none, create one
# @examples
# -----------------------------------------------------------------------------
check_input_foldset = function(foldset, y, K, verbose = FALSE){
  report_partition = function(){
    cat("- Partitioning for Cross Validation ===============================\n")
    if (is.null(foldset)){
      cat("- Random folds were created. Number of folds:", K, "\n")
    } else {
      cat("- Provided folds were Used. Number of folds:", K, "\n")
    }
  }

  # Check y and confirm there is no missing
  if (any(is.na(y))){
    stop("No missing value is allow in y when checking foldset of CV")
  }
  # Setup a Fold Index or using a existing one
  if (is.null(foldset)){
    new_foldset = caret::createFolds(y, K)
  } else {
    if (length(foldset) != K){
      stop("Wrong number of folds in variable foldset")
    }
    if (length(unlist(foldset)) != length(y)){
      stop("Wrong number of total records in variable foldset")
    }
    new_foldset = foldset
  }
  if (verbose) report_partition()
  return(new_foldset)
}

# -----------------------------------------------------------------------------
# FUNCTION:     check_input_seed
#' DESCRIPTION
#' @rdname      check_input_seed
#' @title       check the seed of the analysis
#' @param       seed integer or NULL, the input seed
#' @return      NULL
# -----------------------------------------------------------------------------
# @export      \
#' @note        Create a new seed or check an existing seed
# @examples    \
# -----------------------------------------------------------------------------
check_input_seed = function(seed){
  if (is.null(seed)){
    seed = sample(10^6,1)
  }
  set.seed(seed)
  return(seed)
}

# -----------------------------------------------------------------------------
# FUNCTION:     pre_prosess_in_CV
#' DESCRIPTION
#' @rdname      pre_prosess_in_CV
#' @title       preprocess the feature data in CV
#' @param       X the full feature dataset
#' @param       idx_test the idx of test records in the current fold
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @return      A list including the transformed training and testing data
# -----------------------------------------------------------------------------
# @export      \
#' @note       Pre-process the feature matrix in cross-validation. if n_pca is
#'             given, the processing includes PCA and scaling. Otherwise it is
#'             simply scaling.
# @examples    \
# -----------------------------------------------------------------------------
pre_prosess_in_CV = function(X, idx_test, n_pca){
  if (!is.null(n_pca)){
    preProcValues    = caret::preProcess(
      X[-idx_test,], method = c("pca"), pcaComp = n_pca)
  } else {
    preProcValues    = caret::preProcess(
      X[-idx_test,], method = c("center", "scale"))
  }
  return(list(training = predict(preProcValues, X[-idx_test,]),
              testing  = predict(preProcValues, X[idx_test,]))
  )
}
