# -----------------------------------------------------------------------------
# R FILE             --Tianfu Yang
# Type:              Functions
# Subtype/Project:   pigMLcore
# Descriptions:
#   The file provides functions that enable SVR prediction
# -----------------------------------------------------------------------------
# Contents:
# -----------------------------------------------------------------------------
# To-do
# -----------------------------------------------------------------------------
# Pre-load
# -----------------------------------------------------------------------------
# rm(list = ls())

# -----------------------------------------------------------------------------
# FUNCTION:     calculate_search_grid_svrl
#' DESCRIPTION
#' @rdname      calculate_search_grid_svrl
#' @title       calculate the whole grid for SVR linear kernel
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       n_C  number of Cost in grid search,
#' @return      a data frame
# -----------------------------------------------------------------------------
# @export
#' @note        A search grid calculated based on the input data
# @examples    \
# -----------------------------------------------------------------------------
calculate_search_grid_svrl = function(y, X, n_C){
  res_grid = data.frame(C = 2^((1:n_C) - 3))
  return(res_grid)
}

# -----------------------------------------------------------------------------
# FUNCTION:     calculate_search_grid_svrp
#' DESCRIPTION
#' @rdname      calculate_search_grid_svrp
#' @title       calculate the whole grid for SVR Polynomial kernel
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       n_C  number of Cost in grid search,
#' @param       n_degree  number of degree in grid search,
#' @param       n_Scale  number of Scale in grid search,
#' @return      a data frame
# -----------------------------------------------------------------------------
# @export
#' @note        A search grid calculated based on the input data
# @examples    \
# -----------------------------------------------------------------------------
calculate_search_grid_svrp = function(y, X, n_C, n_degree, n_Scale){
  res_grid = expand.grid(C = 2^((1:n_C) - 3),
                         degree = 2:min(n_degree, 3),
                         scale = 10 ^((1:n_Scale) - 4)
                         )
  return(res_grid)
}

# -----------------------------------------------------------------------------
# FUNCTION:     calculate_search_grid_svrr
#' DESCRIPTION
#' @rdname      calculate_search_grid_svrr
#' @title       calculate the whole grid for SVR RBF kernel
#' @param       y the responding variable. The function assume it is non-missing
#' @param       X the full feature dataset
#' @param       n_C  number of Cost in grid search
#' @return      a data frame
# -----------------------------------------------------------------------------
# @export
#' @note        A search grid calculated based on the input data
# @examples    \
# -----------------------------------------------------------------------------
calculate_search_grid_svrr = function(y, X, n_C){
  ## Sigma may have more than one option?
  sigmas = kernlab::sigest(as.matrix(X), scaled = TRUE)
  res_grid = expand.grid(C = 2^((1:n_C) - 3),
                         sigma = mean(as.vector(sigmas[-2])))
  return(res_grid)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # reg_svrl
#' DESCRIPTION
#' @rdname      reg_svrl
#' @title       Prediction with SVR linear and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       svrl_TC trControl that pass to caret
#' @param       svrl_TG tuneGrid that pass to caret
#' @param       n_C  number of Cost in grid search,
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using SVR linear through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_svrl = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  svrl_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  svrl_TG   = NULL,
  n_C       = 15,
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
    if (is.null(svrl_TG)){
      svrl_TG = calculate_search_grid_svrl(
        y = y[-idx_test],
        X = transformed$training,
        n_C = n_C
      )
    }
    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmLinear", tuneGrid  = svrl_TG, trControl = svrl_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmLinear", tuneGrid  = svrl_TG, trControl = svrl_TC
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
      TC = svrl_TC,
      TG = svrl_TG,
      search_par = list(
        n_C
      )
    ),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}


# -----------------------------------------------------------------------------
# FUNCTION:     # reg_svrp
#' DESCRIPTION
#' @rdname      reg_svrp
#' @title       Prediction with SVR polynomialand evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       svrp_TC trControl that pass to caret
#' @param       svrp_TG tuneGrid that pass to caret
#' @param       n_C  number of Cost in grid search
#' @param       n_degree  number of degree in grid search,
#' @param       n_Scale  number of Scale in grid search,
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using SVR polynomial through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_svrp = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  svrp_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  svrp_TG   = NULL,
  n_C       = 10,
  n_degree  = 3,
  n_Scale   = 3,
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
    if (is.null(svrp_TG)){
      svrp_TG = calculate_search_grid_svrp(
        y = y[-idx_test],
        X = transformed$training,
        n_C = n_C,
        n_degree = n_degree,
        n_Scale  = n_Scale
      )
    }
    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmPoly", tuneGrid  = svrp_TG, trControl = svrp_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmPoly", tuneGrid  = svrp_TG, trControl = svrp_TC
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
      TC = svrp_TC,
      TG = svrp_TG,
      search_par = list(
        n_C     ,
        n_degree,
        n_Scale
      )
    ),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # reg_svrr
#' DESCRIPTION
#' @rdname      reg_svrr
#' @title       Prediction with SVR polynomialand evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       svrr_TC trControl that pass to caret
#' @param       svrr_TG tuneGrid that pass to caret
#' @param       n_C  number of Cost in grid search
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using SVR RBF through CV.
# @examples    \
# -----------------------------------------------------------------------------
reg_svrr = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  svrr_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  svrr_TG   = NULL,
  n_C       = 10,
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
    if (is.null(svrr_TG)){
      svrr_TG = calculate_search_grid_svrr(
        y = y[-idx_test],
        X = transformed$training,
        n_C = n_C
      )
    }
    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmRadial", tuneGrid  = svrr_TG, trControl = svrr_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmRadial", tuneGrid  = svrr_TG, trControl = svrr_TC
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
      TC = svrr_TC,
      TG = svrr_TG,
      search_par = list(
        n_C
      )
    ),
    model = model_fit_list,
    accuracy   = cor.test(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}



# -----------------------------------------------------------------------------
# FUNCTION:     # cla_svrl
#' DESCRIPTION
#' @rdname      cla_svrl
#' @title       Prediction with SVR linear and evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       svrl_TC trControl that pass to caret
#' @param       svrl_TG tuneGrid that pass to caret
#' @param       n_C  number of Cost in grid search,
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using SVR linear through CV.
# @examples    \
# -----------------------------------------------------------------------------
cla_svrl = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  svrl_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  svrl_TG   = NULL,
  n_C       = 15,
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
    if (is.null(svrl_TG)){
      svrl_TG = calculate_search_grid_svrl(
        y = y[-idx_test],
        X = transformed$training,
        n_C = n_C
      )
    }
    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmLinear", tuneGrid  = svrl_TG, trControl = svrl_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmLinear", tuneGrid  = svrl_TG, trControl = svrl_TC
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
      TC = svrl_TC,
      TG = svrl_TG,
      search_par = list(
        n_C
      )
    ),
    model = model_fit_list,
    accuracy   = caret::confusionMatrix(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # cla_svrp
#' DESCRIPTION
#' @rdname      cla_svrp
#' @title       Prediction with SVR polynomialand evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       svrp_TC trControl that pass to caret
#' @param       svrp_TG tuneGrid that pass to caret
#' @param       n_C  number of Cost in grid search
#' @param       n_degree  number of degree in grid search,
#' @param       n_Scale  number of Scale in grid search,
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using SVR polynomial through CV.
# @examples    \
# -----------------------------------------------------------------------------
cla_svrp = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  svrp_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  svrp_TG   = NULL,
  n_C       = 10,
  n_degree  = 3,
  n_Scale   = 3,
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
    if (is.null(svrp_TG)){
      svrp_TG = calculate_search_grid_svrp(
        y = y[-idx_test],
        X = transformed$training,
        n_C = n_C,
        n_degree = n_degree,
        n_Scale  = n_Scale
      )
    }
    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmPoly", tuneGrid  = svrp_TG, trControl = svrp_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmPoly", tuneGrid  = svrp_TG, trControl = svrp_TC
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
      TC = svrp_TC,
      TG = svrp_TG,
      search_par = list(
        n_C     ,
        n_degree,
        n_Scale
      )
    ),
    model = model_fit_list,
    accuracy   = caret::confusionMatrix(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}

# -----------------------------------------------------------------------------
# FUNCTION:     # cla_svrr
#' DESCRIPTION
#' @rdname      cla_svrr
#' @title       Prediction with SVR polynomialand evaluation through CV
#' @param       y the responding variable
#' @param       X the feature matrix. No missing allowed
#' @param       job_name A character of job ID
#' @param       K integer, the number of folds in the CV
#' @param       n_pca number of selected top PCs. Null for no PCA
#' @param       pb show pgbar or nots
#' @param       foldset provide a foldset to control random sampling error.
#' @param       seed provide a seed to reproduce a result. NULL by default
#' @param       svrr_TC trControl that pass to caret
#' @param       svrr_TG tuneGrid that pass to caret
#' @param       n_C  number of Cost in grid search
#' @param       nc number of cpus to use in parallel
#' @return      a list
# -----------------------------------------------------------------------------
#' @export
#' @note        The function provide a general way to evaluate the prediction
#'              using SVR RBF through CV.
# @examples    \
# -----------------------------------------------------------------------------
cla_svrr = function(
  y,
  X,
  job_name  = "new_job",
  K         = 10,
  n_pca     = NULL,
  pb        = FALSE,
  foldset   = NULL,
  seed      = NULL,
  svrr_TC   = trainControl(method = "cv", number = 5, verboseIter = F),
  svrr_TG   = NULL,
  n_C       = 10,
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
    if (is.null(svrr_TG)){
      svrr_TG = calculate_search_grid_svrr(
        y = y[-idx_test],
        X = transformed$training,
        n_C = n_C
      )
    }
    if (nc > 1){
      cl = parallel::makePSOCKcluster(nc)
      doParallel::registerDoParallel(cl)
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmRadial", tuneGrid  = svrr_TG, trControl = svrr_TC
      )
      parallel::stopCluster(cl)
    } else {
      model_fit_list[[i]] = train(
        x = transformed$training, y = y[-idx_test],
        method = "svmRadial", tuneGrid  = svrr_TG, trControl = svrr_TC
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
      TC = svrr_TC,
      TG = svrr_TG,
      search_par = list(
        n_C
      )
    ),
    model = model_fit_list,
    accuracy   = caret::confusionMatrix(y, hat_y),
    prediction = data.frame(true_y = y, hat_y = hat_y)
  )
  return(res)
}