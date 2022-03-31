test_that("check_input_data()", {
  y1 = 1:5
  y2 = c(1:2,NA,4,5)
  X1 = matrix(1:30, ncol = 6)
  X2 = matrix(1:32, ncol = 8)
  X3 = matrix(c(1:28,NA,NA), ncol = 6)
  # Tests
  expect_identical(
    check_input_data(y1, X1), c(TRUE,TRUE,TRUE,TRUE,TRUE),
    info = "Output for non-missing y")
  expect_identical(
    check_input_data(y2, X1), c(TRUE,TRUE,FALSE,TRUE,TRUE),
    info = "Output for y with missing value")
  expect_error(
    check_input_data(y1, X2),
    info = "Throw error when dimentions don't match")
  expect_error(
    check_input_data(y2, X2),
    info = "Throw error when dimentions don't match")
  expect_error(
    check_input_data(y1, X3),
    info = "Throw error when X includes missing value")
  expect_error(
    check_input_data(y2, X3),
    info = "Throw error when X includes missing value")
})

test_that("check_input_foldset()", {
  y1 = 1:10
  y2 = c(1:10,NA,NA)
  y3 = 1:8
  K1 = 5
  K2 = 2
  foldset1 = caret::createFolds(y1, K1)
  foldset2 = caret::createFolds(y1, K2)
  # Tests
  expect_error(
    check_input_foldset(y = y2, foldset = NULL, K = K1),
    info = "No missing data is allowed in y"
  )
  expect_error(
    check_input_foldset(y = y2, foldset = foldset1, K = K1),
    info = "No missing data is allowed in y"
  )
  expect_error(
    check_input_foldset(y = y1, foldset = foldset2, K = K1),
    info = "The length of foldset is different from K"
  )
  expect_error(
    check_input_foldset(y = y3, foldset = foldset1, K = K1),
    info = "The total number of y is different from the foldset"
  )
  expect_error(
    check_input_foldset(y = y1, foldset = NULL, K = K1), regexp = NA,
    info = "Create a new foldset when there is no input"
  )
  expect_error(
    check_input_foldset(
      foldset = check_input_foldset(y = y1, foldset = NULL, K = K1),
      y = y1, K = K1),
    regexp = NA,
    info = "Create a correct foldset when there is no input"
  )
})

test_that("check_input_seed", {
  set.seed(42); a = runif(1000)
  expect_identical(
    {check_input_seed(42); runif(1000)}, a,
    info = "check_seed successfully set the seed")
  expect_type(check_input_seed(NULL), "integer")
})

test_that("pre_prosess_in_CV", {
  expect_error(
    pre_prosess_in_CV(X = iris[,1:4], idx_test = 1:10, n_pca = 2),
    regexp = NA,
    info = "preprocessing the data"
  )
})