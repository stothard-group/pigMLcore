library(mlbench)


test_that("calculate_search_grid_xgbl",{
  ## https://stats.stackexchange.com/questions/166630
  n <- 500; p <- 20; b <- c(-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,3);
  alpha <- .5
  X <- cbind(rep(1,n),scale(matrix(rnorm(p*n),nrow=n)))
  Y <- rbinom(n,1,prob = exp(X%*%b)/(1 + exp(X%*%b)))
  expect_error(
    calculate_search_grid_xgbl(Y, X, n_rounds = 10, n_reg = 10), NA,
    info = "calculate grid")
  expect_equal(
    nrow(calculate_search_grid_xgbl(Y, X, n_rounds = 6, n_reg = 5)), 1080,
    # 6 * (5+1) ^2 * 5 = 1080
    info = "size of the calculated grid"
  )
})

test_that("reg_xgbl", {
  expect_error(
    reg_xgbl(y = iris$Sepal.Length, X = as.matrix(iris[,2:4]),
                  K = 3, n_rounds = 6, n_reg = 3),
    regexp = NA,
    info = "Evaluate the prediction"
  )
})

test_that("cla_xgbl",
  {
    expect_error(label = "Accessing Ionosphere dataset",
                 object = {
                   data(Ionosphere)
                 },
                 regexp = NA
    )
    expect_error(label = "Running the prediction",
                 object = {
                   res1 = cla_xgbl(y = Ionosphere$Class,
                                   X = as.matrix(Ionosphere[,3:34]),
                                   K = 3, n_rounds = 6, n_reg = 3)
                 },
                 regexp = NA
    )
    expect_gt(label = "Evaluate the prediction",
              object = res1$accuracy$overall["Accuracy"], expected = 0.6)
  })