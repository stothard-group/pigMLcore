
library(mlbench)

test_that("calculate_search_grid_rf",{
  ## https://stats.stackexchange.com/questions/166630
  n <- 500; p <- 20; b <- c(-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,3);
  alpha <- .5
  X <- cbind(rep(1,n),scale(matrix(rnorm(p*n),nrow=n)))
  Y <- rbinom(n,1,prob = exp(X%*%b)/(1 + exp(X%*%b)))
  expect_error(
    calculate_search_grid_rf(Y, X, n_mtry = 5, n_MNS = 5), NA,
    info = "calculate grid")
  expect_equal(
    nrow(calculate_search_grid_rf(Y, X, n_mtry = 5, n_MNS = 5)), 75,
    info = "size of the calculated grid"
  )
})

test_that("reg_rf", {
  expect_error(
    reg_rf(y = iris$Sepal.Length, X = as.matrix(iris[,2:4]),
           K = 5, n_mtry = 5, n_MNS = 5),
    regexp = NA,
    info = "Evaluate the prediction"
  )
})

test_that("cla_rf",
  {
    expect_error(label = "Accessing Ionosphere dataset",
                 object = {
                   data(Ionosphere)
                 },
                 regexp = NA
    )

    expect_error(label = "Running the prediction",
                 object = {
                   res = cla_rf(y = Ionosphere$Class,
                                X = as.matrix(Ionosphere[,3:34]),
                                K = 5, n_mtry = 5, n_MNS = 5)
                 },
                 regexp = NA
    )

    expect_gt(label = "Evaluate the prediction",
              object = res$accuracy$overall["Accuracy"], expected = 0.6)
  })