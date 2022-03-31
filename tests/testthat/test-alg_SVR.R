library(mlbench)

test_that("calculate_search_grid_SVRL",{
  ## https://stats.stackexchange.com/questions/166630
  n <- 500; p <- 20; b <- c(-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,3);
  alpha <- .5
  X <- cbind(rep(1,n),scale(matrix(rnorm(p*n),nrow=n)))
  Y <- rbinom(n,1,prob = exp(X%*%b)/(1 + exp(X%*%b)))
  expect_error(
    calculate_search_grid_svrl(Y, X, n_C = 10), NA,
    info = "calculate grid")
  expect_equal(
    nrow(calculate_search_grid_svrl(Y, X, n_C = 10)), 10,
    info = "size of the calculated grid"
  )
})

test_that("reg_SVRL", {
  expect_error(
    reg_svrl(y = iris$Sepal.Length,
             X = as.matrix(iris[,2:4]),
             K = 5, n_C = 10),
    regexp = NA,
    info = "Evaluate the prediction"
  )
})

test_that("calculate_search_grid_SVRP",{
  ## https://stats.stackexchange.com/questions/166630
  n <- 500; p <- 20; b <- c(-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,3);
  alpha <- .5
  X <- cbind(rep(1,n),scale(matrix(rnorm(p*n),nrow=n)))
  Y <- rbinom(n,1,prob = exp(X%*%b)/(1 + exp(X%*%b)))
  expect_error(
    calculate_search_grid_svrp(Y, X, n_C = 10, n_degree = 2, n_Scale = 3), NA,
    info = "calculate grid")
  expect_equal(
    nrow(calculate_search_grid_svrp(Y, X, n_C = 10, n_degree = 3, n_Scale = 3)), 60,
    info = "size of the calculated grid"
  )
})


test_that("reg_SVRP", {
  expect_error(
    reg_svrp(y = iris$Sepal.Length,
             X = as.matrix(iris[,2:4]),
             K = 5, n_C = 10, n_degree = 3, n_Scale = 3),
    regexp = NA,
    info = "Evaluate the prediction"
  )
})

test_that("calculate_search_grid_SVRR",{
  ## https://stats.stackexchange.com/questions/166630
  n <- 500; p <- 20; b <- c(-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,3);
  alpha <- .5
  X <- cbind(rep(1,n),scale(matrix(rnorm(p*n),nrow=n)))
  Y <- rbinom(n,1,prob = exp(X%*%b)/(1 + exp(X%*%b)))
  expect_error(
    calculate_search_grid_svrr(Y, X, n_C = 10), NA,
    info = "calculate grid")
  expect_equal(
    nrow(calculate_search_grid_svrr(Y, X, n_C = 10)), 10,
    info = "size of the calculated grid"
  )
})

test_that("reg_SVRR", {
  expect_error(
    reg_svrr(y = iris$Sepal.Length,
             X = as.matrix(iris[,2:4]),
             K = 5, n_C = 10),
    regexp = NA,
    info = "Evaluate the prediction"
  )
})

test_that("cla_SVR",
          {
            expect_error(label = "Accessing Ionosphere dataset",
                         object = {
                           data(Ionosphere)
                         },
                         regexp = NA
            )

            expect_error(label = "Running the prediction - SVRL",
                         object = {
                           res1 = cla_svrl(y = Ionosphere$Class,
                                        X = as.matrix(Ionosphere[,3:34]),
                                        K = 5, n_C = 10)
                         },
                         regexp = NA
            )

            expect_gt(label = "Evaluate the prediction - SVRL",
                      object = res1$accuracy$overall["Accuracy"], expected = 0.6)

            expect_error(label = "Running the prediction - SVRP",
                         object = {
                           res2 = cla_svrp(y = Ionosphere$Class,
                                           X = as.matrix(Ionosphere[,3:34]),
                                           K = 5, n_C = 10, n_degree = 3, n_Scale = 3)
                         },
                         regexp = NA
            )

            expect_gt(label = "Evaluate the prediction - SVRP",
                      object = res2$accuracy$overall["Accuracy"], expected = 0.6)

            expect_error(label = "Running the prediction - SVRR",
                         object = {
                           res3 = cla_svrr(y = Ionosphere$Class,
                                           X = as.matrix(Ionosphere[,3:34]),
                                           K = 5, n_C = 10)
                         },
                         regexp = NA
            )

            expect_gt(label = "Evaluate the prediction - SVRR",
                      object = res3$accuracy$overall["Accuracy"], expected = 0.6)
          })
