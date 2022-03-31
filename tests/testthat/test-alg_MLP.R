library(mlbench)

test_that("calculate_search_grid_mlpL2",
{
  expect_error(label = "Generating Simulation data 1",
    object = {
      ## https://stats.stackexchange.com/questions/166630
      n <- 500; p <- 20; b <- c(-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,3);
      alpha <- .5
      X <- cbind(rep(1,n),scale(matrix(rnorm(p*n),nrow=n)))
      Y <- rbinom(n,1,prob = exp(X%*%b)/(1 + exp(X%*%b)))
    },
    regexp = NA
  )

  expect_error(label = "calculate grid",
    object = {
      res_grid = calculate_search_grid_mlpL2(Y, X, n_size = 5, n_lambda = 3)
    },
    regexp = NA,
  )

  expect_equal(label = "size of the calculated grid",
    nrow(res_grid), 75
  )
})

test_that("reg_mlpL2",
{
  expect_error(label = "Accessing boston housing dataset",
    object = {
      data(BostonHousing)
    },
    regexp = NA
  )
  expect_error(label = "Running the prediction",
    object = {
      res = reg_mlpL2(y = BostonHousing$medv,
                      X = as.matrix(BostonHousing[,c(1:3,5:13)]),
                      K = 2, n_size = 2, n_lambda = 2, epoch = 100)
    },
    regexp = NA
  )
  expect_gt(label = "Evaluate the prediction",
            object = res$accuracy$estimate, expected = 0.4)
})

