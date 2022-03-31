
library(mlbench)

test_that("calculate_lambda_max",
{
  expect_error(label = "Generating Simulation data 1",
    object = {
      ## https://stats.stackexchange.com/questions/166630
      n <- 500; p <- 20; b <- c(-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,-5,3,2,0,3);
      alpha <- .5
      X <- cbind(rep(1,n),scale(matrix(rnorm(p*n),nrow=n)))
      Y <- rbinom(n,1,prob = exp(X%*%b)/(1 + exp(X%*%b)))
      glmnet_lambda_max =
        glmnet::glmnet(x=X,y=Y,family="binomial",alpha=0.5,standardize=F)$lambda[1]
    },
    regexp = NA
  )

  expect_equal(label = "calculate lambda.max",
    object = calculate_lambda_max(Y, X, 0.5), expected = glmnet_lambda_max
  )

  expect_error(label = "valid upper limit of lambda.max",
    object = {
      calculate_lambda_max(Y, X, 1.1)
    }
  )

  expect_error(label = "valid lower limit of lambda.max",
    object = {
      calculate_lambda_max(Y, X, -2)
    }
  )
})

test_that("calculate_search_grid_en",
{
  expect_error(label = "Generating Simulation data 2",
    object = {
      ## https://stats.stackexchange.com/questions/166630
      n <- 500; p <- 3; b <- c(-5,3,2,0); alpha <- .5
      X <- cbind(rep(1,n),scale(matrix(rnorm(p*n),nrow=n)))
      Y <- rbinom(n,1,prob = exp(X%*%b)/(1 + exp(X%*%b)))
      glmnet_lambda_max =
        glmnet::glmnet(x=X,y=Y,family="binomial",alpha=0.5,standardize=F)$lambda[1]
    },
    regexp = NA
  )

  expect_error(label = "calculate grid",
    object = {
      res_grid = calculate_search_grid_en(Y, X, n_alpha = 5, n_lambda = 100)
    },
    regexp = NA,
  )

  expect_equal(label = "size of the calculated grid",
    nrow(res_grid), 500
  )
})

test_that("reg_en",
{
  expect_error(label = "Accessing boston housing dataset",
    object = {
      data(BostonHousing)
    },
    regexp = NA
  )

  expect_error(label = "Running the prediction",
    object = {
      res = reg_en(y = BostonHousing$medv,
                      X = as.matrix(BostonHousing[,c(1:3,5:13)]),
                      K = 3, n_alpha = 10, n_lambda = 50)
    },
    regexp = NA
  )

  expect_gt(label = "Evaluate the prediction",
            object = res$accuracy$estimate, expected = 0.8)
})

test_that("cla_en",
  {
    expect_error(label = "Accessing Ionosphere dataset",
                 object = {
                   data(Ionosphere)
                 },
                 regexp = NA
    )

    expect_error(label = "Running the prediction",
                 object = {
                   res = cla_en(y = Ionosphere$Class,
                                X = as.matrix(Ionosphere[,3:34]),
                                K = 3, n_alpha = 10, n_lambda = 50)
                 },
                 regexp = NA
    )

    expect_gt(label = "Evaluate the prediction",
              object = res$accuracy$overall["Accuracy"], expected = 0.6)
  })