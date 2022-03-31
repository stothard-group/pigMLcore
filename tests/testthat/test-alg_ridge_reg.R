test_that("reg_rr", {
  expect_error(
    reg_rr(y = iris$Sepal.Length, X = as.matrix(iris[,2:4]),K = 5),
    regexp = NA,
    info = "Evaluate the prediction"
  )
})