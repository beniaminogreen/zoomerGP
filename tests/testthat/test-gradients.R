test_that("multiplication works", {
  skip_if_not_installed("nloptr")
  expect_true(
              test_grad(Sepal.Length ~ rbf(Sepal.Width), data = iris)
              )
  expect_true(
              test_grad(Sepal.Length ~ rbf(Sepal.Width)+rbf(Petal.Length), data = iris)
              )
})
