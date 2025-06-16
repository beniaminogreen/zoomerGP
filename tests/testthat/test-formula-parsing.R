test_that("formula parsing works", {
  data <- iris

  expect_snapshot(
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width), data = iris)
  )
  expect_snapshot(
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width)*spectral2(Sepal.Width), data = iris)
  )
  expect_snapshot(
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width, Petal.Length)*spectral2(Sepal.Width), data = iris)
  )
  expect_snapshot(
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width, Petal.Length)*spectral2(Sepal.Width) + rbf(Sepal.Width), data = iris)
  )
  expect_snapshot(
      r_parse_formula(Sepal.Length ~ rbf(everything())*spectral2(Sepal.Width) + rbf(Sepal.Width), data = iris)
  )
})
