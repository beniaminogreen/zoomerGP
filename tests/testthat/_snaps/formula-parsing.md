# formula parsing works

    Code
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width), data = iris)
    Output
      $columns
      Sepal.Width 
                2 
      
      $outcome
      Sepal.Length 
                 1 
      
      $specification
      $specification$kernel
      [1] "rbf"
      
      $specification$cols
      [1] 1
      
      $specification$kwargs
      list()
      
      

---

    Code
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width), data = iris)
    Output
      $columns
      Sepal.Width 
                2 
      
      $outcome
      Sepal.Length 
                 1 
      
      $specification
      $specification$kernel
      [1] "rbf"
      
      $specification$cols
      [1] 1
      
      $specification$kwargs
      list()
      
      

---

    Code
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width) * spectral2(Sepal.Width), data = iris)
    Output
      $columns
      Sepal.Width 
                2 
      
      $outcome
      Sepal.Length 
                 1 
      
      $specification
      $specification$type
      [1] "multiply"
      
      $specification$left
      $specification$left$kernel
      [1] "rbf"
      
      $specification$left$cols
      [1] 1
      
      $specification$left$kwargs
      list()
      
      
      $specification$right
      $specification$right$kernel
      [1] "spectral2"
      
      $specification$right$cols
      [1] 1
      
      $specification$right$kwargs
      list()
      
      
      

---

    Code
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width, Petal.Length) * spectral2(
        Sepal.Width), data = iris)
    Output
      $columns
       Sepal.Width Petal.Length 
                 2            3 
      
      $outcome
      Sepal.Length 
                 1 
      
      $specification
      $specification$type
      [1] "multiply"
      
      $specification$left
      $specification$left$kernel
      [1] "rbf"
      
      $specification$left$cols
      [1] 1 2
      
      $specification$left$kwargs
      list()
      
      
      $specification$right
      $specification$right$kernel
      [1] "spectral2"
      
      $specification$right$cols
      [1] 1
      
      $specification$right$kwargs
      list()
      
      
      

---

    Code
      r_parse_formula(Sepal.Length ~ rbf(Sepal.Width, Petal.Length) * spectral2(
        Sepal.Width) + rbf(Sepal.Width), data = iris)
    Output
      $columns
       Sepal.Width Petal.Length 
                 2            3 
      
      $outcome
      Sepal.Length 
                 1 
      
      $specification
      $specification$type
      [1] "add"
      
      $specification$left
      $specification$left$type
      [1] "multiply"
      
      $specification$left$left
      $specification$left$left$kernel
      [1] "rbf"
      
      $specification$left$left$cols
      [1] 1 2
      
      $specification$left$left$kwargs
      list()
      
      
      $specification$left$right
      $specification$left$right$kernel
      [1] "spectral2"
      
      $specification$left$right$cols
      [1] 1
      
      $specification$left$right$kwargs
      list()
      
      
      
      $specification$right
      $specification$right$kernel
      [1] "rbf"
      
      $specification$right$cols
      [1] 1
      
      $specification$right$kwargs
      list()
      
      
      

---

    Code
      r_parse_formula(Sepal.Length ~ rbf(everything()) * spectral2(Sepal.Width) + rbf(
        Sepal.Width), data = iris)
    Output
      $columns
       Sepal.Width Petal.Length  Petal.Width      Species 
                 2            3            4            5 
      
      $outcome
      Sepal.Length 
                 1 
      
      $specification
      $specification$type
      [1] "add"
      
      $specification$left
      $specification$left$type
      [1] "multiply"
      
      $specification$left$left
      $specification$left$left$kernel
      [1] "rbf"
      
      $specification$left$left$cols
      [1] 1 2 3 4
      
      $specification$left$left$kwargs
      list()
      
      
      $specification$left$right
      $specification$left$right$kernel
      [1] "spectral2"
      
      $specification$left$right$cols
      [1] 1
      
      $specification$left$right$kwargs
      list()
      
      
      
      $specification$right
      $specification$right$kernel
      [1] "rbf"
      
      $specification$right$cols
      [1] 1
      
      $specification$right$kwargs
      list()
      
      
      

