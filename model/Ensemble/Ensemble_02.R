library(tidyverse)

df.rf.old <- readr::read_csv(
  file = "data/output/RandomForest_20191116T201312.csv",
  col_types = cols(
    id = col_integer(),
    y = col_double()
  )
) %>%
  dplyr::rename(random_forest.old = y)

df.xgb.old <- readr::read_csv(
  file = "data/output/XGBoost_20191119T232225.csv",
  col_types = cols(
    id = col_integer(),
    y = col_double()
  )
) %>%
  dplyr::rename(xgboost.old = y)

df.xgb.current <- readr::read_csv(
  file = "data/output/XGBoost_20191120T143428.csv",
  col_types = cols(
    id = col_integer(),
    y = col_double()
  )
) %>%
  dplyr::rename(xgboost.current = y)

df.data <- dplyr::left_join(
  df.rf.old,
  df.xgb.old,
  by = "id"
) %>%
  dplyr::left_join(df.xgb.current, by = "id")

# Ensemble
rf <- 0
xg <- 0.5
df.data %>%
  dplyr::mutate(
    y = rf * random_forest.old + xg * xgboost.old + (1 - rf - xg) * xgboost.current
  ) %>%
  dplyr::select(id, y) %>%

  {
    df.result <- (.)
    
    # ファイル名
    filename <- stringr::str_c(
      "Ensemble",
      lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
      sep = "_"
    ) %>%
      stringr::str_c("csv", sep = ".")
    
    # 出力ファイルパス
    filepath <- stringr::str_c("data/output", filename, sep = "/")
    
    # 書き出し
    readr::write_csv(df.result, filepath, col_names = T)
  }
