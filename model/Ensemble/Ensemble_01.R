library(tidyverse)

df.rf <- readr::read_csv(
  file = "data/output/RandomForest_20191116T201312.csv",
  col_types = cols(
    id = col_integer(),
    y = col_double()
  )
) %>%
  dplyr::rename(random_forest = y)

df.xgb <- readr::read_csv(
  file = "data/output/XGBoost_20191119T232225.csv",
  col_types = cols(
    id = col_integer(),
    y = col_double()
  )
) %>%
  dplyr::rename(xgboost = y)

df.data <- dplyr::left_join(
  df.rf,
  df.xgb,
  by = "id"
)

df.data %>%
  dplyr::mutate(
    diff = random_forest - xgboost,
    id = dplyr::row_number(diff),
    score = alpha * random_forest + (1 - alpha) * xgboost
  ) %>%
  ggplot(aes(id, diff)) +
    geom_line(alpha = 1/3)


alpha <- 1/4
df.data %>%
  dplyr::mutate(
    score = alpha * random_forest + (1 - alpha) * xgboost
  ) %>%
  dplyr::mutate(id = dplyr::row_number(score)) %>%
  tidyr::gather(key = "algorithm", value = "pred_salary", -id) %>%
  ggplot(aes(id, pred_salary)) +
    geom_line(aes(colour = algorithm), alpha = 1/3)



# Ensemble
alpha <- 1/4
dplyr::left_join(
  df.rf,
  df.xgb,
  by = "id"
) %>%
  dplyr::mutate(
    y = alpha * random_forest + (1 - alpha) * xgboost
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
