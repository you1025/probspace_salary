library(tidyverse)

df.rf <- readr::read_csv("data/output/RandomForest_20191205T213755.csv") %>%
  dplyr::rename(rf = y)
df.xgb.orig <- readr::read_csv("data/output/XGBoost_20191123T004031.csv") %>%
  dplyr::rename(xgb.orig = y)
df.xgb.current <- readr::read_csv("data/output/XGBoost_add_linear_prediction_20191207T004405.csv") %>%
  dplyr::rename(xgb.current = y)


r.rf          <- 0.15
r.xgb.current <- 0.25
r.xgb.orig    <- (1 - r.rf - r.xgb.current)
df.ensembled <- df.xgb.orig %>%
  dplyr::left_join(df.rf, by = "id") %>%
  dplyr::left_join(df.xgb.current, by = "id") %>%
  dplyr::mutate(
    y = r.rf * rf + r.xgb.current * xgb.current + r.xgb.orig * xgb.orig
  )

df.ensembled %>%
  dplyr::mutate(id = dplyr::row_number(y)) %>%
  ggplot(aes(id)) +
    geom_line(aes(y = y)) +
    geom_line(aes(y = xgb.orig),    alpha = 1/3, colour = "green") +
    geom_line(aes(y = rf),          alpha = 1/3, colour = "tomato") +
    geom_line(aes(y = xgb.current), alpha = 1/3, colour = "blue")


df.ensembled %>%
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
