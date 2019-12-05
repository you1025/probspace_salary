library(tidyverse)

df.xgb <- readr::read_csv(
  "data/output/XGBoost_20191123T004031.csv",
  col_types = cols(
    id = col_integer(),
    y = col_double()
  )
) %>%
  dplyr::rename(y_xgb = y)

df.rf  <- readr::read_csv(
  "data/output/RandomForest_20191205T213755.csv",
  col_types = cols(
    id = col_integer(),
    y = col_double()
  )
) %>%
  dplyr::rename(y_rf = y)

alpha <- 0.75
df.ensembled <- df.xgb %>%
  dplyr::left_join(df.rf, by = "id") %>%
  dplyr::mutate(y = alpha * y_xgb + (1 - alpha) * y_rf)

df.ensembled %>%
  dplyr::mutate(id = dplyr::row_number(y)) %>%
  ggplot(aes(id)) +
    geom_line(aes(y = y_rf),  colour = "tomato", alpha = 1/3) +
    geom_line(aes(y = y_xgb), colour = "blue",   alpha = 1/3) +
    geom_line(aes(y = y))


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
