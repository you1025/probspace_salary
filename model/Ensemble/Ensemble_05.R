library(tidyverse)

# 21.707
df.rf <- readr::read_csv("data/output/RandomForest_20191205T213755.csv") %>%
  dplyr::rename(rf = y)

# 21.028
df.xgb <- readr::read_csv("data/output/XGBoost_20191123T004031.csv") %>%
  dplyr::rename(xgb = y)

# 21.017
df.lgb <- readr::read_csv("data/output/LightGBM_20191214T032112.csv") %>%
  dplyr::rename(lgb = y)

r.rf  <- 0.20
r.xgb <- 0.40
df.ensembled <- df.rf %>%
  dplyr::left_join(df.xgb, by = "id") %>%
  dplyr::left_join(df.lgb, by = "id") %>%
  dplyr::mutate(
    y = r.rf * rf + r.xgb * xgb + (1 - r.rf - r.xgb) * lgb
  )

df.ensembled %>%
  dplyr::mutate(id = dplyr::row_number(y)) %>%
  ggplot(aes(id)) +
    geom_line(aes(y =  rf), alpha = 1/3, colour = "tomato") +
    geom_line(aes(y = xgb), alpha = 1/3, colour = "blue") +
    geom_line(aes(y = lgb), alpha = 1/3, colour = "yellow") +
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
