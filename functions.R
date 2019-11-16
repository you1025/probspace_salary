library(tidyverse)


# Data Load ---------------------------------------------------------------

# Train Data
load_train_data <- function(path) {
  readr::read_csv(
    path,
    col_types = cols(
      id = col_integer(),
      position = readr::col_factor(levels = 0:4),
      age = col_integer(),
      area = readr::col_factor(),
      sex = readr::col_factor(levels = 1:2),
      partner = readr::col_factor(levels = 0:1),
      num_child = col_integer(),
      education = readr::col_factor(levels = 0:4),
      service_length = col_integer(),
      study_time = col_double(),
      commute = col_double(),
      overtime = col_double(),
      salary = col_double()
    )
  )
}
#df.train <- load_train_data("data/input/train_data.csv")

# Test Data
load_test_data <- function(path) {
  readr::read_csv(
    path,
    col_types = cols(
      id = col_integer(),
      position = readr::col_factor(levels = 0:4),
      age = col_integer(),
      area = readr::col_factor(),
      sex = readr::col_factor(levels = 1:2),
      partner = readr::col_factor(levels = 0:1),
      num_child = col_integer(),
      education = readr::col_factor(levels = 0:4),
      service_length = col_integer(),
      study_time = col_double(),
      commute = col_double(),
      overtime = col_double()
    )
  )
}
#load_test_data("data/input/test_data.csv")


clip_outlier <- function(x, threshold = 0.99) {
  x_threshold <- quantile(x, probs = threshold)
  ifelse(x < x_threshold, x, x_threshold)
}
#clip_outlier(1:100)

categorical_value <- function(data, feature, target = salary) {

  # for NSE
  feature = dplyr::enquo(feature)
  target  = dplyr::enquo(target)


  data %>%

    dplyr::group_by(!!feature) %>%
    dplyr::summarise(
      n = n(),
      avg = mean(!!target, na.rm = T),
      med = median(!!target, na.rm = T)
    ) %>%

    dplyr::mutate(n_ratio = n / sum(n)) %>%

    dplyr::select(
      !!feature,
      n,
      n_ratio,
      avg,
      med
    )
}
#categorical_value(df.train_data, position, salary)

add_feature_per_category <- function(target_data, train_data, category_feature, target_feature) {

  # for NSE
  category_feature = dplyr::enquo(category_feature)
  target_feature   = dplyr::enquo(target_feature)

  # 新規に生成される項目名
  # ex. "position_avg_salary"
  avg_col_name = stringr::str_c(
    dplyr::quo_name(category_feature),
    "avg",
    dplyr::quo_name(target_feature),
    sep = "_"
  )

  # 代表値の一覧を取得
  # 今回は平均値(avg)のみ使用
  df.category_average <- categorical_value(train_data, !!category_feature, !!target_feature) %>%
    dplyr::select(
      !!category_feature,
      !!avg_col_name := avg
    )

  # target_data に算出した代表値を結合
  target_data %>%
    dplyr::left_join(
      df.category_average,
      by = dplyr::quo_name(category_feature)
    )
}
#add_feature_per_category(df.train_data, df.train_data, position, salary)


