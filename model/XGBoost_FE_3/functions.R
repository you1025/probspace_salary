source("functions.R", encoding = "utf-8")


# area_partner_child_segment ごとに集計値を算出する
smoothed_categorical_value <- function(data, feature, target = salary) {

  # smoothing parameters
  k <- 0
  f <- 1

  # for NSE
  feature = dplyr::enquo(feature)
  target  = dplyr::enquo(target)

  data %>%

    dplyr::group_by(area_partner_child_segment, !!feature) %>%
    dplyr::summarise(
      n = n(),
      avg = mean(!!target, na.rm = T),
      min = min(!!target, na.rm = T),
      max = max(!!target, na.rm = T)
    ) %>%
#    dplyr::ungroup() %>%

    # area_partner_child_segment で grouping 中
    dplyr::mutate(
#      n_ratio = n / sum(n)

      ## smoothing
      lambda = 1 / (1 + exp(-(n - k) / f)),
      # avg
      segment_avg = sum(n * avg) / sum(n), # area_partner_child_segment 単位の平均値
      smoothed_avg = lambda * avg + (1 - lambda) * segment_avg,
      # min
      segment_min = min(min), # area_partner_child_segment 単位の最小値
      smoothed_min = lambda * min + (1 - lambda) * segment_min,
      # max
      segment_max = max(max), # area_partner_child_segment 単位の最大値
      smoothed_max = lambda * max + (1 - lambda) * segment_max
    ) %>%
    dplyr::ungroup() %>%

    dplyr::select(
      area_partner_child_segment,
      !!feature,
      n,

#      lambda,
#      avg,
      avg = smoothed_avg,
#      min,
      min = smoothed_min,
#      max,
      max = smoothed_max
    )
}
#recipes::prep(recipe) %>% recipes::juice() %>% smoothed_categorical_value(position, salary) %>% View


add_feature_per_category <- function(target_data, train_data, category_feature, target_feature) {

  # for NSE
  category_feature = dplyr::enquo(category_feature)
  target_feature   = dplyr::enquo(target_feature)

  # 新規に生成される項目名
  # ex. "position_avg_salary"
  new_col_name <- function(category_feature, target_feature, agg) {
    stringr::str_c(
      dplyr::quo_name(category_feature),
      agg,
      dplyr::quo_name(target_feature),
      sep = "_"
    )
  }

  # 代表値の一覧を取得
  df.category_average <- smoothed_categorical_value(train_data, !!category_feature, !!target_feature) %>%
    dplyr::select(
      area_partner_child_segment,
      !!category_feature,

      !!new_col_name(category_feature, target_feature, "avg")  := avg,
      !!new_col_name(category_feature, target_feature, "min")  := min,
      !!new_col_name(category_feature, target_feature, "max")  := max
    )

  # target_data に算出した代表値を結合
  target_data %>%
    dplyr::left_join(
      df.category_average,
      by = c("area_partner_child_segment", dplyr::quo_name(category_feature))
    )
}

add_mix_features_per_category <- function(target_data, category_feature, target_feature) {

  # for NSE
  category_feature = dplyr::enquo(category_feature)
  target_feature   = dplyr::enquo(target_feature)

  # 新規カラム名
  colname_avg = stringr::str_c(dplyr::quo_name(category_feature), "avg", dplyr::quo_name(target_feature), sep = "_")
  colname_min = stringr::str_c(dplyr::quo_name(category_feature), "min", dplyr::quo_name(target_feature), sep = "_")
  colname_max = stringr::str_c(dplyr::quo_name(category_feature), "max", dplyr::quo_name(target_feature), sep = "_")

  target_data %>%

    dplyr::mutate(
      !!stringr::str_c("diff",  colname_avg, sep = "_") := !!target_feature - !!dplyr::sym(colname_avg),
#      !!stringr::str_c("diff",  colname_min, sep = "_") := !!target_feature - !!dplyr::sym(colname_min),
#      !!stringr::str_c("diff",  colname_max, sep = "_") := !!target_feature - !!dplyr::sym(colname_max),
      !!stringr::str_c("ratio", colname_avg, sep = "_") := !!target_feature / !!dplyr::sym(colname_avg),
#      !!stringr::str_c("ratio", colname_min, sep = "_") := !!target_feature / !!dplyr::sym(colname_min),
#      !!stringr::str_c("ratio", colname_max, sep = "_") := !!target_feature / !!dplyr::sym(colname_max)
    )
}

add_statistic_features_per_category <- function(target_data, train_data, category_feature) {

  # for NSE
  category_feature = dplyr::enquo(category_feature)

  target_data %>%

    # salary
    add_feature_per_category(train_data, !!category_feature, salary) %>%

    # age
    add_feature_per_category(train_data, !!category_feature, age) %>%
    add_mix_features_per_category(!!category_feature, age) %>%
    
    # service_length
    add_feature_per_category(train_data, !!category_feature, service_length) %>%
    add_mix_features_per_category(!!category_feature, service_length) %>%
    
    # study_time
    add_feature_per_category(train_data, !!category_feature, study_time) %>%
    add_mix_features_per_category(!!category_feature, study_time) %>%

    # commute
    add_feature_per_category(train_data, !!category_feature, commute) %>%
    add_mix_features_per_category(!!category_feature, commute) %>%
    
    # overtime
    add_feature_per_category(train_data, !!category_feature, overtime) %>%
    add_mix_features_per_category(!!category_feature, overtime)
}


add_features_per_category <- function(target_data, train_data) {

  target_data %>%

    ## オリジナルカテゴリ

#    # position
#    add_statistic_features_per_category(train_data, position) %>%

   # area
   add_statistic_features_per_category(train_data, area) %>%

    # sex
    add_statistic_features_per_category(train_data, sex) %>%

#    # partner
#    add_statistic_features_per_category(train_data, partner) %>%

#    # education
#    add_statistic_features_per_category(train_data, education) %>%


    ## 生成カテゴリ

#    # area_partner_child_segment
#    add_statistic_features_per_category(train_data, area_partner_child_segment) %>%

    # position_education_partner_segment
    add_statistic_features_per_category(train_data, position_education_partner_segment) %>%

   # # position_education_segment
   # add_statistic_features_per_category(train_data, position_education_segment) %>%

    # flg_staff
    add_statistic_features_per_category(train_data, flg_staff) %>%

    # flg_newbie
    add_statistic_features_per_category(train_data, flg_newbie) 
  #%>%

#    # flg_bigcity
#    add_statistic_features_per_category(train_data, flg_bigcity) %>%

    # # flg_okinawa
    # add_statistic_features_per_category(train_data, flg_okinawa) %>%
    # 
    # # flg_kanagawa
    # add_statistic_features_per_category(train_data, flg_kanagawa) 
  # %>%
  # 
  #   # flg_area_partner_child_commute_high
  #   add_statistic_features_per_category(train_data, flg_area_partner_child_commute_high) %>%
  # 
  #   # flg_area_partner_child_commute_low
  #   add_statistic_features_per_category(train_data, flg_area_partner_child_commute_low) %>%
  # 
  #   # flg_area_partner_child_commute_extra_low
  #   add_statistic_features_per_category(train_data, flg_area_partner_child_commute_extra_low)
}


target_encoding <- function(data, target, category, k = 0, f = 1) {
  # for NSE
  target   = dplyr::enquo(target)
  category = dplyr::enquo(category)
  
  mean_col_name = stringr::str_c(
    dplyr::quo_name(category),
    "mean",
    dplyr::quo_name(target),
    sep = "_"
  )
  
  
  data %>%
    
    dplyr::group_by(!!category) %>%
    dplyr::mutate(
      s = sum(!!target),
      n = n()
    ) %>%
    dplyr::ungroup() %>%
    
    dplyr::mutate(
      # original
      mean = s / n,
      
      # with LOO
      with_loo = dplyr::if_else(n == 1, 0, (s - !!target) / (n - 1)),
      
      # with smoothing
      lambda = 1 / (1 + exp(-(n - k) / f)),
      smoothed = lambda * mean + (1 - lambda) * mean(!!target)
    ) %>%
    
    dplyr::select(-s, -n, -lambda)
  # %>%
  # 
  #   dplyr::rename(
  #     !!stringr::str_c(dplyr::quo_name(category), "mean",     dplyr::quo_name(target), sep = "_") := mean,
  #     !!stringr::str_c(dplyr::quo_name(category), "with_loo", dplyr::quo_name(target), sep = "_") := with_loo,
  #     !!stringr::str_c(dplyr::quo_name(category), "smoothed", dplyr::quo_name(target), sep = "_") := smoothed
  #   )
}
#target_encoding(df.train_data, salary, education)

get_dummies <- function(data) {
  recipes::recipe(salary ~ ., data) %>%
    recipes::step_dummy(all_nominal(), one_hot = T) %>%
    recipes::prep() %>%
    recipes::juice()
}
# label_hoge <- function(data, ...) {
# 
#   # for NSE
#   target_categories <- dplyr::quos(...) %>%
#     purrr::map_chr(dplyr::quo_name)
# 
#   data %>%
#     dplyr::mutate_at(target_categories, function(category) { as.integer(category) - 1L })
# }
# dplyr::select(df.train_data, position, education, area) %>%
#   label_hoge(area, position)



write_important_features <- function(fit) {
  xgboost::xgb.importance(model = fit$fit) %>%
    tibble::as_tibble() %>%
    readr::write_csv("./model/XGBoost_FE_3/feature_importances.csv", col_names = T)
}
#write_important_features(fit)

important_features <- function(threshold = 0.85) {

  from_algorithm <- readr::read_csv("./model/XGBoost_FE_3/feature_importances.csv") %>%
    dplyr::mutate(
      cumsum_gain = cumsum(Gain)
    ) %>%
    dplyr::filter(cumsum_gain < threshold) %>%
    .$Feature

  from_heuristic <- c(
    # area_partner_child_segment
    "area_partner_child_segment",
    # "area_partner_child_segment_avg_salary",
    # "area_partner_child_segment_avg_commute",
    # "diff_area_partner_child_segment_avg_commute",

    # flg_area_partner_child_commute_xxx
    "flg_area_partner_child_commute_high",
    "flg_area_partner_child_commute_low",
    "flg_area_partner_child_commute_extra_low",

    # # position
    # "position_avg_salary",
    # "position_avg_commute",
    # "diff_position_avg_commute",
    # "ratio_position_avg_commute",

    # # education
    # "education_avg_salary",
    # "diff_education_avg_commute",

    # position_education_partner_segment
    "diff_position_education_partner_segment_avg_commute",
    "ratio_position_education_partner_segment_avg_commute"
  )

  c(from_algorithm, from_heuristic) %>%
   unique()
}
#important_features()

