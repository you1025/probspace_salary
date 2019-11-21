source("functions.R", encoding = "utf-8")

add_features_per_category <- function(target_data, train_data) {

  target_data %>%

    # position ごとの平均 salary を追加
    add_feature_per_category(train_data, position, salary) %>%

    # position ごとの commute 平均・差・比の追加
    add_feature_per_category(train_data, position, commute) %>%
    dplyr::mutate(
      diff_position_avg_commute   = commute - position_avg_commute,
       ratio_position_avg_commute = commute / position_avg_commute
    ) %>%


    # education ごとの salary 平均
    add_feature_per_category(train_data, education, salary) %>%

    # education ごとの commute 差
    add_feature_per_category(train_data, education, commute) %>%
    dplyr::mutate(
      diff_education_avg_commute  = commute - education_avg_commute
    ) %>%
    dplyr::select(-education_avg_commute) %>%


    # # position x education ごとの salary 平均
    # add_feature_per_category(train_data, position_education_segment, salary) %>%

#    add_feature_per_category(train_data, position_education_partner_segment, salary) %>%
    add_feature_per_category(train_data, position_education_partner_segment, commute) %>%
    dplyr::mutate(
      diff_position_education_partner_segment_avg_salary  = commute - position_education_partner_segment_avg_commute,
      ratio_position_education_partner_segment_avg_salary = commute / position_education_partner_segment_avg_commute
    ) %>%
    dplyr::select(-position_education_partner_segment_avg_commute) %>%


    # area x partner x num_child ごとの salary 平均
    add_feature_per_category(train_data, area_partner_child_segment, salary) %>%

    # area x partner x num_child ごとの commute 平均・差・比
    add_feature_per_category(train_data, area_partner_child_segment, commute) %>%
    dplyr::mutate(
      diff_area_partner_child_segment_avg_commute  = commute - area_partner_child_segment_avg_commute,
      ratio_area_partner_child_segment_avg_commute = commute / area_partner_child_segment_avg_commute
    ) %>%

    # flg_newbie ごとの salary 平均
    add_feature_per_category(train_data, flg_newbie, salary) %>%
    add_feature_per_category(train_data, flg_newbie, commute)
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
    recipes::step_dummy(all_nominal()) %>%
    recipes::prep() %>%
    recipes::juice()
}
