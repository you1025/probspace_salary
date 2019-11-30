source("functions.R", encoding = "utf-8")

# 警告(Too little...)が出てるので後でちゃんと調べる
create_cv <- function(data, v = 5, commute_length = 0.5, seed = 19) {

  # seed を固定: いくつか試して(1〜1000) 19 が良さげ
  set.seed(seed)

  data %>%

    dplyr::mutate(

      # area x partner x child
      # recipes::bake のタイミングで除去されるよ！
      area_partner_child_segment = dplyr::case_when(
        (area %in% c("東京都", "大阪府")) & (partner == 0) & (num_child == 0) ~ "bigcity_single_nochild",
        (area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child == 0) ~ "bigcity_family_nochild",
        (area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child >  0) ~ "bigcity_family_child",
        !(area %in% c("東京都", "大阪府")) & (partner == 0) & (num_child == 0) ~ "country_single_nochild",
        !(area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child == 0) ~ "country_family_nochild",
        !(area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child >  0) ~ "country_family_child"
      ) %>%
        factor(),

      # commute を分割
      commute_segment = cut(commute, breaks = seq(0, 5, commute_length)),

      cv_segment = stringr::str_c(
        area_partner_child_segment,
        commute_segment,
        sep = "_"
      )
    ) %>%

    # 不要項目の削除
    dplyr::select(-area_partner_child_segment, -commute_segment) %>%

    rsample::vfold_cv(
      v = v, strata = "cv_segment"
    )
}
#create_cv(df.train_data)


sample_sd <- function(x, na.rm) {
  sqrt(sum((mean(x, na.rm = na.rm) - x)^2) / length(x))
}

# area_partner_child_segment ごとに集計値を算出する
smoothed_categorical_value <- function(data, feature, target = salary, fun = mean) {

  # smoothing parameters
  k <- 0
  f <- 1

  # for NSE
  feature = dplyr::enquo(feature)
  target  = dplyr::enquo(target)

  # area_partner_child_segment 単位の集約値
  df.outer_segment <- data %>%
    dplyr::group_by(area_partner_child_segment) %>%
    dplyr::summarise(
      outer_stat = fun(!!target, na.rm = T)
    )

  # area_partner_child_segment & feature 単位の集約値
  df.inner_segment <- data %>%
    dplyr::group_by(area_partner_child_segment, !!feature) %>%
    dplyr::summarise(
      n = n(),
      inner_stat = fun(!!target, na.rm = T)
    ) %>%
    dplyr::ungroup()

  dplyr::left_join(
    df.inner_segment,
    df.outer_segment,
    by = "area_partner_child_segment"
  ) %>%

    # area_partner_child_segment で grouping 中
    dplyr::mutate(
      lambda = 1 / (1 + exp(-(n - k) / f)),
      smoothed_stat = lambda * inner_stat + (1 - lambda) * outer_stat
    ) %>%

    dplyr::select(
      area_partner_child_segment,
      !!feature,

      stat = smoothed_stat
    )
}
#recipes::prep(recipe) %>% recipes::juice() %>% smoothed_categorical_value(position, salary, sample_sd) %>% View


add_feature_per_category <- function(target_data, train_data, category_feature, target_feature, fun) {

  # for NSE
  category_feature = dplyr::enquo(category_feature)
  target_feature   = dplyr::enquo(target_feature)

  # 新規に生成される項目名
  # ex. "position_avg_salary"
  new_col_name <- stringr::str_c(
    dplyr::quo_name(category_feature),
    substitute(fun),
    dplyr::quo_name(target_feature),
    sep = "_"
  )

  # 代表値の一覧を取得
  df.category_average <- smoothed_categorical_value(train_data, !!category_feature, !!target_feature, fun) %>%
    dplyr::select(
      area_partner_child_segment,
      !!category_feature,

      !!new_col_name := stat
    )

  # target_data に算出した代表値を結合
  target_data %>%
    dplyr::left_join(
      df.category_average,
      by = c("area_partner_child_segment", dplyr::quo_name(category_feature))
    )
}

add_features_per_category <- function(target_data, train_data) {

  target_data %>%

    # position
    add_feature_per_category(train_data, position, commute, median) %>%

    # education
    add_feature_per_category(train_data, education, salary, max) %>%

    # position_education_partner
    add_feature_per_category(train_data, position_education_partner_segment, salary, median) %>%
    add_feature_per_category(train_data, position_education_partner_segment, commute, median) %>%

    # area
    add_feature_per_category(train_data, area, salary, median) %>%

    # area x partner x child
    add_feature_per_category(train_data, area_partner_child_segment, salary, mean) %>%
    add_feature_per_category(train_data, area_partner_child_segment, commute, mean) %>%
    dplyr::mutate(
      ratio_area_partner_child_segment_mean_commute = commute / area_partner_child_segment_mean_commute
    ) %>%
    dplyr::select(-area_partner_child_segment_mean_commute) %>%

    # flg_area_partner_child_commute_extra_high
    add_feature_per_category(train_data, flg_area_partner_child_commute_extra_high, commute, mean) %>%
    dplyr::mutate(
      ratio_flg_area_partner_child_commute_extra_high_mean_commute = commute / flg_area_partner_child_commute_extra_high_mean_commute
    ) %>%
    dplyr::select(-flg_area_partner_child_commute_extra_high_mean_commute) %>%

    # flg_area_partner_child_commute_high
    add_feature_per_category(train_data, flg_area_partner_child_commute_high, salary, mean) %>%
    add_feature_per_category(train_data, flg_area_partner_child_commute_high, salary, max) %>%
    add_feature_per_category(train_data, flg_area_partner_child_commute_high, commute, mean) %>%
    add_feature_per_category(train_data, flg_area_partner_child_commute_high, commute, median) %>%

    # flg_area_partner_child_commute_low
    add_feature_per_category(train_data, flg_area_partner_child_commute_low, salary, median) %>%
    add_feature_per_category(train_data, flg_area_partner_child_commute_low, commute, max) %>%
    add_feature_per_category(train_data, flg_area_partner_child_commute_low, commute, mean) %>%
    dplyr::mutate(
      ratio_flg_area_partner_child_commute_low_mean_commute = commute / flg_area_partner_child_commute_low_mean_commute
    ) %>%
    dplyr::select(-flg_area_partner_child_commute_low_mean_commute) %>%

    # flg_area_partner_child_commute_extra_low
    add_feature_per_category(train_data, flg_area_partner_child_commute_extra_low, salary, max) %>%
    add_feature_per_category(train_data, flg_area_partner_child_commute_extra_low, commute, mean) %>%
    add_feature_per_category(train_data, flg_area_partner_child_commute_extra_low, commute, max) %>%
    dplyr::mutate(
      ratio_flg_area_partner_child_commute_extra_low_mean_commute = commute / flg_area_partner_child_commute_extra_low_mean_commute,
      ratio_flg_area_partner_child_commute_extra_low_max_commute  = commute / flg_area_partner_child_commute_extra_low_max_commute
    ) %>%
    dplyr::select(-flg_area_partner_child_commute_extra_low_mean_commute, -flg_area_partner_child_commute_extra_low_max_commute) %>%

    hoge(train_data)
}

hoge <- function(target_data, train_data) {

  target_data %>%



    (function(x) { x })
}



# Category Column ---------------------------------------------------------

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

write_area_label <- function(data, category) {

  # for NSE
  category = dplyr::enquo(category)

  # 出力ファイル名
  # ex. area_label.csv
  filename <- stringr::str_c(dplyr::quo_name(category), "label.csv", sep = "_")
  filepath <- stringr::str_c("model/XGBoost_FE_5/", filename)

  data %>%

    dplyr::count(!!category, sort = T) %>%
    dplyr::mutate(
      !!category := forcats::fct_reorder(!!category, -n),
      label := as.integer(!!category) - 1L
    ) %>%

    dplyr::select(
      !!category,
      label
    ) %>%

    readr::write_csv(filepath, col_names = T)
}
#recipes::prep(recipe) %>% recipes::juice() %>% write_area_label(education)

replace_category_label <- function(target_data, category) {

  # for NSE
  category <- dplyr::enquo(category)

  filepath <- stringr::str_c("model/XGBoost_FE_5/", dplyr::quo_name(category), "_label.csv")

  # category と label(整数値) の対を取得
  df.labels <- readr::read_csv(
    filepath,
    col_types = cols(
      .default = col_character(),
      label = col_integer()
    )
  )

  target_data %>%

    dplyr::mutate(!!category := as.character(!!category)) %>%
    dplyr::left_join(df.labels, by = dplyr::quo_name(category)) %>%

    # category を label で置き換え
    dplyr::mutate(!!category := label) %>%
    dplyr::select(-label)
}
#replace_category_label(df.train_data, position)


get_dummies <- function(data) {
  recipes::recipe(~ ., data) %>%
    recipes::step_dummy(all_nominal(), one_hot = F) %>%
    recipes::prep() %>%
    recipes::juice()
}

transform_categories <- function(data) {

  data %>%

    # # Label-Encoding
    # replace_category_label(position) %>%
    # replace_category_label(area) %>%
    # replace_category_label(education) %>%
    # replace_category_label(position_education_partner_segment) %>%
    # replace_category_label(area_partner_child_segment)

    # OneHot-Encoding
    get_dummies()
}

scaling <- function(data) {
  # スケーリング対象外データの退避
  salary <- data$salary
  area_partner_child_segment <- data$area_partner_child_segment

  data %>%
    dplyr::select(-salary, -area_partner_child_segment) %>%
    get_dummies() %>%
    scale() %>%
    tibble::as_tibble() %>%
    dplyr::mutate(
      salary = salary,
      area_partner_child_segment = area_partner_child_segment
    )
}


create_lmer_model <- function(data) {

  model <- lme4::lmer(
    salary ~
      commute
      + age
      + num_child
##      + service_length
      + study_time
      + overtime
      + flg_staff
      + working_years
#      + flg_newbie
      + flg_area_partner_child_commute_extra_high
      + flg_area_partner_child_commute_high
      + flg_area_partner_child_commute_low
#      + flg_area_partner_child_commute_extra_low
      + position_median_commute
#      + education_max_salary
      + position_education_partner_segment_median_salary
      + position_education_partner_segment_median_commute
      + area_median_salary
#      + area_partner_child_segment_mean_salary
      + ratio_area_partner_child_segment_mean_commute
      + ratio_flg_area_partner_child_commute_extra_high_mean_commute
      + flg_area_partner_child_commute_high_mean_salary
      + flg_area_partner_child_commute_high_max_salary
#      + flg_area_partner_child_commute_high_mean_commute
      + flg_area_partner_child_commute_high_median_commute
      + flg_area_partner_child_commute_low_median_salary
      + ratio_flg_area_partner_child_commute_low_mean_commute
      + flg_area_partner_child_commute_extra_low_max_salary
#      + ratio_flg_area_partner_child_commute_extra_low_mean_commute
      + ratio_flg_area_partner_child_commute_extra_low_max_commute
##      + position_X1
      + position_X2
#      + position_X3
#      + position_X4
#      + area_奈良県
##      + area_山口県
#      + area_東京都
#      + area_鹿児島県
#      + area_兵庫県
      + area_神奈川県
#      + area_宮城県
#      + area_茨城県
#      + area_岩手県
#      + area_鳥取県
#      + area_岡山県
#      + area_愛媛県
#      + area_新潟県
#      + area_島根県
#      + area_和歌山県
      + area_熊本県
      + area_埼玉県
#      + area_大阪府
#      + area_大分県
      + area_徳島県
      + area_沖縄県
      + area_三重県
      + area_秋田県
      + area_福岡県
      + sex_X2
#      + partner_X1
      + education_X1
      + education_X2
      + education_X3
#      + education_X4
      + position_education_partner_segment_position_0_education_0_partner_1
#      + position_education_partner_segment_position_0_education_1_partner_0
      + position_education_partner_segment_position_0_education_1_partner_1
##      + position_education_partner_segment_position_0_education_2_partner_0
      + position_education_partner_segment_position_0_education_2_partner_1
#      + position_education_partner_segment_position_0_education_3_partner_0
#      + position_education_partner_segment_position_0_education_3_partner_1
#      + position_education_partner_segment_position_0_education_4_partner_0
#      + position_education_partner_segment_position_0_education_4_partner_1
##      + position_education_partner_segment_position_1_education_0_partner_0
      + position_education_partner_segment_position_1_education_0_partner_1
      + position_education_partner_segment_position_1_education_1_partner_1
      + position_education_partner_segment_position_1_education_2_partner_1
##      + position_education_partner_segment_position_1_education_3_partner_0
      + position_education_partner_segment_position_1_education_4_partner_0
#      + position_education_partner_segment_position_1_education_4_partner_1
      + position_education_partner_segment_position_2_education_0_partner_1
      + position_education_partner_segment_position_2_education_1_partner_1
      + position_education_partner_segment_position_2_education_2_partner_0
      + position_education_partner_segment_position_2_education_2_partner_1
#      + position_education_partner_segment_position_2_education_3_partner_0
#      + position_education_partner_segment_position_2_education_3_partner_1
#      + position_education_partner_segment_position_2_education_4_partner_0
#      + position_education_partner_segment_position_2_education_4_partner_1
      + position_education_partner_segment_position_3_education_0_partner_0
#      + position_education_partner_segment_position_3_education_0_partner_1
      + position_education_partner_segment_position_3_education_1_partner_0
#      + position_education_partner_segment_position_3_education_1_partner_1
##      + position_education_partner_segment_position_3_education_2_partner_1
      + position_education_partner_segment_position_3_education_3_partner_1
#      + position_education_partner_segment_position_3_education_4_partner_0
#      + position_education_partner_segment_position_3_education_4_partner_1
#      + position_education_partner_segment_position_4_education_0_partner_0
#      + position_education_partner_segment_position_4_education_0_partner_1
      + position_education_partner_segment_position_4_education_1_partner_0
#      + position_education_partner_segment_position_4_education_1_partner_1
      + position_education_partner_segment_position_4_education_2_partner_1
      + position_education_partner_segment_position_4_education_3_partner_1
      + position_education_partner_segment_position_4_education_4_partner_1

      + (1 + commute | area_partner_child_segment),

    data = data,
    control = lme4::lmerControl(optimizer = "bobyqa") # 警告の回避
  )

  print(model)

  model
}
