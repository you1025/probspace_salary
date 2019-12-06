source("functions.R", encoding = "utf-8")

create_cv <- function(data, v = 7, commute_length = 0.5, seed = 4630) {

  # seed を固定: いくつか試した(1〜5000)
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
    add_feature_per_category(train_data, position, salary, mean) %>%
    add_feature_per_category(train_data, position, salary, median) %>%
    add_feature_per_category(train_data, position, salary, max) %>%
    add_feature_per_category(train_data, position, commute, mean) %>%
    dplyr::mutate(
      diff_position_mean_commute  = commute - position_mean_commute,
      ratio_position_mean_commute = commute / position_mean_commute
    ) %>%
    add_feature_per_category(train_data, position, commute, median) %>%
    add_feature_per_category(train_data, position, commute, min) %>%

    # education
    add_feature_per_category(train_data, education, salary, mean) %>%
    add_feature_per_category(train_data, education, salary, max) %>%
    add_feature_per_category(train_data, education, commute, mean) %>%
    dplyr::mutate(
      diff_education_mean_commute  = commute - education_mean_commute
    ) %>%
    dplyr::select(-education_mean_commute) %>%

    # position_education_partner
    add_feature_per_category(train_data, position_education_partner_segment, commute, mean) %>%
    dplyr::mutate(
      diff_position_education_partner_segment_mean_salary  = commute - position_education_partner_segment_mean_commute,
      ratio_position_education_partner_segment_mean_salary = commute / position_education_partner_segment_mean_commute
    ) %>%
    add_feature_per_category(train_data, position_education_partner_segment, commute, max) %>%
    dplyr::select(-position_education_partner_segment_mean_commute) %>%

    # area x partner x num_child
    add_feature_per_category(train_data, area_partner_child_segment, salary, mean) %>%
    add_feature_per_category(train_data, area_partner_child_segment, salary, min) %>%
    add_feature_per_category(train_data, area_partner_child_segment, commute, mean) %>%
    dplyr::mutate(
      diff_area_partner_child_segment_mean_commute  = commute - area_partner_child_segment_mean_commute,
      ratio_area_partner_child_segment_mean_commute = commute / area_partner_child_segment_mean_commute
    ) %>%

    # flg_newbie ごとの salary 平均
    add_feature_per_category(train_data, flg_newbie, salary, mean) %>%
    add_feature_per_category(train_data, flg_newbie, commute, mean)
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

get_dummies <- function(data) {
  recipes::recipe(salary ~ ., data) %>%
    recipes::step_dummy(all_nominal()) %>%
    recipes::prep() %>%
    recipes::juice()
}

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

transform_categories <- function(data) {

  data %>%

    # Label-Encoding
    replace_category_label(position) %>%
    replace_category_label(area) %>%
    replace_category_label(education) %>%
    replace_category_label(position_education_partner_segment) %>%
    replace_category_label(area_partner_child_segment)

    # # OneHot-Encoding
    # get_dummies()
}



# 線形モデルを構築して予測値の一覧を返す
linear_model_prediction <- function(target_data, train_data) {

  # commute のビン化
  bin_width <- 0.5
  df.target_data.with_bin <- target_data %>%
    dplyr::mutate(
      bin_commute = cut(commute, breaks = seq(0, 5, bin_width))
    )
  df.train_data.with_bin <- train_data %>%
    dplyr::mutate(
      bin_commute = cut(commute, breaks = seq(0, 5, bin_width))
    )


  # モデル定義: ElasticNet
  model <- parsnip::linear_reg(
    mode = "regression",
    penalty = 0.05,
    mixture = 0.05
  ) %>%
    parsnip::set_engine(engine = "glmnet") %>%

    # 学習
    parsnip::fit(
      salary ~
        # 交互作用
        commute * area_partner_child_segment * bin_commute

        + flg_area_partner_child_commute_low
        + flg_area_partner_child_commute_extra_low,
      df.train_data.with_bin
    ) %>%

    # 予測
    predict(df.target_data.with_bin, type = "numeric") %>%
    .$.pred
}

# 線形モデルを構築して target_data に予測値のカラムを追加
add_linear_model_predictions <- function(target_data, train_data, pred_colname = "predicted_salary") {

  # 予測値の取得
  preds <- linear_model_prediction(target_data, train_data)

  # 予測値の追加
  target_data %>%
    dplyr::mutate(
      !!pred_colname := preds
      # ,
      # 
      # # 予測値との差分
      # diff_salary = salary - preds
    )
}
