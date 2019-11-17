source("functions.R", encoding = "utf-8")

add_features_per_category <- function(target_data, train_data) {

  target_data %>%

    # position ごとの平均 salary を追加
    add_feature_per_category(train_data, position, salary) %>%

    # position ごとの commute 平均・差・比の追加
    add_feature_per_category(train_data, position, commute) %>%
    dplyr::mutate(
      diff_position_avg_commute  = commute - position_avg_commute
    ) %>%

    # area x partner x num_child ごとの commute 平均・差・比
    add_feature_per_category(train_data, area_partner_child_segment, commute) %>%
    dplyr::mutate(
      diff_area_partner_child_segment_avg_commute  = commute - area_partner_child_segment_avg_commute
    ) %>%

    # area x partner x num_child ごとの salary 平均
    add_feature_per_category(train_data, area_partner_child_segment, salary) %>%

    # flg_newbie ごとの salary 平均
    add_feature_per_category(train_data, flg_newbie, salary) %>%

    # flg_service_length_0 ごとの salary 平均
    add_feature_per_category(train_data, flg_service_length_0, salary)
}
#recipes::prep(recipe) %>% juice() %>% add_features_per_category(., .) %>% View

# 線形モデルを構築して予測値の一覧を返す
linear_model_prediction <- function(target_data, train_data) {

  # area_partner_child_segment ごとに線形回帰モデルを構築
  lst.models <- train_data %>%

    # area_partner_child_segment ごとにデータを分割
    dplyr::group_by(area_partner_child_segment) %>%
    tidyr::nest() %>%
    dplyr::ungroup() %>%

    # area_partner_child_segment ごとに線形回帰モデルを構築
    dplyr::mutate(
      model = purrr::map(data, function(data) {
        # あとでもっと増やしたい
        lm(
          log(salary) ~ commute + age + position_education_segment + flg_service_length_0_avg_salary,
          data
        )
      })
    ) %>%

    # tibble を list に変換
    dplyr::select(-data) %>%
    tibble::deframe()

  print(lst.models)

  # 予測値の一覧を生成
  # これもっとうまいことできないかな
  purrr::map_dbl(1:nrow(target_data), function(k, data, models) {
    # 対象行の area_partner_child_segment の値ごとにモデルを切り替える
    row <- data[k,]
    model <- models[as.character(row$area_partner_child_segment)][[1]]

    # 予測！
    predict(model, row)
  }, data = target_data, models = lst.models)
}

# 線形モデルを構築して予測値の一覧を返す
lmer_model_prediction <- function(target_data, train_data) {

  # area_partner_child_segment ごとに線形回帰モデルを構築
  model <- lme4::lmer(
    log(salary) ~ commute + age + position_education_segment + (1 + commute | area_partner_child_segment),
    data = train_data
  )

  # 予測値の一覧を生成
  # これもっとうまいことできないかな
  purrr::map_dbl(1:nrow(target_data), function(k, data, model) {
    row <- data[k,]
    predict(model, row)
  }, data = target_data, model = model)
}
# system.time(
#   recipes::prep(recipe) %>%
#     recipes::juice() %>%
#     add_features_per_category(., .) %>%
#     lmer_model_prediction(., .)
# )
#   100:   2.047
#   500:   5.138
#  1000:   9.355
#  2000:  17.345
#  5000:  38.423
# 10000:  76.477
# 21000: 164.660
tibble(
  n = c(100, 500, 1000, 2000, 5000, 10000, 21000),
  time = c(2.047, 5.138, 9.355, 17.345, 38.423, 76.477, 164.660)
) %>%
  ggplot(aes(n, time)) +
    geom_point() +
    geom_line()

# 線形モデルを構築して target_data に予測値のカラムを追加
add_linear_model_predictions <- function(target_data, train_data, pred_colnames = "predicted_salary") {

  # 予測値の取得
  preds <- lmer_model_prediction(target_data, train_data)

  # 予測値の追加
  target_data %>%
    dplyr::mutate(
      !!pred_colnames := preds
    )
}
# system.time(
#   df.hoge <- recipes::prep(recipe) %>%
#     recipes::juice() %>%
#     add_linear_model_predictions(., .)
# )

