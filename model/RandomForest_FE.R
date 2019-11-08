# OOB RMSE
# - スタート: 35.6513
#   - ダミー変換時に１つ除去なし: 35.32794
# - position_education の追加
#   - 不要項目の削除なし: 36.00631
#   - 不要項目の削除あり: 35.58131
# - 入社区分の追加
#   - 不要項目の削除なし: 36.69857
#   - 不要項目の削除あり: 35.7186
# - partner_child の追加
#   - 不要項目の削除なし: 35.69291
#   - 不要項目の削除あり: 36.09685
# - エリア区分を追加: 35.70194
#   - area の削除: 35.234
# - 入社区分の排除: 35.28539
# - 入社区分を戻して第2新卒を除外: 35.24157
# - 第二新卒を新卒の方に入れ込む: 35.24939
# - partner_child の子供4人以上をまとめる: 35.54597 なぜorz
# - 3人以上に変更: 36.12041
# - 5人以上に変更: 35.364
# - 6人以上に変更: 35.34541

library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- rsample::vfold_cv(df.train_data, v = 5)


# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  recipes::step_mutate(

    # 入社区分
    base_age = dplyr::case_when(
      education == 0 ~ 18,
      education == 1 ~ 20,
      education == 2 ~ 22,
      education == 3 ~ 24,
      education == 4 ~ 27
    ),
    d = age - base_age - service_length,
    join_segment = dplyr::case_when(
      d %in% 0:1 ~ "d_0",   # 新卒・第二新卒
      T          ~ "others" # 中途その他
    ) %>%
      factor(),

    # position x education
    position_education = dplyr::case_when(
      # position: 0
      (position == 0) & (education %in% 0:1) ~ "position_0_x_education_0_1",
      (position == 0) & (education == 2)     ~ "position_0_x_education_2",
      (position == 0) & (education == 3)     ~ "position_0_x_education_3",
      (position == 0) & (education == 4)     ~ "position_0_x_education_4",

      # position: 1
      (position == 1) & (education %in% 0:1) ~ "position_1_x_education_0_1",
      (position == 1) & (education == 2)     ~ "position_1_x_education_2",
      (position == 1) & (education == 3)     ~ "position_1_x_education_3",
      (position == 1) & (education == 4)     ~ "position_1_x_education_4",

      # position: 2
      (position == 2) & (education %in% 0:1) ~ "position_2_x_education_0_1",
      (position == 2) & (education == 2)     ~ "position_2_x_education_2",
      (position == 2) & (education == 3)     ~ "position_2_x_education_3",
      (position == 2) & (education == 4)     ~ "position_2_x_education_4",

      # position: 3
      (position == 3) & (education %in% 0:1) ~ "position_3_x_education_0_1",
      (position == 3) & (education == 2)     ~ "position_3_x_education_2",
      (position == 3) & (education == 3)     ~ "position_3_x_education_3",
      (position == 3) & (education == 4)     ~ "position_3_x_education_4",

      # position: 4
      (position == 4) & (education %in% 0:1) ~ "position_4_x_education_0_1",
      (position == 4) & (education == 2)     ~ "position_4_x_education_2",
      (position == 4) & (education == 3)     ~ "position_4_x_education_3",
      (position == 4) & (education == 4)     ~ "position_4_x_education_4"
    ) %>%
      factor(),

    # partner & num child
    partner_child = dplyr::case_when(
      partner == 0 ~ "no_partner",
      (partner == 1) & (num_child == 0) ~ "child_0",
      (partner == 1) & (num_child == 1) ~ "child_1",
      (partner == 1) & (num_child == 2) ~ "child_2",
      (partner == 1) & (num_child == 3) ~ "child_3",
      (partner == 1) & (num_child == 4) ~ "child_4",
      (partner == 1) & (num_child == 5) ~ "child_5",
      (partner == 1) & (num_child >= 6) ~ "child_gte_6"
    ) %>%
      factor(),

    # エリア区分
    area_segment = dplyr::case_when(
      area %in% c("東京都", "大阪府") ~ "big_city",
      area == "沖縄県"                ~ "okinawa",
      T                               ~ "others"
    ) %>%
      factor()
  ) %>%

  recipes::step_rm(
    id,

    # join_segment
    base_age,
    d,

    # position_education
    position,
    education,

    # partner_child
    partner,
    num_child,

    # area_segment
    area
  ) %>%

  recipes::step_dummy(recipes::all_nominal(), one_hot = T)

#recipes::prep(recipe) %>% juice() %>% summary()


# Model Definition --------------------------------------------------------

model <- parsnip::rand_forest(
  mode = "regression",
  mtry = parsnip::varying(),
  min_n = parsnip::varying(),
  trees = parsnip::varying()
) %>%
  parsnip::set_engine(
    engine = "ranger",
    num.threads = 8,
    seed = 1234
  )


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::mtry(c(37, 37)),
  dials::min_n(c(2, 2)),
  dials::trees(c(1125, 1125)),
  levels = 1
)
df.grid.params


# Tuning ------------------------------------------------------------------

# 並列処理
future::plan(future::sequential)

system.time(
  df.results <-

    # ハイパーパラメータをモデルに適用
    # merge(df.grid.params, model) %>%
    purrr::pmap(df.grid.params, function(mtry, min_n, trees) {
      parsnip::set_args(
        model,
        mtry = mtry,
        min_n = min_n,
        trees = trees
      )
    }) %>%

    # ハイパーパラメータの組み合わせごとにループ
    #  purrr::map(function(model.applied) {
    furrr::future_map_dfr(function(model.applied) {

      # 前処理済データの作成
      df.train <- recipe %>%
        recipes::prep() %>%
        recipes::bake(df.train_data)

      fit <- parsnip::fit(model.applied, salary ~ ., df.train)
      tibble(oob_rmse = sqrt(fit$fit$prediction.error))
    }) %>%

    # 評価結果とパラメータを結合
    dplyr::bind_cols(df.grid.params) %>%

    # 評価スコアの順にソート(昇順)
    dplyr::arrange(
      oob_rmse
    ) %>%

    dplyr::select(
      mtry,
      min_n,
      trees,

      oob_rmse
    )
)


# Importance --------------------------------------------------------------

# model %>%
#   {
#     model <- (.) %>%
#       parsnip::set_args(importance = "permutation")
# 
#     # 前処理済データの作成
#     df.train <- recipe %>%
#       recipes::prep() %>%
#       recipes::bake(df.train_data)
#     
#     # 学習の実施
#     parsnip::set_args(
#       model,
#       mtry = 37,
#       min_n = 2,
#       trees = 1125,
#       num.threads = 8,
#       seed = 1234
#     ) %>%
#       parsnip::fit(salary ~ ., df.train)
#   } %>%
#   {
#     fit <- (.)
#     ranger::importance(fit$fit) %>%
#       tibble::enframe(name = "feature", value = "importance") %>%
#       dplyr::arrange(desc(importance)) %>%
#       dplyr::mutate(feature = forcats::fct_reorder(feature, importance)) %>%
#       ggplot(aes(feature, importance)) +
#         geom_col() +
#         theme_gray(base_family = "Osaka") +
#         coord_flip()
#   }


# Predict by Test Data ----------------------------------------------------

# モデルの学習
{
  # 前処理済データの作成
  df.train <- recipe %>%
    recipes::prep() %>%
    recipes::bake(df.train_data)

  # 学習の実施
  parsnip::set_args(
    model,
    mtry = 37,
    min_n = 2,
    trees = 1125,
    num.threads = 8,
    seed = 1234
  ) %>%
    parsnip::fit(salary ~ ., df.train)
} %>%

  # テストデータを用いた予測
  {
    fit <- (.)

    # 前処理済データの作成
    df.test <- recipe %>%
      recipes::prep() %>%
      recipes::bake(load_test_data("data/input/test_data.csv"))

    # 予測結果データセット
    tibble(
      id = 0:(nrow(df.test)-1),
      y = predict(fit, df.test, type = "numeric")[[1]]
    )
  } %>%

  {
    df.result <- (.)

    # ファイル名
    filename <- stringr::str_c(
      "RandomForest",
      lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
      sep = "_"
    ) %>%
      stringr::str_c("csv", sep = ".")

    # 出力ファイルパス
    filepath <- stringr::str_c("data/output", filename, sep = "/")

    # 書き出し
    readr::write_csv(df.result, filepath, col_names = T)
  }
