# TODO
# - 異常値のクリッピング(OK)
# - カテゴリ内の平均 etc.(OK)
# - splitrule の変更を検討してみる(NG)
# - 四則演算による特徴量の導入
# - FrequencyEncoding
# - カテゴリ単位での commute の統計情報
# - カテゴリ単位での commute の統計情報と commute の 差/比


# OOB RMSE
# mtry = 12, min_n = 12, trees = 1400 : 32.60802
# Clipping: salary, commute, study_time: 32.23625 ↑
# 代表値の追加: position(mean): 32.14894 ↑
# 平社員フラグの追加: 32.12778 ↑

# mtry = 12, min_n = 12, trees = 1400 : 32.12778
# mtry = 12, min_n = 11, trees = 1500 : 32.12167
# mtry = 12, min_n = 10, trees = 1400 : 32.09429
# mtry = 12, min_n =  8, trees = 1400 : 32.07686

# position ごとの commute: 32.11157 ↓ orz
# area ごとの commute: 32.07686
# sex ごとの commute: 32.07686
# partner ごとの commute: 32.07686
# education ごとの commute: 32.07686

# position ごとの commute, 平均, 差, 比: 31.92813(☆)
# area     ごとの commute, 平均, 差, 比: 31.96778
# sex      ごとの commute, 平均, 差, 比: 31.99516
# partner  ごとの commute, 平均, 差, 比: 31.92813
# educationごとの commute, 平均, 差, 比: 32.14818 えーーーーorz


# 13:45〜
# mtry = 12, min_n = 12, trees = 1400 : 32.60802
# - Clipping
# - salary: 32.24373
# - commute, study_time: 32.23625
# ここからが本番
# カテゴリ代表値の追加: position, area, education: 31.96825 ↑

# mtry = xx, min_n = xx, trees = xxxx : xxx
# mtry = 12, min_n = 11, trees = 1500 : 31.93544 ↑
# mtry = 12, min_n = 11, trees = 1700 : 31.93267(☆) ↑

# 


library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv") %>%
  add_features_per_category()


# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  # clipping
  recipes::step_mutate(
    salary = ifelse(salary < 874.2022, salary, 874.2022),
    skip = T # train のみ
  ) %>%
  recipes::step_mutate(
    commute = ifelse(commute < 32.24373, commute, 32.24373),
#    ovetime = ifelse(overtime < 25.301, overtime, 25.301),
    study_time = ifelse(study_time <= 13, study_time, 13)
  ) %>%

  recipes::step_mutate(
#    overtime_commute = overtime + commute * 20
#    overtime_commute_studytime = overtime + commute * 20 + study_time * 4
#    commute_per_overtime = commute / overtime
#    studytime_per_overtime = ifelse(overtime > 0, study_time / overtime, NA)
#    studytime_per_commute = study_time / commute
#    commute_per_studytime = commute / study_time

#    age_servicelength = age - service_length
  ) %>%

  recipes::step_mutate(
    # 平社員フラグ
    flg_staff = (position == 0),

    commute_single_bigcity = ((partner == 0) &  (area %in% c("東京都", "大阪府"))) * commute,
    commute_single_country = ((partner == 0) & !(area %in% c("東京都", "大阪府"))) * commute,
    commute_family_bigcity = ((partner == 1) &  (area %in% c("東京都", "大阪府"))) * commute,
    commute_family_country = ((partner == 1) & !(area %in% c("東京都", "大阪府"))) * commute,

    # partner & num child
    partner_child = ifelse(partner == 0, "no_partner", stringr::str_c("child", num_child, sep = "_")) %>%
      factor(levels = c(
        "no_partner",
        stringr::str_c("child", 0:9, sep = "_")
      )),


    # flg_partner_area_commute_extra_high = ifelse(
    #   ((partner == 0)   & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1.5, 2))
    #   | ((partner == 0) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 2.5, 3))
    #   | ((partner == 1) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 2.5, 3))
    #   | ((partner == 1) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 4.5, 5)),
    #   1,
    #   0
    # ),
    # flg_partner_area_commute_high = ifelse(
    #   ((partner == 0) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1, 1.5))
    #   | ((partner == 0) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1.5, 2.5))
    #   | ((partner == 1) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1.5, 2.5))
    #   | ((partner == 1) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 3.5, 4.5)),
    #   1,
    #   0
    # ),
    # flg_partner_area_commute_high = ifelse(
    #   ((partner == 0) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1, 2))
    #   | ((partner == 0) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 2, 3))
    #   | ((partner == 1) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 2, 3))
    #   | ((partner == 1) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 4, 5)),
    #   1,
    #   0
    # ),
    flg_partner_area_commute_low = ifelse(
      ((partner == 0) & (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 0.5, 1))
      | ((partner == 1) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1, 1.5))
      | ((partner == 1) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1.5, 2.5)),
      1,
      0
    ),
    flg_partner_area_commute_extra_low = ifelse(
      ((partner == 0) & dplyr::between(commute, 0, 0.5))
      | ((partner == 1) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 0, 1))
      | ((partner == 1) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1, 1.5)),
      1,
      0
    ),

    # 子供ありフラグ
    flg_child = ifelse(num_child > 0, 1, 0),

    # エリアセグメント
    area_segment = dplyr::case_when(
      area %in% c("東京都", "大阪府") ~ "bigcity",
      T                               ~ "country"
    ) %>%
      factor(),
  ) %>%

  recipes::step_rm(
    id
  )

recipes::prep(recipe) %>% recipes::juice() %>% summary()


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
  dials::mtry(c(12, 12)),
  dials::min_n(c(11, 11)),
  dials::trees(c(1700, 1700)),
  levels = 1
)
df.grid.params


# Tuning ------------------------------------------------------------------

# 並列処理
#future::plan(future::multisession(workers = 4))
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
        recipes::juice()

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



# Importances -------------------------------------------------------------

model %>%

  {
    model <- (.)

    # 前処理済データの作成
    df.train <- recipe %>%
      recipes::prep() %>%
      recipes::juice()

    # 学習の実施
    parsnip::set_args(
      model,
      mtry = 12,
      min_n = 12,
      trees = 1400,
      importance = "permutation",
      num.threads = 8,
      seed = 1234
    ) %>%
      parsnip::fit(salary ~ ., df.train)
  } %>%

  .$fit %>%
  ranger::importance() %>%
  tibble::enframe(name = "feature", value = "importance") %>%
  dplyr::mutate(feature = forcats::fct_reorder(feature, importance)) %>%

  ggplot(aes(feature, importance)) +
    geom_col() +
    coord_flip() +
    theme_gray(base_family = "Osaka")



# Predict by Test Data ----------------------------------------------------

# モデルの学習
{
  # 前処理済データの作成
  df.train <- recipe %>%
    recipes::prep() %>%
    recipes::juice()

  # 学習の実施
  parsnip::set_args(
    model,
    mtry = 12,
    min_n = 12,
    trees = 1400,
    num.threads = 8,
    seed = 1234
  ) %>%
    parsnip::fit(salary ~ ., df.train)
} %>%

  # テストデータを用いた予測
  {
    fit <- (.)

    # 前処理済データの作成
    df.test <- load_test_data("data/input/test_data.csv") %>%
      add_features_per_category() %>%
      {
        test_data <- (.)
        recipes::prep(recipe) %>%
          recipes::bake(test_data)
      }

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
