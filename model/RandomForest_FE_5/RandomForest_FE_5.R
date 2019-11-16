# mtry: xx, min_n: xx, max.depth: xx, oob_rmse: xxxxxxxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# mtry:  4, min_n:  4, max.depth:  6, oob_rmse: 59.50211, train_mae: 40.65735,  test_mae: 41.36683
# mtry: 10, min_n:  4, max.depth: 16, oob_rmse: 37.14770, train_mae: 11.945572, test_mae: 23.85419 - 仮 fix!

# oob_rmse: 37.272,   train_mae: 11.97903, test_mae: 23.92324 ↓ - clipping: commute なぜや orz
# oob_rmse: 37.14512, train_mae: 11.9446,  test_mae: 23.85259 ↑ - clipping: overtime
# oob_rmse: 37.13773, train_mae: 11.93276, test_mae: 23.85479 ↑ - clipping: study_time
# oob_rmse: 37.16269, train_mae: 11.98118, test_mae: 23.89184 ↓ - flg_child orz
# oob_rmse: 35.02323, train_mae: 11.63978, test_mae: 22.93842 ↑ - area_segment めっちゃ効く
# oob_rmse: 34.37194, train_mae: 11.63929, test_mae: 22.63041 ↑ - area_partner_child_segment_avg_commute
# oob_rmse: 33.08677, train_mae: 11.07644, test_mae: 22.11169 ↑ - diff_area_partner_child_segment_avg_commute
# oob_rmse: 33.25382, train_mae: 11.19377, test_mae: 22.29098 ↓ - ratio_area_partner_child_segment_avg_commute なんでやorz
# oob_rmse: 33.01218, train_mae: 11.14548, test_mae: 22.07682 ↑ - area_partner_child_segment_avg_salary
# oob_rmse: 32.84733, train_mae: 11.12229, test_mae: 21.89498 ↑ - position_avg_salary
# oob_rmse: 32.77698, train_mae: 11.1647,  test_mae: 21.85294 ↑ - position_avg_commute
# oob_rmse: 32.72569, train_mae: 11.22042, test_mae: 21.83474 ↑ - diff_position_avg_commute
# oob_rmse: 32.75552, train_mae: 11.30228, test_mae: 21.8541  ↓ - ratio_position_avg_commute
# oob_rmse: 32.75945, train_mae: 11.23381, test_mae: 21.83552 ↑ - area_segment の除去 重複はいかんのね

# oob_rmse: 32.77093, train_mae: 11.31414, test_mae: 21.84638 ↓ d
# oob_rmse: 32.76306, train_mae: 11.30282, test_mae: 21.85519 ↓ flg_newbie orz なんでや
# oob_rmse: 32.22213, train_mae: 11.12597, test_mae: 21.6463  ↑ commute の clipping を除去(漏れてた orz)

# oob_rmse: 32.22815, train_mae: 11.1933,  test_mae: 21.64394 ↑ flg_newbie ふたたび 上がった！
# oob_rmse: 32.16321, train_mae: 11.1874,  test_mae: 21.57617 ↑ flg_newbie_avg_salary
# oob_rmse: 32.16903, train_mae: 11.25993, test_mae: 21.6084  ↓ flg_newbie_avg_commute orz
# oob_rmse: 32.23755, train_mae: 11.38058, test_mae: 21.65541 ↓ diff_flg_newbie_avg_commute orz
# oob_rmse: 32.24876, train_mae: 11.36708, test_mae: 21.64429 ↓ ratio_flg_newbie_avg_commute
# oob_rmse: 32.14619, train_mae: 11.19436, test_mae: 21.55949 ↑ d <= 1

# oob_rmse: 32.10909, train_mae: 11.19952, test_mae: 21.50674 ↑ position_education_segment
# oob_rmse: 32.25153, train_mae: 11.2265,  test_mae: 21.56345 ↓ position_education_segment_avg_salary orz
# oob_rmse: 32.26012, train_mae: 11.22341, test_mae: 21.55375 ↓ position_education_segment_avg_commute
# oob_rmse: 32.31746, train_mae: 11.33384, test_mae: 21.62226 ↓ diff_position_education_segment_avg_commute
# oob_rmse: 32.32254, train_mae: 11.32898, test_mae: 21.60248 - ratio_position_education_segment_avg_commute

# oob_rmse: 32.07058, train_mae: 11.24363, test_mae: 21.51137 ↓ flg_staff
# oob_rmse: 32.08282, train_mae: 11.33021, test_mae: 21.5036  ↑ flg_staff_avg_salary
# oob_rmse: 32.07885, train_mae: 11.40731, test_mae: 21.5159  ↓ flg_staff_avg_commute
# oob_rmse: 32.12753, train_mae: 11.50094, test_mae: 21.52884 ↓ diff_flg_staff_avg_commute
# oob_rmse: 32.12567, train_mae: 11.49791, test_mae: 21.52473 ↓ ratio_flg_staff_avg_commute

# oob_rmse: 32.08258, train_mae: 11.25854, test_mae: 21.50858 ↓ flg_service_length_0
# oob_rmse: 32.07073, train_mae: 11.3389,  test_mae: 21.49953 ↑ flg_service_length_0_avg_salary 上がった!
# oob_rmse: 32.10458, train_mae: 11.42887, test_mae: 21.51673 ↓ flg_service_length_0_avg_commute orz

# oob_rmse: 32.08581, train_mae: 11.40562, test_mae: 21.51113 ↓ flg_study_time_0
# oob_rmse: 32.08479, train_mae: 11.45423, test_mae: 21.5175  ↓ flg_study_time_0_avg_salary

# oob_rmse: xxxxxxxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# oob_rmse: 32.07556, train_mae: 11.46915, test_mae: 21.49952 ↑ flg_overtime_0
# oob_rmse: 32.0644,  train_mae: 11.54174, test_mae: 21.51372 ↓ flg_overtime_0_avg_salary
# oob_rmse: 32.0644,  train_mae: 11.54174, test_mae: 21.51372 ↓ flg_overtime_0_avg_commute

# mtry: 11, min_n: xx, trees: xxxx, max.depth: 15, oob_rmse: xxxxxxxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# mtry: 11, min_n:  4, trees:  500, max.depth: 15, oob_rmse: 32.06794, train_mae: 12.12825, test_mae: 21.48355
# mtry: 11, min_n:  5, trees:  750, max.depth: 15, oob_rmse: 32.04988, train_mae: 12.43028, test_mae: 21.48276
# mtry: 11, min_n:  7, trees:  750, max.depth: 15, oob_rmse: 32.03905, train_mae: 13.04593, test_mae: 21.47146
# mtry: 11, min_n:  7, trees:  700, max.depth: 15, oob_rmse: 32.04384, train_mae: 13.04598, test_mae: 21.47109
# mtry: 11, min_n:  7, trees:  600, max.depth: 15, oob_rmse: 32.04841, train_mae: 13.04754, test_mae: 21.46919
# mtry: 11, min_n:  7, trees:  550, max.depth: 15, oob_rmse: 32.04987, train_mae: 13.04984, test_mae: 21.46758


library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/RandomForest_FE_5/functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- rsample::vfold_cv(df.train_data, v = 4)


# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  # clipping
  recipes::step_mutate(
    overtime = ifelse(overtime < 25.301, overtime, 25.301),
    study_time = ifelse(study_time <= 13, study_time, 13)
  ) %>%

  recipes::step_mutate(
    # area x partner x child
    area_partner_child_segment = dplyr::case_when(
       (area %in% c("東京都", "大阪府")) & (partner == 0) & (num_child == 0) ~ "bigcity_single_nochild",
       (area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child == 0) ~ "bigcity_family_nochild",
       (area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child >  0) ~ "bigcity_family_child",
      !(area %in% c("東京都", "大阪府")) & (partner == 0) & (num_child == 0) ~ "country_single_nochild",
      !(area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child == 0) ~ "country_family_nochild",
      !(area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child >  0) ~ "country_family_child"
    ),

    # 新卒フラグ
    base_age = dplyr::case_when(
      education == 0 ~ 18,
      education == 1 ~ 20,
      education == 2 ~ 22,
      education == 3 ~ 24,
      education == 4 ~ 27
    ),
    d = age - base_age - service_length,
    flg_newbie = (d <= 1),

    # position x education
    position_education_segment = stringr::str_c("position", position, "education", education, sep = "_") %>%
      factor(),

    # service_length == 0
    flg_service_length_0 = (service_length == 0),

    flg_overtime_0 = (overtime == 0)
  ) %>%

  recipes::step_rm(
    id,

    # 新卒フラグ(flg_newbie)関連
    base_age,
    d
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
    num.threads = 2,
    seed = 1234
  )


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::mtry(c(11, 11)),
  dials::min_n(c(7, 7)),
  dials::trees(c(550, 550)),
  levels = 1
) %>%
  tidyr::crossing(max.depth = 15)
df.grid.params


# Tuning ------------------------------------------------------------------

# 並列処理
furrr::future_options(seed = F)
future::plan(future::multisession(workers = 4))
#future::plan(future::sequential)


system.time(

  df.results <-

    # ハイパーパラメータをモデルに適用
    purrr::pmap(df.grid.params, function(mtry, min_n, trees, max.depth) {
      parsnip::set_args(
        model,
        mtry = mtry,
        min_n = min_n,
        trees = trees,

        max.depth = max.depth
      )
    }) %>%

    # ハイパーパラメータの組み合わせごとにループ
    furrr::future_map_dfr(function(model.applied) {

      # クロスバリデーションの分割ごとにループ
      purrr::map_dfr(df.cv$splits, model = model.applied, function(df.split, model) {

        # 前処理済データの作成
        df.train <- recipe %>%
          recipes::prep() %>%
          recipes::bake(rsample::analysis(df.split)) %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(., .)
        df.test <- recipe %>%
          recipes::prep() %>%
          recipes::bake(rsample::assessment(df.split)) %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(df.train)

        model %>%

          # モデルの学習
          {
            model <- (.)
            parsnip::fit(model, salary ~ ., df.train)
          } %>%

          # 学習済モデルによる予測
          {
            fit <- (.)
            list(
              oob_rmse = sqrt(fit$fit$prediction.error),
              train = predict(fit, df.train, type = "numeric")[[1]],
              test  = predict(fit, df.test,  type = "numeric")[[1]]
            )
          } %>%

          # 評価
          {
            lst.predicted <- (.)

            # 評価指標の一覧を定義
            metrics <- yardstick::metric_set(
              yardstick::mae
            )

            # train データでモデルを評価
            df.result.train <- df.train %>%
              dplyr::mutate(
                predicted = lst.predicted$train
              ) %>%
              metrics(truth = salary, estimate = predicted) %>%
              dplyr::select(-.estimator) %>%
              dplyr::mutate(
                .metric = stringr::str_c("train", .metric, sep = "_")
              ) %>%
              tidyr::spread(key = .metric, value = .estimate)

            # test データでモデルを評価
            df.result.test <- df.test %>%
              dplyr::mutate(
                predicted = lst.predicted$test
              ) %>%
              metrics(truth = salary, estimate = predicted) %>%
              dplyr::select(-.estimator) %>%
              dplyr::mutate(
                .metric = stringr::str_c("test", .metric, sep = "_")
              ) %>%
              tidyr::spread(key = .metric, value = .estimate)

            dplyr::bind_cols(
              oob_rmse = lst.predicted$oob_rmse,
              df.result.train,
              df.result.test
            )
          }
      }) %>%

        # CV 分割全体の平均値を評価スコアとする
        dplyr::summarise_all(mean)
    }) %>%

    # 評価結果とパラメータを結合
    dplyr::bind_cols(df.grid.params) %>%

    # 評価スコアの順にソート(昇順)
    dplyr::arrange(
      test_mae
    ) %>%

    dplyr::select(
      mtry,
      min_n,
      trees,
      
      max.depth,
      
      oob_rmse,
      train_mae,
      test_mae
    )
)


# Importances -------------------------------------------------------------

model %>%

  {
    model <- (.)

    # 前処理済データの作成
    df.train <- recipe %>%
      recipes::prep() %>%
      recipes::juice() %>%
      add_features_per_category(., .)

    # 学習の実施
    parsnip::set_args(
      model,
      mtry = 11,
      min_n = 7,
      trees = 550,
      max.depth = 15,
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
  df.train.baked <- recipe %>%
    recipes::prep() %>%
    recipes::bake(df.train_data)
  df.train <- df.train.baked %>%
    add_features_per_category(., .)

  # 学習の実施
  model.fitted <- parsnip::set_args(
    model,
    mtry = 11,
    min_n = 7,
    trees = 550,
    max.depth = 15,
    num.threads = 8,
    seed = 1234
  ) %>%
    parsnip::fit(salary ~ ., df.train)

  list(
    df.train.baked = df.train.baked,
    model.fitted = model.fitted
  )
} %>%

  # テストデータを用いた予測
  {
    lst.results <- (.)

    # 学習済みモデル
    fit <- lst.results$model.fitted

    # 前処理済データの作成
    df.test <- recipe %>%
      recipes::prep() %>%
      recipes::bake(load_test_data("data/input/test_data.csv")) %>%
      add_features_per_category(lst.results$df.train.baked)

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
