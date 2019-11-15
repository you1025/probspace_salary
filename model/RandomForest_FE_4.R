# TODO
# - 記録上の値になるように RF を戻す(これはもうしゃーない)
# - CV を入れて MAE で評価する方式に切り替える(OK)
# - 特徴量エンジニアリング！ この前のとこまで(OK)
# - パラメータチューニング(OK)
# - max.depth を指定してみる(うーん・・・)
# - Submit! 今日はここまでが目標です！(OK)

# mtry: 12, min_n: 11, trees: 1700, oob_rmse: 33.6, train_mae: 12.3, test_mae: 22.4
# mtry: 12, min_n: 11, trees: 1700, oob_rmse: 32.99678, train_mae: 12.58253, test_mae: 22.0715
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.01727, train_mae: 12.96376, test_mae: 22.07845
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.03698, train_mae: 13.01689, test_mae: 22.08932 ↓ - 平社員フラグの導入
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.01727, train_mae: 12.96376, test_mae: 22.07845 ↑ - clipping(commute)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.01095, train_mae: 12.85109, test_mae: 22.08067 ↓ - clipping(overtime)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.01577, train_mae: 12.97231, test_mae: 22.08165 ↓ - clipping(study_time)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.41681, train_mae: 13.38839, test_mae: 22.22809    - カテゴリの代表値追加処理を除去(間違ってたorz)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.25839, train_mae: 13.3736,  test_mae: 22.07932 ↑ - position ごとの salary 平均を追加
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.28411, train_mae: 13.09873, test_mae: 22.08658 ↓ - area ごとの salary 平均(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.25422, train_mae: 13.44836, test_mae: 22.08818 ↓ - area_segment ごとの salary 平均(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.25572, train_mae: 13.43882, test_mae: 22.10189 ↓ - sex ごとの salary 平均(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.261,   train_mae: 13.44902, test_mae: 22.08931 ↓ - partner ごとの salary 平均(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.24336, train_mae: 13.4424,  test_mae: 22.09015 ↓ - num_child ごとの salary 平均(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.23214, train_mae: 13.44226, test_mae: 22.0801  ↓ - flg_child ごとの salary 平均(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.36009, train_mae: 13.44536, test_mae: 22.12514 ↓ - education ごとの salary 平均(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.22137, train_mae: 13.42774, test_mae: 22.06303 ↑ - position ごとの commute 平均
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.02435, train_mae: 13.47694, test_mae: 21.96275 ↑ - position ごとの commute 平均との差・比
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.03454, train_mae: 12.98316, test_mae: 22.09191 ↓ - area ごとの commute の平均・差・比(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 32.77051, train_mae: 13.62359, test_mae: 21.91387 ↑ - area_segment ごとの commute の平均・差・比
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 32.95606, train_mae: 13.82111, test_mae: 22.09609 ↓ - sex ごとの commute 平均・差・比(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 33.01,    train_mae: 13.90751, test_mae: 22.08793 ↓ - partner ごとの commute 平均・差・比(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 32.67231, train_mae: 13.76527, test_mae: 21.95321 ↓ - num_child ごとの commute 平均・差・比(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 32.64794, train_mae: 13.79577, test_mae: 21.92917 ↓ - flg_child ごとの commute 平均・差・比(除去)
# mtry: 12, min_n: 12, trees: 1400, oob_rmse: 32.88578, train_mae: 13.64312, test_mae: 21.96463 ↓ - education ごとの commute 平均・差・比(除去)

# mtry: xx, min_n: xx, trees: xxxx, oob_rmse: xxxxxxxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# mtry: 13, min_n: 12, trees: 1400, oob_rmse: 32.77024, train_mae: 13.48326, test_mae: 21.88381
# mtry: 14, min_n: 12, trees: 1400, oob_rmse: 32.72223, train_mae: 13.35133, test_mae: 21.87503
# mtry: 14, min_n: 11, trees: 1100, oob_rmse: 32.71504, train_mae: 12.96384, test_mae: 21.85096

# mtry: 14, min_n: 11, trees: 1100, oob_rmse: 43.56951, train_mae: 28.63744, test_mae: 29.50092



library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- rsample::vfold_cv(df.train_data, v = 4)


# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  # clipping
  recipes::step_mutate(
    commute = ifelse(commute < 32.24373, commute, 32.24373),
#    ovetime = ifelse(overtime < 25.301, overtime, 25.301),
#    study_time = ifelse(study_time <= 13, study_time, 13)
  ) %>%

  recipes::step_mutate(

    # # 平社員フラグ
    # flg_staff = (position == 0),

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
    num.threads = 2,
    seed = 1234
  )


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::mtry(c(14, 14)),
  dials::min_n(c(11, 11)),
  dials::trees(c(1100, 1100)),
  levels = 1
) %>%
  tidyr::crossing(max.depth = 15:18)
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
        df.train.baked <- recipe %>%
          recipes::prep() %>%
          recipes::bake(rsample::analysis(df.split))
        df.test.baked <- recipe %>%
          recipes::prep() %>%
          recipes::bake(rsample::assessment(df.split))

        ## 訓練/検証 データに代表値を付与
        df.train <- df.train.baked %>%
          add_features_per_category(df.train.baked)
        df.test <- df.test.baked %>%
          add_features_per_category(df.train.baked)


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
      mtry = 14,
      min_n = 11,
      trees = 1100,
      max.depth = 18,
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

  # 学習の実施
  model.fitted <- parsnip::set_args(
    model,
    mtry = 14,
    min_n = 11,
    trees = 1100,
#    max.depth = 18,
    num.threads = 8,
    seed = 1234
  ) %>%
    parsnip::fit(salary ~ ., df.train.baked)

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
