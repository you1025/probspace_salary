# Test MAE: 51.14197

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
  recipes::step_dummy(recipes::all_nominal())


# Model Definition --------------------------------------------------------

model <- parsnip::linear_reg(
  mode = "regression",
  penalty = parsnip::varying(),
  mixture = parsnip::varying()
) %>%
  parsnip::set_engine(engine = "glmnet")


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::penalty %>% dials::range_set(c(0, 0.5)),
  dials::mixture,
  levels = 10
)
df.grid.params



future::plan(future::multisession)

df.results <-

  # ハイパーパラメータをモデルに適用
  merge(df.grid.params, model) %>%

  # ハイパーパラメータの組み合わせごとにループ
  #  purrr::map(function(model.applied) {
  furrr::future_map(function(model.applied) {

    # クロスバリデーションの分割ごとにループ
    purrr::map(df.cv$splits, model = model.applied, function(df.split, model) {

      # 前処理済データの作成
      df.train <- recipe %>%
        recipes::prep() %>%
        recipes::bake(rsample::analysis(df.split))
      df.test <- recipe %>%
        recipes::prep() %>%
        recipes::bake(rsample::assessment(df.split))

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
            df.result.train,
            df.result.test
          )
        }
    }) %>%

      # CV 分割全体の平均値を評価スコアとする
      purrr::reduce(dplyr::bind_rows) %>%
      dplyr::summarise_all(mean)
  }) %>%

  # 評価結果とパラメータを結合
  purrr::reduce(dplyr::bind_rows) %>%
  dplyr::bind_cols(df.grid.params) %>%

  # 評価スコアの順にソート(昇順)
  dplyr::arrange(
    test_mae
  ) %>%

  dplyr::select(
    penalty,
    mixture,

    train_mae,
    test_mae
  )


