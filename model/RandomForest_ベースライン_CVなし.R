# Test MAE: 47.72200, 37.69149, 33.37720, 29.13506, 26.98884
# OOB RMSE: 40.35716, 37.72067, 37.01268, 36.41928, 36.10955, 35.82377, 35.79773, 35.78491

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
  recipes::step_rm(id) %>%
  recipes::step_dummy(recipes::all_nominal())

# Model Definition --------------------------------------------------------

model <- parsnip::rand_forest(
  mode = "regression",
  mtry = parsnip::varying(),
  min_n = parsnip::varying(),
  trees = parsnip::varying()
) %>%
  parsnip::set_engine(engine = "ranger", num.threads = 2, seed = 1234)


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::mtry(c(37, 37)),
  dials::min_n(c(2, 3)),
  dials::trees(c(750, 1250)),
  levels = 5
)
df.grid.params


# Tuning ------------------------------------------------------------------

# 並列処理
future::plan(future::multisession(workers = 4))

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



