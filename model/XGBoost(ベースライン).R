# Test MAE: 25.27212, 24.54160, 24.54194, 24.38002

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

model <- parsnip::boost_tree(
  mode = "regression",
  trees = parsnip::varying(),
  tree_depth = parsnip::varying(),     # max_depth
  mtry = parsnip::varying(),           # colsample_bytree
  min_n = parsnip::varying(),          # min_child_weight
  learn_rate = parsnip::varying(),
  loss_reduction = parsnip::varying(), # gamma
#  sample_size = parsnip::varying()     # subsample
  sample_size = 0.8
) %>%
  parsnip::set_engine(engine = "xgboost")


# Hyper Parameter ---------------------------------------------------------

# # 552 => trees
# recipe %>%
#   recipes::prep() %>%
#   recipes::juice() %>%
#   {
#     df.data <- (.)
#     x <- df.data %>% dplyr::select(-salary) %>% as.matrix()
#     y <- df.data$salary
# 
#     xgboost::xgb.cv(
#       params = list(
#         objective = "reg:linear",
#         eval_metric = "mae"
#       ),
# 
#       data = x,
#       label = y,
# 
#       nthread = 8,
# 
#       nfold = 5,
#       nrounds = 2000,
#       early_stopping_rounds = 100,
# 
#       eta = 0.1,
# 
#       max_depth = 5,
#       min_child_weight = 1,
#       gamma = 0,
#       subsample = 0.8,
#       colsample_bytree = 0.8
#     )
#   }

df.grid.params <- dials::grid_regular(
  dials::trees(c(552, 552)),

  dials::learn_rate(c(-1, -1)),

  dials::tree_depth(c(6, 6)),
  dials::min_n(c(2, 2)),

  dials::loss_reduction(c(-1.5714286, -1.5714286)),

  dials::mtry(c(60, 60)),             # 63 * 0.8 = 50

  levels = 4
) %>%
  dplyr::distinct() %>%
  tidyr::crossing(sample_size = seq(0.95, 0.95, length.out = 1))
df.grid.params


# Parametr Fitting --------------------------------------------------------

future::plan(future::multisession)

system.time(
  df.results <-

    # ハイパーパラメータをモデルに適用
    #  merge(df.grid.params, model) %>%
    purrr::pmap(df.grid.params, function(trees, learn_rate, tree_depth, min_n, loss_reduction, mtry, sample_size) {
      parsnip::set_args(
        model,
        trees = trees,
        learn_rate = learn_rate,
        tree_depth = tree_depth,
        min_n = min_n,
        loss_reduction = loss_reduction,
        sample_size = sample_size,
        mtry = mtry
      )
    }) %>%

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
      trees,
      learn_rate,
      tree_depth,
      min_n,
      loss_reduction,
      mtry,
      sample_size,

      train_mae,
      test_mae
    )
)



# # 2990 => trees
# recipe %>%
#   recipes::prep() %>%
#   recipes::juice() %>%
#   {
#     df.data <- (.)
#     x <- df.data %>% dplyr::select(-salary) %>% as.matrix()
#     y <- df.data$salary
# 
#     xgboost::xgb.cv(
#       params = list(
#         objective = "reg:linear",
#         eval_metric = "mae"
#       ),
# 
#       data = x,
#       label = y,
# 
#       nthread = 8,
# 
#       nfold = 5,
#       nrounds = 5000,
#       early_stopping_rounds = 100,
# 
#       eta = 0.01,
# 
#       max_depth = 6,
#       min_child_weight = 2,
#       gamma = 0.02682696,
#       colsample_bytree = 0.952381,
#       subsample = 0.95
#     )
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
    trees = 2990,
    learn_rate = 0.01,
    tree_depth = 6,
    min_n = 2,
    loss_reduction = 0.02682696,
    mtry = 60,
    sample_size = 0.95
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
      "XGBoost",
      lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
      sep = "_"
    ) %>%
      stringr::str_c("csv", sep = ".")

    # 出力ファイルパス
    filepath <- stringr::str_c("data/output", filename, sep = "/")

    # 書き出し
    readr::write_csv(df.result, filepath, col_names = T)
  }