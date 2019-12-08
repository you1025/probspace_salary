# TODO
# - 適当なパラメータ探し(OK)
# - 対数変換(OK)
# - 特徴量の追加

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - xxx
# train_mae: 1188.048, test_mae: 1177.353 - 初期値
# train_mae: 60.50451, test_mae: 60.79072 - min_data = 2500 お試し
# train_mae: 226.4191, test_mae: 226.4209 - ベースラインとかいいうのに合わせた orz
# R と python で初期値が異なるのかな？？？

# train_mae: 19.65834, test_mae: 26.11733 - カテゴリを全て LabelEncoding ようやくまともな値に
# train_mae: 19.23358, test_mae: 26.44987 - max_depth = 7
# train_mae: 20.38744, test_mae: 24.8537  - cv を 7 fold に変更(忘れてた)
# train_mae: 20.38744, test_mae: 24.8537  - パラメータ指定を動的に変更できるように修正 - baseline


library(tidyverse)
library(tidymodels)
library(furrr)
library(lightgbm)

source("model/LightGBM_01_BaseLine/functions.R", encoding = "utf-8")

# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- create_cv(df.train_data, v = 7)

# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  # Label Encoding
  recipes::step_mutate(
    position  = as.integer(position)  - 1,
    area      = as.integer(area)      - 1,
    sex       = as.integer(sex)       - 1,
    partner   = as.integer(partner)   - 1,
    education = as.integer(education) - 1
  ) %>%

  # 対数変換: salary
  recipes::step_log(salary, offset = 1) %>%

  recipes::step_rm(
    id
  )

recipes::prep(recipe) %>% recipes::juice()


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- tibble(
  max_depth = 7,
  num_leaves = 31,
  bagging_freq = 5,
  bagging_fraction = 0.8,
  feature_fraction = 0.9,
  min_data_in_leaf = 20,

  lambda_l1 = 0,
  lambda_l2 = 0
)
df.grid.params


# Parametr Fitting --------------------------------------------------------

# 並列処理
future::plan(future::multisession(workers = 8))

system.time({

  df.results <- purrr::pmap_dfr(df.grid.params, function(max_depth, num_leaves, bagging_freq, bagging_fraction, feature_fraction, min_data_in_leaf, lambda_l1, lambda_l2) {

    hyper_params <- list(
      max_depth = max_depth,
      num_leaves = num_leaves,
      bagging_freq = bagging_freq,
      bagging_fraction = bagging_fraction,
      feature_fraction = feature_fraction,
      min_data_in_leaf = min_data_in_leaf,

      lambda_l1 = lambda_l1,
      lambda_l2 = lambda_l2
    )

    furrr::future_map_dfr(df.cv$splits, function(split, recipe, hyper_params) {
      print(hyper_params)

      # 前処理済データの作成
      lst.train_valid_test <- recipe %>%
        {
          recipe <- (.)

          # train data
          df.train <- recipes::prep(recipe) %>%
            recipes::bake(rsample::training(split))
          x.train <- df.train %>%
            dplyr::select(-salary) %>%
            as.matrix()
          y.train <- df.train$salary

          # for early_stopping
          train_valid_split <- rsample::initial_split(df.train, prop = 4/5, strata = NULL)
          x.train.train <- rsample::training(train_valid_split) %>%
            dplyr::select(-salary) %>%
            as.matrix()
          y.train.train <- rsample::training(train_valid_split)$salary
          x.train.valid <- rsample::testing(train_valid_split) %>%
            dplyr::select(-salary) %>%
            as.matrix()
          y.train.valid <- rsample::testing(train_valid_split)$salary

          # for LightGBM Dataset
          dtrain <- lightgbm::lgb.Dataset(
            data  = x.train.train,
            label = y.train.train
          )
          dvalid <- lightgbm::lgb.Dataset(
            data  = x.train.valid,
            label = y.train.valid,
            reference = dtrain
          )


          # test data
          df.test  <- recipes::prep(recipe) %>%
            recipes::bake(rsample::testing(split))
          x.test <- df.test %>%
            dplyr::select(-salary) %>%
            as.matrix()
          y.test <- df.test$salary


          list(
            # train data
            x.train = x.train,
            y.train = y.train,
            train.dtrain = dtrain,
            train.dvalid = dvalid,

            # test data
            x.test = x.test,
            y.test = y.test
          )
        }

      # 学習
      model.fitted <- lightgbm::lgb.train(

        # 学習パラメータの指定
        params = list(
          boosting_type = "gbdt",
          objective = "fair",
          metric = "fair",

          # user defined
          max_depth = hyper_params$max_depth,
          num_leaves = hyper_params$num_leaves,
          bagging_freq = hyper_params$bagging_freq,
          bagging_fraction = hyper_params$bagging_fraction,
          feature_fraction = hyper_params$feature_fraction,
          min_data_in_leaf = min_data_in_leaf,
          lambda_l1 = hyper_params$lambda_l1,
          lambda_l2 = hyper_params$lambda_l2,

          seed = 1234
        ),

        # 学習＆検証データ
        data   = lst.train_valid_test$train.dtrain,
        valids = list(valid = lst.train_valid_test$train.dvalid),

        # 木の数など
        learning_rate = 0.1,
        nrounds = 20000,
        early_stopping_rounds = 200,
        verbose = 1,

        # カテゴリデータの指定
        categorical_feature = c("position", "area", "sex", "partner", "education")
      )

      # MAE の算出
      train_mae <- tibble::tibble(
        actual = lst.train_valid_test$y.train,
        pred   = predict(model.fitted, lst.train_valid_test$x.train)
      ) %>%
        dplyr::mutate_all(function(x) { exp(x) - 1 }) %>%
        yardstick::mae(truth = actual, estimate = pred) %>%
        .$.estimate
      test_mae <- tibble::tibble(
        actual = lst.train_valid_test$y.test,
        pred    = predict(model.fitted, lst.train_valid_test$x.test)
      ) %>%
        dplyr::mutate_all(function(x) { exp(x) - 1 }) %>%
        yardstick::mae(truth = actual, estimate = pred) %>%
        .$.estimate

      tibble::tibble(
        train_mae = train_mae,
        test_mae  = test_mae
      )
    }, recipe = recipe, hyper_params = hyper_params, .options = furrr::future_options(seed = 5963L)) %>%

      # CV 分割全体の平均値を評価スコアとする
      dplyr::summarise_all(mean)
  }) %>%

    # 評価結果とパラメータを結合
    dplyr::bind_cols(df.grid.params, .) %>%

    # 評価スコアの順にソート(昇順)
    dplyr::arrange(test_mae)
})
