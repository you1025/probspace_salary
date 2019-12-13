library(tidyverse)
library(tidymodels)

df.grid.params <- tibble(
  max_depth = c(11, 10, 11),
  num_leaves = c(36, 35, 35),
  min_data_in_leaf = 25,

  feature_fraction = 0.5,
  bagging_freq = 2,
  bagging_fraction = 0.925,

  lambda_l1 = 0.65,
  lambda_l2 = 0.95
) %>%

  # random seed averaging
  tidyr::crossing(
    seed = sample(1:10000, size = 10, replace = F)
  )

# 並列処理
future::plan(future::multisession(workers = 8))

system.time({

  furrr::future_pmap_dfr(df.grid.params, function(max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_freq, bagging_fraction, lambda_l1, lambda_l2, seed, recipe) {

    # 学習パラメータの設定
    model.params <- list(
      boosting_type = "gbdt",
      objective     = "fair",
      metric        = "fair",

      # user defined
      max_depth        = max_depth,
      num_leaves       = num_leaves,
      min_data_in_leaf = min_data_in_leaf,
      feature_fraction = feature_fraction,
      bagging_freq     = bagging_freq,
      bagging_fraction = bagging_fraction,
      lambda_l1        = lambda_l1,
      lambda_l2        = lambda_l2,

      seed = seed
    )

    print(recipe)
    # 前処理済データの作成
    lst.train_valid_test <- recipe %>%
      {
        recipe <- (.)

        # train data
        df.train.baked <- recipes::prep(recipe) %>%
          recipes::bake(df.train_data)
        df.train.no_dummied <- df.train.baked %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(., .)
        df.train <- df.train.no_dummied %>%
          # カテゴリ値の処理
          transform_categories()

        # for early_stopping
        train_valid_split <- rsample::initial_split(df.train.no_dummied, prop = 6/7, strata = "area_partner_child_segment")
        x.train.train <- rsample::training(train_valid_split) %>%
          transform_categories() %>%
          dplyr::select(-salary) %>%
          as.matrix()
        y.train.train <- rsample::training(train_valid_split)$salary
        x.train.valid <- rsample::testing(train_valid_split) %>%
          transform_categories() %>%
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
        df.test.raw <- load_test_data("data/input/test_data.csv")
        df.test  <- recipes::prep(recipe) %>%
          recipes::bake(df.test.raw) %>%
          # # 線形モデルによる予測値を特徴量に追加
          # add_linear_model_predictions(df.train.baked) %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(df.train.baked) %>%
          # カテゴリ値の処理
          transform_categories()
        x.test <- df.test %>%
          dplyr::select(-salary) %>%
          as.matrix()
        y.test <- df.test$salary


        list(
          ## model 学習用
          train.dtrain = dtrain,
          train.dvalid = dvalid,

          ## 提出用
          x.test = x.test,
          y.test = y.test
        )
      }


    # 学習
    model.fitted <- lightgbm::lgb.train(

      # 学習パラメータの指定
      params = model.params,

      # 学習＆検証データ
      data   = lst.train_valid_test$train.dtrain,
      valids = list(valid = lst.train_valid_test$train.dvalid),

      # 木の数など
      learning_rate = 0.01,
      nrounds = 20000,
      early_stopping_rounds = 200,
      verbose = -1,

      # カテゴリデータの指定
      categorical_feature = c("position", "sex", "education")
    )

    # 予測結果
    pred = exp(predict(model.fitted, lst.train_valid_test$x.test)) - 1
    tibble(
      id = 0:(length(pred)-1),
      y  = pred
    )
  }, recipe = recipe) %>%

    # 単一モデル内での Blending
    dplyr::group_by(id) %>%
    dplyr::summarise(y = mean(y)) %>%

    # ファイルに出力
    {
      df.submit <- (.)
      print(df.submit)

      # ファイル名
      filename <- stringr::str_c(
        "LightGBM",
        lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
        sep = "_"
      ) %>%
        stringr::str_c("csv", sep = ".")

      # 出力ファイルパス
      filepath <- stringr::str_c("data/output", filename, sep = "/")

      # 書き出し
      readr::write_csv(df.submit, filepath, col_names = T)
    }
})
