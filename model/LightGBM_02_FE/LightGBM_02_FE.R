# TODO
# - 線型モデルの追加(OK)
# - 特徴量の追加(OK)

# train_mae: 20.38744, test_mae: 24.8537  - baseline

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - カテゴリ変換を recipe から transform_categories に移行
# train_mae: 20.19547, test_mae: 24.97273 - get_dummies を追加したらなぜか・・・

# train_mae: 20.30642, test_mae: 24.11919 + 全変数をダミー化
# train_mae: 20.43077, test_mae: 23.89177 + LabelEncoding: position, sex, education
# train_mae: 20.10673, test_mae: 23.51613 + area_partner_child_segment の追加(ダミー変数)

# train_mae: 19.15837, test_mae: 22.32334 + LinearModel を追加
# train_mae: 19.1177,  test_mae: 22.48789 - LinearModel による差分を追加(後で試す)

# train_mae: 18.25359, test_mae: 20.83715 + position 周りの統計量
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - education 周りの統計量(ダメよ)

# train_mae: 18.5848,  test_mae: 21.50004 - position 周りの削除 & position_education_partner_segment 周りの統計量
# train_mae: 18.50938, test_mae: 21.06838 - position 周りの追加
# train_mae: 18.50985, test_mae: 21.04636 - position_education_partner_segment の削除
# 結論: position_education_partner_segment はあかん

# train_mae: 18.41548, test_mae: 20.82122 + area_partner_child_segment 周りの統計量(微妙)
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - area 周りの統計量(ダメよ)
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - sex 周りの統計量(ダメよ)
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - partner 周りの統計量(ダメよ)
# train_mae: 18.29039, test_mae: 20.79498 + 線形モデルを外してみる 何でやwww

# train_mae: 18.39379, test_mae: 20.78707 + flg_area_partner_child_commute_high 周りの統計量はダメ
# train_mae: 18.63496, test_mae: 20.78477 + flg_area_partner_child_commute_low 周りの統計量もダメ
# train_mae: 18.37403, test_mae: 20.77874 + flg_area_partner_child_commute_extra_low 周りの統計量

# max_depth:  7, num_leaves: 36, min_data_in_leaf: 20, train_mae: 18.37790, test_mae: 20.76375
# max_depth:  7, num_leaves: 36, min_data_in_leaf: 23, train_mae: 18.23202, test_mae: 20.75742
# max_depth:  7, num_leaves: 39, min_data_in_leaf: 23, train_mae: 18.44269, test_mae: 20.72784

# bagging_freq: 4, bagging_fraction: 0.90, train_mae: 18.20304, test_mae: 20.71331
# bagging_freq: 4, bagging_fraction: 0.95, train_mae: 18.25866, test_mae: 20.70830

# lambda_l1: 0.400, lambda_l2: 0.900, train_mae: 18.19040, test_mae: 20.58790
# lambda_l1: 0.450, lambda_l2: 0.900, train_mae: 18.18846, test_mae: 20.56552

# learning_rate: 0.010, train_mae: 18.06226, test_mae: 20.43785


# strata を指定するの忘れてた orz
# train_mae: 17.87875, test_mae: 20.47616 - strata を指定
# max_depth:  8, num_leaves: 39, min_data_in_leaf: 23, train_mae: 17.88695, test_mae: 20.44510
# max_depth: 10, num_leaves: 39, min_data_in_leaf: 23, train_mae: 17.88522, test_mae: 20.43538
# max_depth: 14, num_leaves: 39, min_data_in_leaf: 23, train_mae: 17.87060, test_mae: 20.42249

# learning_rate: 0.100, train_mae: 18.18889, test_mae: 20.56572 - いろいろとやってみた

# train_mae: 18.08943, test_mae: 20.63689 - 線型モデルを試してみた ダメやw
# learning_rate: 0.010, train_mae: 18.11365, test_mae: 20.46561
# learning_rate: xxxxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# learning_rate: xxxxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx








library(tidyverse)
library(tidymodels)
library(furrr)
library(lightgbm)

source("model/LightGBM_02_FE/functions.R", encoding = "utf-8")

# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- create_cv(df.train_data, v = 7)

# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  recipes::step_mutate(

    # area x partner x child
    area_partner_child_segment = dplyr::case_when(
       (area %in% c("東京都", "大阪府")) & (partner == 0) & (num_child == 0) ~ "bigcity_single_nochild",
       (area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child == 0) ~ "bigcity_family_nochild",
       (area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child >  0) ~ "bigcity_family_child",
      !(area %in% c("東京都", "大阪府")) & (partner == 0) & (num_child == 0) ~ "country_single_nochild",
      !(area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child == 0) ~ "country_family_nochild",
      !(area %in% c("東京都", "大阪府")) & (partner == 1) & (num_child >  0) ~ "country_family_child"
    ) %>%
      factor(),

    flg_area_partner_child_commute_high = ifelse(
      (area_partner_child_segment == "country_single_nochild")   & dplyr::between(commute, 1.5, 2)
      | (area_partner_child_segment == "country_family_child")   & dplyr::between(commute, 2.5, 3)
      | (area_partner_child_segment == "bigcity_single_nochild") & dplyr::between(commute, 2.5, 3)
      | (area_partner_child_segment == "bigcity_family_nochild") & dplyr::between(commute, 3.5, 4)
      | (area_partner_child_segment == "bigcity_family_child")   & dplyr::between(commute, 4.5, 5)
      | (area_partner_child_segment == "country_single_nochild")   & dplyr::between(commute, 1, 1.5)
      | (area_partner_child_segment == "country_family_nochild") & dplyr::between(commute, 1.5, 2)
      | (area_partner_child_segment == "country_family_child")   & dplyr::between(commute, 2, 2.5)
      | (area_partner_child_segment == "bigcity_single_nochild") & dplyr::between(commute, 2, 2.5)
      | (area_partner_child_segment == "bigcity_family_nochild") & dplyr::between(commute, 2.5, 3.5)
      | (area_partner_child_segment == "bigcity_family_child")   & dplyr::between(commute, 3.5, 4.5),
      1,
      0
    ),
    flg_area_partner_child_commute_low = ifelse(
      (area_partner_child_segment == "country_family_child")     & dplyr::between(commute, 1, 1.5)
      | (area_partner_child_segment == "bigcity_single_nochild") & dplyr::between(commute, 0.5, 1)
      | (area_partner_child_segment == "bigcity_family_nochild") & dplyr::between(commute, 1.5, 2)
      | (area_partner_child_segment == "bigcity_family_child")   & dplyr::between(commute, 2, 2.5),
      1,
      0
    ),
    flg_area_partner_child_commute_extra_low = ifelse(
      (area_partner_child_segment == "country_single_nochild")   & dplyr::between(commute, 0, 0.5)
      | (area_partner_child_segment == "country_family_nochild") & dplyr::between(commute, 0, 0.5)
      | (area_partner_child_segment == "country_family_child")   & dplyr::between(commute, 0.5, 1)
      | (area_partner_child_segment == "bigcity_single_nochild") & dplyr::between(commute, 0, 0.5)
      | (area_partner_child_segment == "bigcity_family_nochild") & dplyr::between(commute, 1, 1.5)
      | (area_partner_child_segment == "bigcity_family_child")   & dplyr::between(commute, 1.5, 2),
      1,
      0
    ),
  ) %>%

  # 対数変換: salary
  recipes::step_log(salary, offset = 1) %>%

  recipes::step_rm(
    id,

    # # 新卒フラグ(flg_newbie)関連
    # base_age,
    # working_years
  )

recipes::prep(recipe) %>% recipes::juice()


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- tibble(
  # max_depth = 11,
  # num_leaves = 36,
  # min_data_in_leaf = 25,

  feature_fraction = 0.5,
  bagging_freq = 2,
  bagging_fraction = 0.925,

lambda_l1 = 0.65,
lambda_l2 = 0.95
) %>%
  tidyr::crossing(
    max_depth = seq(10, 12, 1),
    num_leaves = seq(35, 37, 1),
    min_data_in_leaf = seq(24, 26, 1)
  )
df.grid.params


# Parametr Fitting --------------------------------------------------------

# 並列処理
future::plan(future::multisession(workers = 8))

system.time({

  df.results <- purrr::pmap_dfr(df.grid.params, function(max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_freq, bagging_fraction, lambda_l1, lambda_l2) {

    hyper_params <- list(
      max_depth = max_depth,
      num_leaves = num_leaves,
      min_data_in_leaf = min_data_in_leaf,
      feature_fraction = feature_fraction,
      bagging_freq = bagging_freq,
      bagging_fraction = bagging_fraction,

      lambda_l1 = lambda_l1,
      lambda_l2 = lambda_l2
    )

    furrr::future_map_dfr(df.cv$splits, function(split, recipe, hyper_params) {

      # 前処理済データの作成
      lst.train_valid_test <- recipe %>%
        {
          recipe <- (.)

          # train data
          df.train.baked <- recipes::prep(recipe) %>%
            recipes::bake(rsample::training(split))
          df.train.no_dummied <- df.train.baked %>%
            # # 線形モデルによる予測値を特徴量に追加
            # add_linear_model_predictions(., .) %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(., .)
          df.train <- df.train.no_dummied %>%
            # カテゴリ値の処理
            transform_categories()
          x.train <- df.train %>%
            dplyr::select(-salary) %>%
            as.matrix()
          y.train <- df.train$salary

          # for early_stopping
          train_valid_split <- rsample::initial_split(df.train.no_dummied, prop = 4/5, strata = "area_partner_child_segment")
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
          df.test  <- recipes::prep(recipe) %>%
            recipes::bake(rsample::testing(split)) %>%
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

            # MAE 算出用: train
            x.train = x.train,
            y.train = y.train,

            ## MAE 算出用: test
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
          min_data_in_leaf = min_data_in_leaf,
          feature_fraction = hyper_params$feature_fraction,
          bagging_freq = hyper_params$bagging_freq,
          bagging_fraction = hyper_params$bagging_fraction,
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
        verbose = -1,

        # カテゴリデータの指定
        categorical_feature = c("position", "sex", "education")
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
        pred   = predict(model.fitted, lst.train_valid_test$x.test)
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
