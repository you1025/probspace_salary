## TODO
# - validation の変更(OK)
# - 線型モデルの追加
# - sd の導入
# - 残渣の大きいレコードの調査

# BaseLine
# learn_rate: 0.3, trees: 300
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 0.03162278, alpha = 1.778279e-08, train_mae: 15.25398, test_mae: 21.98041

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - xxx

# train_mae: 15.27372, test_mae: 21.89471 + cv の変更
# train_mae: 15.2089,  test_mae: 21.82498 + 対数変換のオフセットを 0 に変更
# train_mae: 15.23484, test_mae: 21.75776 - 線型モデルの導入

# [155]	train-mae:0.049593+0.000168	test-mae:0.060148+0.000643
# [1569]	train-mae:0.050180+0.000164	test-mae:0.059645+0.000412

# train_mae: 17.13025, test_mae: 20.45694 - チューニング後(最終)





library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/XGBoost_FE_6/functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- create_cv(df.train_data)


# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  # clipping
  recipes::step_mutate(
    overtime = ifelse(overtime < 25.301, overtime, 25.301)
  ) %>%

  recipes::step_mutate(

    # 平社員フラグ
    flg_staff = (position == 0),

    # 新卒フラグ
    base_age = dplyr::case_when(
      education == 0 ~ 18,
      education == 1 ~ 20,
      education == 2 ~ 22,
      education == 3 ~ 24,
      education == 4 ~ 27
    ),
    working_years = age - base_age,
    ratio_othercompany_years_per_service_length = ifelse(service_length == 0, 0, (working_years - service_length) / service_length),

    # 生え抜きフラグ
    flg_newbie = (working_years - service_length <= 1),

    # # position x education x partner
    # position_education_partner_segment = stringr::str_c(
    #   "position", position,
    #   "education", education,
    #   "partner", partner,
    #   sep = "_"
    # ) %>%
    #   factor(),

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

  recipes::step_rm(
    id,

    # 新卒フラグ(flg_newbie)関連
    base_age,
    working_years
  ) %>%

  recipes::step_log(salary)

recipes::prep(recipe) %>% recipes::juice() %>% 
  summary()



# Detect [trees] ---------------------------------------------------------

# # 1569 => trees
# recipe %>%
#   recipes::prep() %>%
#   recipes::juice() %>%
#   add_features_per_category(., .) %>%
#   add_linear_model_predictions(., .) %>%
#   transform_categories() %>%
# 
#   {
#     df.data <- (.)
#     x <- df.data %>%
#       get_dummies() %>%
#       dplyr::select(-salary) %>%
#       as.matrix()
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
#       nrounds = 3000,
#       early_stopping_rounds = 50,
# 
#       eta = 0.01,
# 
#       colsample_bytree = 0.9512195,
#       subsample = 0.8,
# 
#       max_depth = 8,
#       min_child_weight = 13,
#       gamma = 0.03801894,
# 
#       lambda = 0.03162278,
#       alpha = 1.778279e-08
#     )
#   }


# Model Definition --------------------------------------------------------

model <- parsnip::boost_tree(
  mode = "regression",
  learn_rate = parsnip::varying(),
  trees = parsnip::varying(),

  mtry = parsnip::varying(),           # colsample_bytree
  sample_size = parsnip::varying(),    # subsample

  tree_depth = parsnip::varying(),     # max_depth
  min_n = parsnip::varying(),          # min_child_weight
  loss_reduction = parsnip::varying()  # gamma
) %>%
  parsnip::set_engine(engine = "xgboost")


# Hyper Parameter ---------------------------------------------------------

# チューニング
# 1. learn_rate を 0.1 にして trees を決める
# 2. mtry, sample_size (サンプリング) を決める
# 3. tree_depth, min_n, loss_reduction (決定的) を決める
# 4. lambda, alpha を決める
# 5. learning_rate を 0.01 にして trees を決める
df.grid.params <- dials::grid_regular(
  dials::learn_rate(c(-2, -2)), # 10^(-0.5228787) = 0.3
  dials::trees(c(1569, 1569)),

  dials::mtry(c(39, 39)),

  dials::tree_depth(c(8, 8)),
  dials::min_n(c(13, 13)),

  dials::loss_reduction(c(-1.42, -1.42)),

  levels = 1
) %>%
  tidyr::crossing(sample_size = seq(0.80, 0.80, length.out = 1)) %>%
  tidyr::crossing(
    lambda = 10^seq(-1.50, -1.50, length.out = 1),
    alpha  = 10^seq(-7.75, -7.75, length.out = 1)
  )
df.grid.params


# Parametr Fitting --------------------------------------------------------

# 並列処理
future::plan(future::multisession(workers = 8))
#future::plan(future::sequential)

system.time(
  {
    # sequential 用
    # multisessions 用は future_map の .option 引数で指定
    set.seed(5963)

    df.results <-

      # ハイパーパラメータをモデルに適用
      purrr::pmap(df.grid.params, function(trees, learn_rate, tree_depth, min_n, loss_reduction, mtry, sample_size, lambda, alpha) {
        parsnip::set_args(
          model,

          trees = trees,
          learn_rate = learn_rate,

          tree_depth = tree_depth,
          min_n = min_n,
          loss_reduction = loss_reduction,
          sample_size = sample_size,
          mtry = mtry,

          lambda = lambda,
          alpha = alpha
        )
      }) %>%

      # ハイパーパラメータの組み合わせごとにループ
#      furrr::future_map(function(model.applied) {
      purrr::map(function(model.applied) {

        # クロスバリデーションの分割ごとにループ
#        purrr::map(df.cv$splits, model = model.applied, function(df.split, model) {
        furrr::future_map(df.cv$splits, model = model.applied, function(df.split, model) {

          # 前処理済データの作成
          df.train.no_dummy <- recipes::prep(recipe) %>%
            recipes::bake(rsample::analysis(df.split)) %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(., .) %>%
            # 線形モデルによる予測値を特徴量に追加
            add_linear_model_predictions(., .)
          df.train <- df.train.no_dummy %>%
            # カテゴリ値の処理
            transform_categories()
          df.test <- recipes::prep(recipe) %>%
            recipes::bake(rsample::assessment(df.split)) %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(df.train.no_dummy) %>%
            # 線形モデルによる予測値を特徴量に追加
            add_linear_model_predictions(df.train.no_dummy) %>%
            # カテゴリ値の処理
            transform_categories()

          model %>%

            # モデルの学習
            {
              model <- (.)
              parsnip::fit(
                model,
                salary ~ .,
                df.train
              )
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
                metrics(
                  truth    = salary    %>% exp(),
                  estimate = predicted %>% exp()
                ) %>%
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
                metrics(
                  truth    = salary    %>% exp(),
                  estimate = predicted %>% exp()
                ) %>%
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
        }, .options = furrr::future_options(seed = 5963L)) %>%

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
        learn_rate,
        trees,

        mtry,
        sample_size,
        tree_depth,
        min_n,
        loss_reduction,

        lambda,
        alpha,

        train_mae,
        test_mae
      )
  }
)
# xxx

# Importances -------------------------------------------------------------

fit <- model %>%

  {
    model <- (.)

    # 前処理済データの作成
    df.train <- recipe %>%
      recipes::prep() %>%
      recipes::juice() %>%
      add_features_per_category(., .) %>%
      # 線形モデルによる予測値を特徴量に追加
      add_linear_model_predictions(., .) %>%
      # カテゴリ値の処理
      transform_categories()

    # 学習の実施
    parsnip::set_args(
      model,

      trees = 1569,
      learn_rate = 0.01,
      tree_depth = 8,
      min_n = 13,
      loss_reduction = 0.03801894,
      mtry = 39,
      sample_size = 0.8,

      lambda = 0.03162278,
      alpha = 1.778279e-08
    ) %>%
      parsnip::fit(salary ~ ., df.train)
  }

fit$fit %>%
  xgboost::xgb.importance(model = .) %>%
  xgboost::xgb.ggplot.importance()


# Predict by Test Data ----------------------------------------------------

system.time(
  {
    set.seed(1025)
    future::plan(future::multisession(workers = 8))

    # モデル定義
    model <- parsnip::boost_tree(
      mode = "regression",

      learn_rate = 0.01,
      trees = 1569,
      mtry = 39,
      sample_size = 0.8,
      loss_reduction = 0.03801894,
      
      tree_depth = parsnip::varying(),
      min_n = parsnip::varying()
    ) %>%
      parsnip::set_engine(engine = "xgboost") %>%
      parsnip::set_args(
        lambda = 0.03162278,
        alpha = 1.778279e-08
      )

    # パラメータ毎に作成する seed の数
    # 全部で パラメータ種類 x num.seed 個のモデルが作成される
    num.seed = 10

    # ハイパーパラメータを複数指定
    # (tree_depth, min_n) = (8, 13), (7, 13), (6, 17)
    tibble(
      tree_depth = c(8, 7, 6),
      min_n = c(13, 13, 17)
    ) %>%

      # seed average
      tidyr::crossing(seed = 1:num.seed) %>%
      dplyr::mutate(seed = sample(1:10000, 2 * num.seed, replace = F)) %>%

      # ハイパラ x seed の分だけモデル作成＆予測
      furrr::future_pmap_dfr(function(tree_depth, min_n, seed) {

        # 前処理済データの作成
        df.train.no_dummy <- recipes::prep(recipe) %>%
          recipes::bake(df.train_data) %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(., .) %>%
          # 線形モデルによる予測値を特徴量に追加
          add_linear_model_predictions(., .)
        df.train <- df.train.no_dummy %>%
          # カテゴリ値の処理
          transform_categories()
        df.test <- recipes::prep(recipe) %>%
          recipes::bake(load_test_data("data/input/test_data.csv")) %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(df.train.no_dummy) %>%
          # 線形モデルによる予測値を特徴量に追加
          add_linear_model_predictions(df.train.no_dummy) %>%
          # カテゴリ値の処理
          transform_categories()

        model.fitted <- parsnip::set_args(
          model,
          tree_depth = tree_depth,
          min_n      = min_n,

          seed = seed
        ) %>%
          parsnip::fit(salary ~ ., df.train)

        # 予測結果データセット
        tibble(
          id = 0:(nrow(df.test)-1),
          y = predict(model.fitted, df.test, type = "numeric")[[1]] %>% exp()
        )
      }) %>%

      # 単一モデル内での Blending
      dplyr::group_by(id) %>%
      dplyr::summarise(y = mean(y)) %>%

      # ファイルに出力
      {
        df.submit <- (.)
        
        # ファイル名
        filename <- stringr::str_c(
          "XGBoost_add_linear_prediction",
          lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
          sep = "_"
        ) %>%
          stringr::str_c("csv", sep = ".")
        
        # 出力ファイルパス
        filepath <- stringr::str_c("data/output", filename, sep = "/")
        
        # 書き出し
        readr::write_csv(df.submit, filepath, col_names = T)
      }
  }
)

