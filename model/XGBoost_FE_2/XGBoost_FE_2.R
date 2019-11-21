# 対数変換を試す
# learning_rate: 0.1, trees: 168, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# learning_rate: 0.1, trees: 168, train_mae: 16.7351,  test_mae: 21.24218 - baseline
# learning_rate: 0.1, trees: 168, train_mae: 18.50391, test_mae: 21.4656  ↓ 対数変換: log(salary+1)

# learn_rate: 0.1, trees: 168, mtry: 80, sample_size: 0.8285714, tree_depth:  7, min_n:  8, loss_reduction: -1.557143, train_mae: 18.51084, test_mae: 21.44901 ↑
# learn_rate: 0.1, trees: 168, mtry: 79, sample_size: 0.8285714, tree_depth:  7, min_n:  8, loss_reduction: -1.557143, train_mae: 18.55571, test_mae: 21.46164 ↓
# learn_rate: 0.1, trees: 168, mtry: 79, sample_size: 0.8285714, tree_depth:  8, min_n:  8, loss_reduction: -1.557143, train_mae: 17.92119, test_mae: 21.42776 ↑
# learn_rate: 0.1, trees: 168, mtry: 79, sample_size: 0.8285714, tree_depth:  9, min_n:  8, loss_reduction: -1.557143, train_mae: 17.43819, test_mae: 21.41316 ↑
# learn_rate: 0.1, trees: 168, mtry: 79, sample_size: 0.8285714, tree_depth:  9, min_n:  7, loss_reduction: -1.557143, train_mae: 17.33079, test_mae: 21.36009 ↑
# learn_rate: 0.1, trees: 168, mtry: 79, sample_size: 0.8285714, tree_depth:  9, min_n:  7, loss_reduction: 0.02511886, train_mae: 17.15750, test_mae: 21.40089 ↓
# learn_rate: 0.1, trees: 168, mtry: 79, sample_size: 0.8285714, tree_depth:  9, min_n:  7, loss_reduction: 0.02772407, train_mae: 17.33079, test_mae: 21.36009(☆)

# [1923]	train-mae:0.049679+0.000110	test-mae:0.060624+0.000293


library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/XGBoost_FE_1/functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- rsample::vfold_cv(df.train_data, v = 5)


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


    # # position x education
    # position_education_segment = stringr::str_c("position", position, "education", education, sep = "_") %>%
    #   factor(),
    # position x education x partner
    position_education_partner_segment = stringr::str_c(
      "position", position,
      "education", education,
      "partner", partner,
      sep = "_"
    ) %>%
      factor(),

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

  recipes::step_log(salary, offset = 1)

recipes::prep(recipe) %>% recipes::juice() %>% 
  summary()



# Hyper Parameter ---------------------------------------------------------

# # 1923 => trees
# recipe %>%
#   recipes::prep() %>%
#   recipes::juice() %>%
#   add_features_per_category(., .) %>%
# #  get_dummies()
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
#       early_stopping_rounds = 100,
# 
#       eta = 0.01,
# 
#       colsample_bytree = 0.5766423,
# 
#       max_depth = 9,
#       min_child_weight = 7,
# 
#       gamma = 0.02772407,
#       subsample = 0.8285714
#     )
#   }


# Model Definition --------------------------------------------------------

model <- parsnip::boost_tree(
  mode = "regression",
  trees = parsnip::varying(),
  learn_rate = parsnip::varying(),

  tree_depth = parsnip::varying(),     # max_depth
  mtry = parsnip::varying(),           # colsample_bytree
  min_n = parsnip::varying(),          # min_child_weight
  loss_reduction = parsnip::varying(), # gamma
  sample_size = parsnip::varying()     # subsample
) %>%
  parsnip::set_engine(engine = "xgboost")


# チューニング
# 1. learn_rate を 0.1 にして trees を決める
# 2. mtry, sample_size (サンプリング) を決める
# 3. tree_depth, min_n, loss_reduction (決定的) を決める
# 4. learning_rate を 0.01 にして trees を決める
df.grid.params <- dials::grid_regular(
  dials::learn_rate(c(-2, -2)),
  dials::trees(c(1923, 1923)),

  dials::mtry(c(79, 79)),

  dials::tree_depth(c(9, 9)),
  dials::min_n(c(7, 7)),

  dials::loss_reduction(c(-1.557143, -1.557143)),

  levels = 1
) %>%
  tidyr::crossing(sample_size = seq(0.8285714, 0.8285714, length.out = 1))
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
      furrr::future_map(function(model.applied) {

        # クロスバリデーションの分割ごとにループ
        purrr::map(df.cv$splits, model = model.applied, function(df.split, model) {

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
                  truth    = salary    %>% exp() %>% { (.) - 1 },
                  estimate = predicted %>% exp() %>% { (.) - 1 }
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
                  truth    = salary    %>% exp() %>% { (.) - 1 },
                  estimate = predicted %>% exp() %>% { (.) - 1 }
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
        }) %>%

          # CV 分割全体の平均値を評価スコアとする
          purrr::reduce(dplyr::bind_rows) %>%
          dplyr::summarise_all(mean)
      }, .options = furrr::future_options(seed = 5963L)) %>%

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
  }
)


# Importances -------------------------------------------------------------

fit <- model %>%

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

      trees = 1618,
      learn_rate = 0.01,
      tree_depth = 7,
      min_n = 8,
      loss_reduction = 0.02772407,
      mtry = 82,
      sample_size = 0.8285714
    ) %>%
      parsnip::fit(salary ~ ., df.train)
  }

fit$fit %>%
  xgboost::xgb.importance(model = .) %>%
  xgboost::xgb.ggplot.importance()


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
    trees = 1923,
    learn_rate = 0.01,
    tree_depth = 9,
    min_n = 7,
    loss_reduction = 0.02772407,
    mtry = 79,
    sample_size = 0.8285714
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
      y = predict(fit, df.test, type = "numeric")[[1]] %>% exp() %>% { (.) - 1 }
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


