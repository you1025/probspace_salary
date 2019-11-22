# TODO
# 先に fit を作成せないかんのめんどいので固定値にしたい(OK)
# 変数の絞りをもっと厳しくしてみる(OK)
# 効くとわかってる変数は手動で入れ込んだほうが良いかも(OK)
# flg_area_partner_child_commute_xxx で統計量を作る(OK)
# min/max -(/) xxx 系の変数を除去(avg -(/) xxx と相関が出てしまう)(OK)
# area_partner_child_segment 単位で集計値(avg/min/max)を算出(OK) smoothing あり
# label_encoding の導入
# 正則化パラメータをいじってみる
# 残渣の大きいレコードの調査

# [1923]	train-mae:0.049679+0.000110	test-mae:0.060624+0.000293 - baseline

# [126]	train-mae:0.046563+0.000503	test-mae:0.061843+0.001131 - 変数大量にぶっこんだ版 Overfitting している模様
# learn_rate: 0.1, trees: 126, mtry: xxx, sample_size: 0.8285714, tree_depth:  9, min_n:  7, loss_reduction: 0.02772407, train_mae: xxxxxxxx, test_mae: xxxxxxxx -
# learn_rate: 0.1, trees: 126, mtry: 220, sample_size: 0.8285714, tree_depth:  9, min_n:  7, loss_reduction: 0.02772407, train_mae: 17.77122, test_mae: 22.53801 -
# learn_rate: 0.1, trees: 126, mtry: 320, sample_size: 0.8285714, tree_depth:  9, min_n:  7, loss_reduction: 0.02772407, train_mae: 17.54650, test_mae: 22.47717 ↑

# 変数絞りました 124 個くらい
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.90, tree_depth:  9, min_n:  7, loss_reduction: 0.02772407, train_mae: 15.79716, test_mae: 21.55554 -
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth:  9, min_n:  7, loss_reduction: 0.02772407, train_mae: 15.83885, test_mae: 21.52104 ↑
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth: 10, min_n:  6, loss_reduction: 0.02772407, train_mae: 15.19507, test_mae: 21.51844 ↑
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth: 11, min_n:  5, loss_reduction: 0.02772407, train_mae: 14.69526, test_mae: 21.52339 ↓ orz
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth: 11, min_n:  5, loss_reduction: 0.02865120, train_mae: 14.85327, test_mae: 21.50126 ↑
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth: 11, min_n:  5, loss_reduction: 1e-02,      train_mae: 11.407162, test_mae: 21.69397 ↓
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth: 11, min_n:  5, loss_reduction: xxxxxxxxxx,  train_mae: xxxxxxxx, test_mae: xxxxxxxx -
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth: 11, min_n:  5, loss_reduction: 0.051794747, train_mae: 16.910708, test_mae: 21.50559 ↑
# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth: 11, min_n:  5, loss_reduction: 0.03727594,  train_mae: 15.74248, test_mae: 21.51042 ↓ orz

# learn_rate: 0.1, trees: 126, mtry: 75, sample_size: 0.95, tree_depth: 11, min_n:  5, loss_reduction: 0.051794747, train_mae: 16.910708, test_mae: 21.50559(☆)

# [2199]	train-mae:0.047752+0.000064	test-mae:0.060572+0.000266
# rearn_rate: 0.01, trees: 2199, train_mae: 16.31401, test_mae: 21.19913

# 変数を 41 個に絞った

# learn_rate: 0.1, trees: 491, mtry: 74, sample_size: 0.90, tree_depth: 11, min_n:  5, loss_reduction: 0.051794747, train_mae: 16.58762, test_mae: 21.55902
# learn_rate: 0.1, trees: 491, mtry: 74, sample_size: 0.925, tree_depth: 11, min_n:  5, loss_reduction: 0.051794747, train_mae: 16.71961, test_mae: 21.52937


# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 13, min_n:  8, loss_reduction: xxxxxxxxxxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# learn_rate: 0.1, trees: 491, mtry: 25, sample_size: 0.90,  tree_depth: 11, min_n:  5, loss_reduction: 0.051794747, train_mae: 16.75880, test_mae: 21.58583
# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 11, min_n:  5, loss_reduction: 0.051794747, train_mae: 16.82377, test_mae: 21.57927
# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 12, min_n:  5, loss_reduction: 0.051794747, train_mae: 16.67495, test_mae: 21.57614
# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 13, min_n:  6, loss_reduction: 0.051794747, train_mae: 16.50594, test_mae: 21.58472
# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 13, min_n:  8, loss_reduction: 0.051794747, train_mae: 16.55356, test_mae: 21.63484
# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 13, min_n:  8, loss_reduction: 0.1,         train_mae: 18.892873, test_mae: 21.73781
# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 13, min_n:  8, loss_reduction: 0.06105402,  train_mae: 17.24705, test_mae: 21.61915
# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 13, min_n:  8, loss_reduction: 0.05966426,  train_mae: 17.20735, test_mae: 21.56691

## 新しく
# learn_rate: 0.1, trees: 491, mtry: 24, sample_size: 0.925, tree_depth: 13, min_n:  8, loss_reduction: 0.05966426,  train_mae: 21.48122, test_mae: 24.69202 - BL
# learn_rate: 0.1, trees: 491, mtry: 54, sample_size: 0.900, tree_depth: xx, min_n:  x, loss_reduction: 0.05966426,  train_mae: xxxxxxxx, test_mae: xxxxxxxx
# learn_rate: 0.1, trees: 491, mtry: 52, sample_size: 0.90,  tree_depth: 13, min_n:  8, loss_reduction: 0.05966426,  train_mae: 21.39907, test_mae: 24.65139
# learn_rate: 0.1, trees: 491, mtry: 54, sample_size: 0.900, tree_depth: 13, min_n:  8, loss_reduction: 0.05966426,  train_mae: 21.35863, test_mae: 24.63219
# learn_rate: 0.1, trees: 491, mtry: 54, sample_size: 0.900, tree_depth: 14, min_n:  9, loss_reduction: 0.05966426,  train_mae: 21.30941, test_mae: 24.63951

## このままではダメや

# 25.84766, 28.52595




library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/XGBoost_FE_3/functions.R", encoding = "utf-8")


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

    # position x education x partner
    position_education_partner_segment = stringr::str_c(
      "position", position,
      "education", education,
      "partner", partner,
      sep = "_"
    ) %>%
      factor(),

    # position x education
    position_education_segment = stringr::str_c("position", position, "education", education, sep = "_") %>%
      factor(),

    # 平社員フラグ
    flg_staff = ifelse(position == 0, 1, 0),

    # 就業期間など
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
    flg_newbie = ifelse(working_years - service_length <= 1, 1, 0),


    # 大都市フラグ
    flg_bigcity = ifelse(area %in% c("東京都", "大阪府"), 1, 0),
    # 沖縄フラグ
    flg_okinawa = ifelse(area == "沖縄県", 1, 0),
    # 神奈川フラグ
    flg_kanagawa = ifelse(area == "神奈川県", 1, 0),


    # セグメントごとの salary 熱量
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

    position,
    education,
    partner,

    # 生え抜きフラグ関連
    base_age
  ) %>%

  recipes::step_log(salary, offset = 1)

recipes::prep(recipe) %>% recipes::juice() %>% 
  summary()



# Hyper Parameter ---------------------------------------------------------

# # 490 => trees
# recipe %>%
#   recipes::prep() %>%
#   recipes::juice() %>%
#   add_features_per_category(., .) %>%
#   dplyr::select(c("salary", important_features())) %>%
#   get_dummies() %>%
# 
#   {
#     df.data <- (.)
#     x <- df.data %>%
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
#       eta = 0.1,
# 
#       colsample_bytree = 1.0,
#       subsample = 1.0,
# 
#       max_depth = 20,
#       min_child_weight = 3,
# 
#       gamma = 0.00,
# 
#       lambda = 1,
#       alpha = 0.1
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
  parsnip::set_engine(engine = "xgboost") %>%
  parsnip::set_args(
    lambda = 1,
    alpha = 0.1
  )


# チューニング
# 1. learn_rate を 0.1 にして trees を決める
# 2. mtry, sample_size (サンプリング) を決める
# 3. tree_depth, min_n, loss_reduction (決定的) を決める
# 4. learning_rate を 0.01 にして trees を決める
df.grid.params <- dials::grid_regular(
  dials::learn_rate(c(-1, -1)),
  dials::trees(c(270, 270)),

  dials::mtry(c(27, 27)),

  dials::tree_depth(c(14, 14)),
  dials::min_n(c(8, 8)),

  dials::loss_reduction(c(-2, -2)),

  levels = 1
) %>%
  tidyr::crossing(sample_size = seq(0.9, 0.9, length.out = 1))
df.grid.params


# Parametr Fitting --------------------------------------------------------

# 並列処理
furrr::future_options(seed = F)
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
          df.train.baked <- recipe %>%
            recipes::prep() %>%
            recipes::bake(rsample::analysis(df.split))
          df.train <- df.train.baked %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(., .) %>%
            # 重要な変数のみに絞る
            dplyr::select(c("salary", important_features())) %>%
            # ダミー変数化
            get_dummies()
          df.test <- recipe %>%
            recipes::prep() %>%
            recipes::bake(rsample::assessment(df.split)) %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(df.train.baked) %>%
            # 重要な変数のみに絞る
            dplyr::select(c("salary", important_features())) %>%
            # ダミー変数化
            get_dummies()

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
      add_features_per_category(., .) %>%
      get_dummies()

    # 学習の実施
    set.seed(5963)
    parsnip::set_args(
      model,

      learn_rate = 0.1,
      trees = 126,

      mtry = 24,
      sample_size = 0.935,

      tree_depth = 13,
      min_n = 8,
      loss_reduction = 0.05966422
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
    # 訓練/検証 データに代表値を付与
    add_features_per_category(., .) %>%
    # ダミー変数化
    get_dummies() %>%
    # 重要な変数のみに絞る
    dplyr::select(c("salary", important_features()))

  # 学習の実施
  model.fitted <- parsnip::set_args(
    model,

    trees = 491,
    learn_rate = 0.1,
    tree_depth = 13,
    min_n = 8,
    loss_reduction = 0.05966422,
    mtry = 24,
    sample_size = 0.935
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
      # 訓練/検証 データに代表値を付与
      add_features_per_category(lst.results$df.train.baked) %>%
      # ダミー変数化
      get_dummies() %>%
      # 重要な変数のみに絞る
      dplyr::select(c(important_features()))

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


