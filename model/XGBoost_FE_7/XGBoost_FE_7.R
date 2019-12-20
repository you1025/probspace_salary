## TODO
# - cv を変更(OK)
# - 集計処理を変更(少数カテゴリで NA が出ないように変更)(OK)
# - 統計量に sd を追加してみる(OK)
# - ハイパラチューニング(OK)
# - lambda, alpha をやるorz
# - 良さげなハイパラ 3 個セットを見繕う
# - 単一モデルでの Blending

# train_mae: 15.77988, test_mae: 21.99415 - original - baseline
# train_mae: 15.84299, test_mae: 21.91327 + CV

# train_mae: 15.84299, test_mae: 21.91567 - 集計処理の変更(最新の方式に合わせる)
# XGBoost/LightGBM の方に最新の集計処理が反映されていないっぽい orz
# そんなに差は無いなっていうか下がっとるw まあいい

# train_mae: 15.85452, test_mae: 21.80042 + sd 入れてみた 上手く行った

# 以下 learning_rate = 0.3
# mtry: 108, sample_size: 0.825, train_mae: 15.79295, test_mae: 21.87191
# mtry: 108, sample_size: 0.850, train_mae: 15.78592, test_mae: 21.84916
# mtry: 106, sample_size: 0.900, train_mae: 15.82551, test_mae: 21.76777
# mtry: 105, sample_size: 0.950, train_mae: 15.95475, test_mae: 21.58795
# tree_depth:  6, min_n:  9, train_mae: 17.21920, test_mae: 21.45607
# tree_depth:  5, min_n: 10, train_mae: 18.05908, test_mae: 21.44333
# tree_depth:  4, min_n: 12, train_mae: 18.96684, test_mae: 21.35076
# tree_depth:  4, min_n: 11, train_mae: 18.94156, test_mae: 21.41534
# loss_reduction: 0.03281341, train_mae: 19.16956, test_mae: 21.40395
# loss_reduction: 0.03300000, train_mae: 19.19842, test_mae: 21.39957
# loss_reduction: 0.03786079, train_mae: 19.35757, test_mae: 21.38406
# sample_size: 0.9371429, train_mae: 19.35143, test_mae: 21.42080


# learning_rate: 0.01, trees: 3000, train_mae: 19.57548, test_mae: 21.00116 んーよくわからん 微妙？w

# Top 3
# tree_depth:  5, min_n: 12, train_mae: 18.96319, test_mae: 20.83862(☆)
# tree_depth:  5, min_n: 11, train_mae: 18.91771, test_mae: 20.91644
# tree_depth:  5, min_n: 10, train_mae: 18.96499, test_mae: 20.93606

# やり直し
# tree_depth:  6, min_n: 13, train_mae: 17.99458, test_mae: 20.70864

# [3795]	train-mae:0.052927+0.000116	test-mae:0.059472+0.000615
# learning_rate: 0.01, train_mae: xxxxxxxx, test_mae: xxxxxxxx

# train_mae: xxxxxxxx, test_mae: xxxxxxxx


# train_mae: xxxxxxxx, test_mae: xxxxxxxx





library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/XGBoost_FE_7/functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- create_cv(df.train_data, v = 7)


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



# Detect [trees] ---------------------------------------------------------

# # 3795 => trees 上限までぶっちぎってもうたw
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
#       nrounds = 5000,
#       early_stopping_rounds = 50,
# 
#       eta = 0.01,
# 
#       colsample_bytree = 0.7191781,
# 
#       max_depth = 6,
#       min_child_weight = 13,
# 
#       gamma = 0.03786082,
#       subsample = 0.9371429,
#
#       lambda = xxx,
#       alpha = xxx
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
# 4. learning_rate を 0.01 にして trees を決める
df.grid.params <- dials::grid_regular(
  dials::learn_rate(c(-1, -1)), # 10^(-0.5228787) = 0.3
#  dials::trees(c(3795, 3795)),
  dials::trees(c(474, 474)),

  dials::mtry(c(105, 105)),

  dials::tree_depth(c(6, 8)),
  dials::min_n(c(13, 17)),

  dials::loss_reduction(c(-1.42181, -1.42181)),

  levels = 3
) %>%
  tidyr::crossing(sample_size = seq(0.9371429, 0.9371429, length.out = 1))
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
          df.train <- recipes::prep(recipe) %>%
            recipes::bake(rsample::analysis(df.split)) %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(., .)
          df.test <- recipes::prep(recipe) %>%
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
        learn_rate,
        trees,

        mtry,
        sample_size,
        tree_depth,
        min_n,
        loss_reduction,

        train_mae,
        test_mae
      )
  }
)
#xxx


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

system.time({
  set.seed(1025)
  future::plan(future::multisession(workers = 8))

  # モデル定義
  model <- parsnip::boost_tree(
    mode = "regression",

    learn_rate = 0.01,
    trees = 3795,
    mtry = 105,
    sample_size = 0.9371429,
    loss_reduction = 0.03786082,

    tree_depth = parsnip::varying(),
    min_n = parsnip::varying()
  ) %>%
    parsnip::set_engine(engine = "xgboost")

  # パラメータ毎に作成する seed の数
  # 全部で パラメータ種類 x num.seed 個のモデルが作成される
  num.seed = 10

  # ハイパーパラメータを複数指定
  # (tree_depth, min_n) = (6, 13), (7, 17), (7, 15)
  tibble(
    tree_depth = c(6, 7, 7),
    min_n = c(13, 17, 15)
  ) %>%

    # seed average
    tidyr::crossing(seed = sample(1:10000, num.seed, replace = F)) %>%

    # ハイパラ x seed の分だけモデル作成＆予測
    furrr::future_pmap_dfr(function(tree_depth, min_n, seed) {

      # 前処理済データの作成
      df.train <- recipes::prep(recipe) %>%
        recipes::bake(df.train_data) %>%
        # 訓練/検証 データに代表値を付与
        add_features_per_category(., .)
      df.test <- recipes::prep(recipe) %>%
        recipes::bake(load_test_data("data/input/test_data.csv")) %>%
        # 訓練/検証 データに代表値を付与
        add_features_per_category(df.train)

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
        y = predict(model.fitted, df.test, type = "numeric")[[1]] %>% exp() %>% { (.) - 1 }
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
        "XGBoost",
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
