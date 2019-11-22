## TODO
# area_partner_child_segment 単位で集計値(avg/min/max)を算出 smoothing あり(OK) めっちゃ効いた！
# xxx_min_xxx の追加(OK)
# xxx_max_xxx の追加(OK)
# xxx_median_xxx の追加(OK)
# xxx_xxx_age の追加
# diff_xxx_age が効いてた
# label_encoding の導入
# 線型モデルによる特徴量(再)
# 正則化パラメータをいじってみる
# 残渣の大きいレコードの調査

# learning_rate: 0.3, [42]	train-mae:0.053166+0.000193	test-mae:0.063390+0.000664

## learning_rate: 0.3, trees: 42
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 18.44916, test_mae: 22.19666 - baseline
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.97631, test_mae: 21.84503 ↑ per area_partner_child_segment
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.79592, test_mae: 21.85773 ↓ min の一律追加
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.87045, test_mae: 21.93582 ↓ position_min_salary
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.90099, test_mae: 21.8161  ↑ position_max_salary
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.91255, test_mae: 21.80878 ↑ position_max_salary のみ
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.91102, test_mae: 21.79101 ↑ position_min_commute
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.95098, test_mae: 21.86759 ↓ position_max_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.99927, test_mae: .82806   ↓ education_min_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.93423, test_mae: 21.74774↑ education_max_salary
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.87984, test_mae: 21.86893 ↓ education_min_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.86315, test_mae: 21.87928 ↓ education_max_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.7964,  test_mae: 21.85948 ↓ position_education_partner_segment_min_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.83837, test_mae: 21.91096 ↓ position_education_partner_segment_max_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.85842, test_mae: 21.87004 ↓ position_education_partner_segment_min_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.81694, test_mae: 21.73724 ↑ position_education_partner_segment_max_commute
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.83941, test_mae: 21.73328 ↑ area_partner_child_segment_min_salary
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.74868, test_mae: 21.76867 ↓ area_partner_child_segment_max_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.92972, test_mae: 21.89171 ↓ area_partner_child_segment_min_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.89522, test_mae: 21.88475 ↓ area_partner_child_segment_max_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.8622,  test_mae: 21.87487 ↓ flg_newbie_min_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.89419, test_mae: 21.83056 ↓ flg_newbie_max_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.85377, test_mae: 21.86261 ↓ flg_newbie_min_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.78669, test_mae: 21.71593 ↑ position_median_salary
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.83818, test_mae: 21.5749(☆) ↑ position_median_commute
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.73017, test_mae: 21.74182 ↓ education_median_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.71978, test_mae: 21.74608 ↓ education_median_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.76244, test_mae: 21.87094 ↓ position_education_partner_segment_median_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.80172, test_mae: 21.78979 ↓ position_education_partner_segment_median_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.72918, test_mae: 21.74382 ↓ area_partner_child_segment_median_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.8175,  test_mae: 21.76563 ↓ area_partner_child_segment_median_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.77345, test_mae: 21.7469  ↓ flg_newbie_median_salary(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.74621, test_mae: 21.79445 ↓ flg_newbie_median_commute(除去)
# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.81609, test_mae: 21.80933 ↓ flg_newbie_max_commute(除去)

# train_mae: 17.79949, test_mae: 21.82255 ↓ position_mean_age
# train_mae: 17.67412, test_mae: 21.79579 ↑ diff_position_mean_age
# train_mae: 17.58086, test_mae: 21.68415 ↑ ratio_position_mean_age
# train_mae: 17.616,   test_mae: 21.75254 - diff_position_mean_age & ratio_position_mean_age(全消去)
# train_mae: 17.73407, test_mae: 21.79484 ↓ position_median_age(除去)
# train_mae: 17.65979, test_mae: 21.89485 ↓ diff_position_median_age(除去)
# train_mae: 17.66616, test_mae: 21.76035 ↑ ratio_position_median_age(除去)
# train_mae: 17.73214, test_mae: 21.6609  ↓ position_min_age(除去)
# train_mae: 17.67375, test_mae: 21.68058 ↓ diff_position_min_age(除去)
# train_mae: 17.66666, test_mae: 21.87425 ↓ ratio_position_min_age(除去)
# train_mae: 17.72489, test_mae: 21.80834 ↓ position_max_age(除去)
# train_mae: 17.64411, test_mae: 21.66079 ↓ diff_position_max_age(除去)
# train_mae: 17.67433, test_mae: 21.6161  ↓ ratio_position_max_age(除去)

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - xxx
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - xxx


## 提出用

# mtry: 105, sample_size: 0.825, tree_depth: 8, min_n: 9, loss_reduction: xxxxxxxxxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# mtry: 100, sample_size: 0.83,  tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.72312, test_mae: 21.62917
# mtry: 105, sample_size: 0.83,  tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.66241, test_mae: 21.63975
# mtry: 105, sample_size: 0.825, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.77167, test_mae: 21.59384
# mtry: 105, sample_size: 0.825, tree_depth: 8, min_n: 7, loss_reduction: 0.02772407, train_mae: 18.20405, test_mae: 21.58031
# mtry: 105, sample_size: 0.825, tree_depth: 8, min_n: 8, loss_reduction: 0.02772407, train_mae: 18.27990, test_mae: 21.51671
# mtry: 105, sample_size: 0.825, tree_depth: 8, min_n: 9, loss_reduction: 0.02772407, train_mae: 18.31459, test_mae: 21.52064
# mtry: 105, sample_size: 0.825, tree_depth: 8, min_n: 9, loss_reduction: 0.02865120, train_mae: 18.41036, test_mae: 21.47870
# [1759]	train-mae:0.050632+0.000148	test-mae:0.059440+0.000372
## learning_rate: 0.01, trees: 1759, train_mae: 17.2382, test_mae: 20.60903


library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/XGBoost_FE_4/functions.R", encoding = "utf-8")


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



# Detect [trees] ---------------------------------------------------------

# # 1759 => trees
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
#       colsample_bytree = 0.7291667,
# 
#       max_depth = 8,
#       min_child_weight = 9,
# 
#       gamma = 0.02865121,
#       subsample = 0.825
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
  dials::learn_rate(c(-2, -2)), # 10^(-0.5228787) = 0.3
  dials::trees(c(1759, 1759)),

  dials::mtry(c(105, 105)),

  dials::tree_depth(c(8, 8)),
  dials::min_n(c(9, 9)),

#  dials::loss_reduction(c(-1.557143, -1.557143)),
  dials::loss_reduction(c(-1.542857, -1.542857)),

  levels = 1
) %>%
  tidyr::crossing(sample_size = seq(0.825, 0.825, length.out = 1))
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

    learn_rate = 0.01,
    trees = 1759,

    mtry = 105,
    sample_size = 0.825,

    tree_depth = 8,
    min_n = 9,
    loss_reduction = 0.02865121
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
