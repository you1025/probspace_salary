## TODO
# xxx_xxx_age の追加 diff_xxx_age が効いてた(OK)
# sd の導入
# label_encoding の導入(OK)
# 正則化パラメータをいじってみる(OK)
# 線型モデルによる特徴量(再)
# 残渣の大きいレコードの調査

# mtry: 79, sample_size: 0.83, tree_depth: 9, min_n: 7, loss_reduction: 0.02772407, train_mae: 17.83818, test_mae: 21.5749 - baseline

# 結論: age 系は全部あかんかった orz
# train_mae: 17.71998, test_mae: 21.74042 ↓ education_mean_age
# train_mae: 17.70762, test_mae: 21.81612 ↓ diff_education_mean_age
# train_mae: 17.6326,  test_mae: 21.83547 ↓ ratio_education_mean_age
# train_mae: 17.72607, test_mae: 21.6687  ↓ education_median_age
# train_mae: 17.63312, test_mae: 21.77004 ↓ diff_education_median_age
# train_mae: 17.65228, test_mae: 21.76756 ↓ ratio_education_median_age
# train_mae: 17.73307, test_mae: 21.75151 ↓ education_min_age
# train_mae: 17.6318,  test_mae: 21.64716 ↓ diff_education_min_age
# train_mae: 17.60851, test_mae: 21.66225 ↓ ratio_education_min_age
# train_mae: 17.77166, test_mae: 21.72024 - education_max_age
# train_mae: 17.63512, test_mae: 21.78312 - diff_education_max_age
# train_mae: 17.66061, test_mae: 21.78494 - ratio_education_max_age
# train_mae: 17.68565, test_mae: 21.77254 - position_education_partner_segment_mean_age
# train_mae: 17.59501, test_mae: 21.67878 - diff_position_education_partner_segment_mean_age
# train_mae: 17.69128, test_mae: 21.74884 - ratio_position_education_partner_segment_mean_age
# train_mae: 17.67927, test_mae: 21.82259 - position_education_partner_segment_median_age
# train_mae: 17.69672, test_mae: 21.67613 - diff_position_education_partner_segment_median_age
# train_mae: 17.63661, test_mae: 21.77471 - ratio_position_education_partner_segment_median_age
# train_mae: 17.71193, test_mae: 21.72608 - position_education_partner_segment_min_age
# train_mae: 17.67408, test_mae: 21.66569 - diff_position_education_partner_segment_min_age
# train_mae: 17.68136, test_mae: 21.71696 - ratio_position_education_partner_segment_min_age
# train_mae: 17.77439, test_mae: 21.70059 - position_education_partner_segment_max_age
# train_mae: 17.61859, test_mae: 21.73111 - diff_position_education_partner_segment_max_age
# train_mae: 17.52921, test_mae: 21.76797 - ratio_position_education_partner_segment_max_age
# train_mae: 17.77726, test_mae: 21.81065 - area_partner_child_segment_mean_age
# train_mae: 17.71366, test_mae: 21.73518 - diff_area_partner_child_segment_mean_age
# train_mae: 17.78486, test_mae: 21.77816 - ratio_area_partner_child_segment_mean_age
# train_mae: 17.763,   test_mae: 21.7537  - area_partner_child_segment_median_age
# train_mae: 17.8434,  test_mae: 21.90867 - diff_area_partner_child_segment_median_age
# train_mae: 17.70978, test_mae: 21.81117 - ratio_area_partner_child_segment_median_age
# train_mae: 17.82778, test_mae: 21.79379 - area_partner_child_segment_min_age
# train_mae: 17.78507, test_mae: 21.79958 - diff_area_partner_child_segment_min_age
# train_mae: 17.78507, test_mae: 21.79958 - ratio_area_partner_child_segment_min_age
# train_mae: 17.84389, test_mae: 21.79971 - area_partner_child_segment_max_age
# train_mae: 17.81152, test_mae: 21.82001 - diff_area_partner_child_segment_max_age
# train_mae: 17.70924, test_mae: 21.78088 - ratio_area_partner_child_segment_max_age
# train_mae: 17.82326, test_mae: 21.8332  - flg_newbie_mean_age
# train_mae: 17.78313, test_mae: 21.82652 - diff_flg_newbie_mean_age
# train_mae: 17.76682, test_mae: 21.79176 - ratio_flg_newbie_mean_age
# train_mae: 17.7122,  test_mae: 21.83888 - flg_newbie_median_age
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - diff_flg_newbie_median_age
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - ratio_flg_newbie_median_age
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - flg_newbie_min_age
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - diff_flg_newbie_min_age
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - ratio_flg_newbie_min_age
# train_mae: 17.80487, test_mae: 21.77804 - flg_newbie_max_age
# train_mae: 17.78311, test_mae: 21.84115 - diff_flg_newbie_max_age
# train_mae: 17.76031, test_mae: 21.70937 - ratio_flg_newbie_max_age

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - baseline
# train_mae: 17.75824, test_mae: 21.8055  - position_sample_sd_salary
# train_mae: 17.8324,  test_mae: 21.77198 - position_sample_sd_commute
# train_mae: 17.73462, test_mae: 21.71608 - scaled_position_mean_commute
# train_mae: 17.75094, test_mae: 21.84907 - education_sample_sd_salary
# train_mae: 17.79769, test_mae: 21.78263 - education_sample_sd_commute
# train_mae: 17.83332, test_mae: 21.89197 - scaled_education_mean_commute
# train_mae: .77407,   test_mae: 21.79754 - position_education_partner_segment_sample_sd_salary
# train_mae: 17.6825,  test_mae: 21.78275 - position_education_partner_segment_sample_sd_commute
# train_mae: 17.7508,  test_mae: 21.77105 - scaled_position_education_partner_segment_mean_commute
# train_mae: 17.8074,  test_mae: 21.76605 - area_partner_child_segment_sample_sd_salary
# train_mae: 17.83294, test_mae: 21.82858 - area_partner_child_segment_sample_sd_commute
# train_mae: 17.88175, test_mae: 17.88175 - scaled_area_partner_child_segment_mean_commute

# overtime を試してみる -> NO!
# train_mae: 17.74026, test_mae: 21.73597 - position_mean_overtime
# train_mae: 17.84086, test_mae: 21.76947 - diff_position_mean_overtime
# train_mae: 17.80652, test_mae: 21.6975  - ratio_position_mean_overtime
# train_mae: 17.82981, test_mae: 21.82852 - position_median_overtime
# train_mae: 17.77336, test_mae: 21.81629 - position_min_overtime
# train_mae: 17.83015, test_mae: 21.85365 - position_max_overtime
# train_mae: 17.78025, test_mae: 21.88589 - education_mean_overtime
# train_mae: 17.7585,  test_mae: 21.71865 - diff_education_mean_overtime
# train_mae: 17.78775, test_mae: 21.83261 - ratio_education_mean_overtime
# train_mae: 17.79555, test_mae: 21.83943 - education_median_overtime
# train_mae: 17.83687, test_mae: 21.86004 - education_min_overtime
# train_mae: 17.79952, test_mae: 21.80531 - education_max_overtime

# Label Encoding
# train_mae: 17.83818, test_mae: 21.5749 - baseline
# train_mae: 17.1779,  test_mae: 21.63129 - area
# train_mae: 17.6101,  test_mae: 21.57567 - position_education_partner_segment
# train_mae: 17.7221,  test_mae: 21.95381 - area の削除
# train_mae: 17.87505, test_mae: 21.6763  - area_partner_child_segment

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - xxx
# train_mae: 17.1779,  test_mae: 21.63129 - area
# train_mae: 17.24585, test_mae: 21.64939 - area & mtry: 86
# train_mae: 17.19577, test_mae: 21.65448 - area & mtry: 98
# train_mae: 17.17413, test_mae: 21.61698 - area & mtry: 102
# train_mae: 16.96981, test_mae: 21.64716 - area & position_education_partner_segment, mtry: 102
# train_mae: 16.87722, test_mae: 21.63325 - area & position_education_partner_segment, mtry: 49

# train_mae: 16.95885, test_mae: 21.66377 - 全部のせ, mtry: 38

# etha: 0.1, [112]	train-mae:0.049673+0.000202	test-mae:0.059883+0.000573

# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = xxx, alpha = xxx, train_mae: xxxxxxxx, test_mae: xxxxxxxx
# mtry: 40, sample_size: 0.800 train_mae: 16.88649, test_mae: 20.81290
# mtry: 39, sample_size: 0.800 train_mae: 16.88083, test_mae: 20.84814
# mtry: 39, sample_size: 0.792, train_mae: 16.90961, test_mae: 20.83331
# mtry: 39, sample_size: 0.792, tree_depth:  8, min_n:  6, train_mae: 17.38014, test_mae: 20.82439
# mtry: 39, sample_size: 0.792, tree_depth:  8, min_n:  5, train_mae: 17.38795, test_mae: 20.86615
# mtry: 39, sample_size: 0.792, tree_depth:  8, min_n:  5, los_reduction: 0.02682696, train_mae: 17.34014, test_mae: 20.84913
# mtry: 39, sample_size: 0.792, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, train_mae: 17.92259, test_mae: 20.87873
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, train_mae: 17.96232, test_mae: 20.83379
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 1, alpha = 0, train_mae: 17.91725, test_mae: 20.8309
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 1e-02, alpha = 1e-08, train_mae: 17.07995, test_mae: 20.82513
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 0.031622777, alpha = 1e-08, train_mae: 17.24047, test_mae: 20.81005
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 0.05623413, alpha = 1.778279e-08, train_mae: 17.34943, test_mae: 20.77744
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 0.03981072, alpha = 1.000000e-08, train_mae: 17.30964, test_mae: 20.76359
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 0.03981072, alpha = 1.412538e-08, train_mae: 17.19963, test_mae: 20.77196
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 0.03162278, alpha = 1.584893e-08, train_mae: 17.25670, test_mae: 20.74935
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 0.03162278, alpha = 1.778279e-08, train_mae: 17.25670, test_mae: 20.74935


# [1751]	train-mae:0.048681+0.000111	test-mae:0.059469+0.000438
# learn_rate: 0.01, trees: 1751
# mtry: 39, sample_size: 0.800, tree_depth:  8, min_n:  5, los_reduction: 0.03801894, lambda = 0.03162278, alpha = 1.778279e-08, train_mae: 16.56729, test_mae: 20.52055


library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/XGBoost_FE_5/functions.R", encoding = "utf-8")


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

# # 1751 => trees
# recipe %>%
#   recipes::prep() %>%
#   recipes::juice() %>%
#   add_features_per_category(., .) %>%
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
#       early_stopping_rounds = 100,
# 
#       eta = 0.01,
# 
#       colsample_bytree = 0.9512195,
#       subsample = 0.8,
# 
#       max_depth = 8,
#       min_child_weight = 5,
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
  dials::trees(c(1751, 1751)),

  dials::mtry(c(39, 39)),

  dials::tree_depth(c(8, 8)),
  dials::min_n(c(5, 5)),

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
            # カテゴリ値の処理
            transform_categories()
          df.test <- recipe %>%
            recipes::prep() %>%
            recipes::bake(rsample::assessment(df.split)) %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(df.train.baked) %>%
            # カテゴリ値の処理
            transform_categories()

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

        lambda,
        alpha,

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
      # カテゴリ値の処理
      transform_categories()

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
    add_features_per_category(., .) %>%
    # カテゴリ値の処理
    transform_categories()

  # 学習の実施
  model.fitted <- parsnip::set_args(
    model,

    learn_rate = 0.01,
    trees = 1751,

    mtry = 39,
    sample_size = 0.8,

    tree_depth = 8,
    min_n = 5,
    loss_reduction = 0.03801894,

    lambda = 0.03162278,
    alpha = 1.778279e-08
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
      add_features_per_category(lst.results$df.train.baked) %>%
      # カテゴリ値の処理
      transform_categories()

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
