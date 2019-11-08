# Test MAE:
# - position_education の追加: 50.00217
# - 入社区分の追加: 49.95175
# - partner_child の追加: 46.49692
# - ダミー変数への変換時にカテゴリを 1 つ除去: 46.48439

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

  recipes::step_mutate(

    # 入社区分
    base_age = dplyr::case_when(
      education == 0 ~ 18,
      education == 1 ~ 20,
      education == 2 ~ 22,
      education == 3 ~ 24,
      education == 4 ~ 27
    ),
    d = age - base_age - service_length,
    join_segment = dplyr::case_when(
      d == 0 ~ "d_0",   # 新卒
      d == 1 ~ "d_1",   # 第二新卒
      T      ~ "others" # 中途その他
    ) %>%
      factor(levels = c(stringr::str_c("d", 0:1, sep = "_"), "others")),

    # position x education
    position_education = dplyr::case_when(
      # position: 0
      (position == 0) & (education %in% 0:1) ~ "position_0_x_education_0_1",
      (position == 0) & (education == 2)     ~ "position_0_x_education_2",
      (position == 0) & (education == 3)     ~ "position_0_x_education_3",
      (position == 0) & (education == 4)     ~ "position_0_x_education_4",

      # position: 1
      (position == 1) & (education %in% 0:1) ~ "position_1_x_education_0_1",
      (position == 1) & (education == 2)     ~ "position_1_x_education_2",
      (position == 1) & (education == 3)     ~ "position_1_x_education_3",
      (position == 1) & (education == 4)     ~ "position_1_x_education_4",

      # position: 2
      (position == 2) & (education %in% 0:1) ~ "position_2_x_education_0_1",
      (position == 2) & (education == 2)     ~ "position_2_x_education_2",
      (position == 2) & (education == 3)     ~ "position_2_x_education_3",
      (position == 2) & (education == 4)     ~ "position_2_x_education_4",

      # position: 3
      (position == 3) & (education %in% 0:1) ~ "position_3_x_education_0_1",
      (position == 3) & (education == 2)     ~ "position_3_x_education_2",
      (position == 3) & (education == 3)     ~ "position_3_x_education_3",
      (position == 3) & (education == 4)     ~ "position_3_x_education_4",

      # position: 4
      (position == 4) & (education %in% 0:1) ~ "position_4_x_education_0_1",
      (position == 4) & (education == 2)     ~ "position_4_x_education_2",
      (position == 4) & (education == 3)     ~ "position_4_x_education_3",
      (position == 4) & (education == 4)     ~ "position_4_x_education_4"
    ) %>%
      factor(),

    # partner & num child
    partner_child = ifelse(partner == 0, "no_partner", stringr::str_c("child", num_child, sep = "_")) %>%
      factor(levels = c(
        "no_partner",
        stringr::str_c("child", 0:9, sep = "_")
      ))
  ) %>%

  recipes::step_rm(
    id,

    # join_segment
    base_age,
    d,

    # position_education
    position,
    education,

    # partner_child
    partner,
    num_child
  ) %>%

  recipes::step_dummy(recipes::all_nominal(), one_hot = F)

#recipes::prep(recipe) %>% recipes::juice() %>% summary()


# Model Definition --------------------------------------------------------

model <- parsnip::linear_reg(
  mode = "regression",
  penalty = parsnip::varying(),
  mixture = parsnip::varying()
) %>%
  parsnip::set_engine(engine = "glmnet")


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::penalty(c(0, 0.5)),
  dials::mixture(),
  levels = 10
)
df.grid.params


future::plan(future::multisession)

system.time(
  df.results <-

    # ハイパーパラメータをモデルに適用
    # merge(df.grid.params, model) %>%
    purrr::pmap(df.grid.params, function(penalty, mixture) {
      parsnip::set_args(
        model,
        penalty = penalty,
        mixture = mixture
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
      penalty,
      mixture,

      train_mae,
      test_mae
    )
)
