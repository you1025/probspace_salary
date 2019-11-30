# TODO
# ダミー変数で傾きもほげほげ

# penalty: 1.096478, mixture: 0.2444444, train_mae: 34.36765, test_mae: 34.62781 - baseline

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - xxx
# train_mae: 34.30577, test_mae: 34.57043 - commute と area_partner_child_segment との交互作用の追加
# train_mae: 113.8698, test_mae: 113.779  - 対数変換(salary) なんでや・・・


library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/LinearRegression_FE_2/functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

df.cv <- create_cv(df.train_data, v = 5)


# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

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
#    ratio_othercompany_years_per_service_length = ifelse(service_length == 0, 0, (working_years - service_length) / service_length),

    # 生え抜きフラグ
    flg_newbie = (working_years - service_length <= 1),


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

    # low〜high フラグ
    flg_area_partner_child_commute_extra_high = ifelse(
      (area_partner_child_segment == "country_single_nochild")   & dplyr::between(commute, 1.5, 2)
      | (area_partner_child_segment == "country_family_child")   & dplyr::between(commute, 2.5, 3)
      | (area_partner_child_segment == "bigcity_single_nochild") & dplyr::between(commute, 2.5, 3)
      | (area_partner_child_segment == "bigcity_family_nochild") & dplyr::between(commute, 3.5, 4)
      | (area_partner_child_segment == "bigcity_family_child")   & dplyr::between(commute, 4.5, 5),
      1,
      0
    ),
    flg_area_partner_child_commute_high = ifelse(
      (area_partner_child_segment == "country_single_nochild") & dplyr::between(commute, 1, 1.5)
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

    # # partner & num child
    # partner_child = ifelse(partner == 0, "no_partner", stringr::str_c("child", num_child, sep = "_")) %>%
    #   factor(levels = c(
    #     "no_partner",
    #     stringr::str_c("child", 0:9, sep = "_")
    #   ))
  ) %>%

  recipes::step_rm(
    id,

    base_age,
  ) %>%

  # 対数変換: salary
  recipes::step_log(salary)

recipes::prep(recipe) %>% recipes::juice() %>% summary()


# Model Definition --------------------------------------------------------

model <- parsnip::linear_reg(
  mode = "regression",
  penalty = parsnip::varying(),
  mixture = parsnip::varying()
) %>%
  parsnip::set_engine(engine = "glmnet")


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::penalty(c(0.03999992, 0.03999992)),
  dials::mixture(c(0.2444444, 0.2444444)),
  levels = 1
)
df.grid.params


future::plan(future::multisession)

system.time(
  {
    set.seed(5963)

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
      furrr::future_map(function(model.applied) {

        # クロスバリデーションの分割ごとにループ
        purrr::map(df.cv$splits, model = model.applied, function(df.split, model) {

          # 前処理済データの作成
          df.train.baked <- recipe %>%
            recipes::prep() %>%
            recipes::bake(rsample::training(df.split))
          df.train <- df.train.baked %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(., .) %>%
            # カテゴリ値の処理
            transform_categories()
          df.test <- recipe %>%
            recipes::prep() %>%
            recipes::bake(rsample::testing(df.split)) %>%
            # 訓練/検証 データに代表値を付与
            add_features_per_category(df.train.baked) %>%
            # カテゴリ値の処理
            transform_categories()

          model %>%

            # モデルの学習
            {
              model <- (.)
              parsnip::fit(
                model,
                salary ~ .
                  + commute:area_partner_child_segment_bigcity_family_nochild
                  + commute:area_partner_child_segment_bigcity_single_nochild
                  + commute:area_partner_child_segment_country_family_child
                  + commute:area_partner_child_segment_country_family_nochild
                  + commute:area_partner_child_segment_country_single_nochild
                ,
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

              print(lst.predicted$train)

              # train データでモデルを評価
              df.result.train <- df.train %>%
                dplyr::mutate(
                  predicted = lst.predicted$train
                ) %>%
                metrics(
                  # truth    = salary,
                  # estimate = predicted
                  # truth    = salary    %>% exp() %>% { (.) - 1 },
                  # estimate = predicted %>% exp() %>% { (.) - 1 }
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
                  # truth    = salary,
                  # estimate = predicted
                  # truth    = salary    %>% exp() %>% { (.) - 1 },
                  # estimate = predicted %>% exp() %>% { (.) - 1 }
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
  }
)
