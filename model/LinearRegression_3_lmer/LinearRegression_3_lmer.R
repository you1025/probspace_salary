# TODO
# - 階層モデル(OK)
# - 対数変換(salary)(OK)
# - 通常の重回帰モデル(再)

# train_mae: 52.93582, test_mae: 52.96725 - baseline, commute

# train_mae: 51.5919,  test_mae: 51.61632 ↑ age
# train_mae: 50.18818, test_mae: 50.21498 ↑ num_child
# train_mae: 49.36995, test_mae: 49.39749 ↑ service_length 効くんだ・・・
# train_mae: 49.31704, test_mae: 49.34671 ↑ study_time 微妙
# train_mae: 48.50173, test_mae: 48.52829 ↑ overtime
# train_mae: 47.8669,  test_mae: 47.89947 ↑ flg_staff
# train_mae: 47.31423, test_mae: 47.34495 ↑ working_years
# train_mae: 47.31381, test_mae: 47.35011 - flg_newbie(除去)
# train_mae: 46.99016, test_mae: 47.03395 - flg_area_partner_child_commute_extra_high
# train_mae: 42.25736, test_mae: 42.30346 - flg_area_partner_child_commute_high
# train_mae: 42.02357, test_mae: 42.0725  - flg_area_partner_child_commute_low
# train_mae: 42.04218, test_mae: 42.09317 ↓ flg_area_partner_child_commute_extra_low(除去)
# train_mae: 39.43273, test_mae: 39.47262 - position_median_commute
# train_mae: 39.41806, test_mae: 39.47796 ↓ education_max_salary(除去)
# train_mae: 37.27462, test_mae: 37.4112  - position_education_partner_segment_median_salary
# train_mae: 37.24999, test_mae: 37.38903 - position_education_partner_segment_median_commute
# train_mae: 37.20774, test_mae: 37.3693  - area_median_salary
# train_mae: 37.21002, test_mae: 37.37148 ↓ area_partner_child_segment_mean_salary(除去)
# train_mae: 37.20737, test_mae: 37.36894 - ratio_area_partner_child_segment_mean_commute
# train_mae: 37.20123, test_mae: 37.36924 - ratio_flg_area_partner_child_commute_extra_high_mean_commute
# train_mae: 36.96468, test_mae: 37.14598 - flg_area_partner_child_commute_high_mean_salary
# train_mae: 36.96198, test_mae: 37.14496 - flg_area_partner_child_commute_high_max_salary
# train_mae: 36.97212, test_mae: 37.15819 ↓ flg_area_partner_child_commute_high_mean_commute(除去)
# train_mae: 36.97166, test_mae: 37.15602 - flg_area_partner_child_commute_high_median_commute
# train_mae: 36.97145, test_mae: 37.15599 - flg_area_partner_child_commute_low_median_salary
# train_mae: 36.54267, test_mae: 36.72613 - ratio_flg_area_partner_child_commute_low_mean_commute
# train_mae: 36.58395, test_mae: 36.76924 ↓ flg_area_partner_child_commute_extra_low_max_salary(除去)
# train_mae: 35.17493, test_mae: 35.35229 - ratio_flg_area_partner_child_commute_extra_low_mean_commute
# train_mae: 34.30426, test_mae: 34.48308 - ratio_flg_area_partner_child_commute_extra_low_max_commute
# train_mae: 34.43545, test_mae: 34.686   - 標準化の適用
# train_mae: 34.36171, test_mae: 34.62349 - position_X1
# train_mae: 34.29912, test_mae: 34.54827 - position_X2
# train_mae: 34.30092, test_mae: 34.55266 ↓ position_X3(除去)
# train_mae: 34.30092, test_mae: 34.55265 ↓ position_X4(除去)
# train_mae: 34.30058, test_mae: 34.55216 ↓ area_奈良県(除去)
# train_mae: 34.29757, test_mae: 34.54817 - area_山口県
# train_mae: 34.2978,  test_mae: 34.54885 ↓ area_東京都(除去)
# train_mae: 34.29768, test_mae: 34.55224 ↓ area_鹿児島県(除去)
# train_mae: 34.2976,  test_mae: 34.55094 ↓ area_兵庫県(除去)
# train_mae: 34.22119, test_mae: 34.4791  - area_神奈川県
# train_mae: 34.22184, test_mae: 34.4821  ↓ area_宮城県(除去)
# train_mae: 34.22256, test_mae: 34.47994 ↓ area_茨城県(除去)
# train_mae: 34.2216,  test_mae: 34.48113 ↓ area_岩手県(除去)
# train_mae: 34.22053, test_mae: 34.47945 ↓ area_鳥取県(除去)
# train_mae: 34.22322, test_mae: 34.48056 ↓ area_岡山県(除去)
# train_mae: 34.2206,  test_mae: 34.48061 ↓ area_愛媛県(除外)
# train_mae: 34.22095, test_mae: 34.47931 ↓ area_新潟県(除外)
# train_mae: 34.2214,  test_mae: 34.48105 ↓ area_島根県(除外)
# train_mae: 34.22065, test_mae: 34.48158 ↓area_和歌山県(除外)
# train_mae: 34.22033, test_mae: 34.47955 - area_熊本県
# train_mae: 34.22012, test_mae: 34.47894 - area_埼玉県
# train_mae: 34.22044, test_mae: 34.47974 ↓ area_大阪府(除外)
# train_mae: 34.21956, test_mae: 34.48122 ↓ area_大分県(除外)
# train_mae: 34.21935, test_mae: 34.4785  - area_徳島県
# train_mae: 34.10933, test_mae: 34.35978 - area_沖縄県
# train_mae: 34.10237, test_mae: 34.35481 - area_三重県
# train_mae: 34.09752, test_mae: 34.35399 - area_秋田県
# train_mae: 34.0762,  test_mae: 34.33217 - area_福岡県
# train_mae: 34.0533,  test_mae: 34.31293 - sex_X2
# train_mae: 34.05261, test_mae: 34.314   ↓ partner_X1(除去)
# train_mae: 34.04005, test_mae: 34.30152 - education_X1
# train_mae: 34.03361, test_mae: 34.28795 - education_X2
# train_mae: 33.91266, test_mae: 34.16474 - education_X3
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - education_X4(除去 おそらくマルチコ)
# train_mae: 33.90532, test_mae: 34.16249 - position_education_partner_segment_position_0_education_0_partner_1
# train_mae: 33.90298, test_mae: 34.16309 ↓ position_education_partner_segment_position_0_education_1_partner_0(除去)
# train_mae: 33.87544, test_mae: 34.13455 - position_education_partner_segment_position_0_education_1_partner_1
# train_mae: 33.86606, test_mae: 34.12542 - position_education_partner_segment_position_0_education_2_partner_0
# train_mae: 33.85531, test_mae: 34.11841 - position_education_partner_segment_position_0_education_2_partner_1
# train_mae: 33.85171, test_mae: 34.11851 ↓ position_education_partner_segment_position_0_education_3_partner_0(除外)
# train_mae: 33.86251, test_mae: 34.12441 ↓ position_education_partner_segment_position_0_education_3_partner_1(除外)
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - position_education_partner_segment_position_0_education_4_partner_0(除外)
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - position_education_partner_segment_position_0_education_4_partner_1(除外)
# train_mae: 33.84333, test_mae: 34.1064  - position_education_partner_segment_position_1_education_0_partner_0
# train_mae: 33.83353, test_mae: 34.09624 - position_education_partner_segment_position_1_education_0_partner_1
# train_mae: 33.81716, test_mae: 34.08468 - position_education_partner_segment_position_1_education_1_partner_1
# train_mae: 33.76775, test_mae: 34.04504 - position_education_partner_segment_position_1_education_2_partner_1
# train_mae: 33.74816, test_mae: 34.02902 - position_education_partner_segment_position_1_education_3_partner_0
# train_mae: 33.68695, test_mae: 33.97543 - position_education_partner_segment_position_1_education_4_partner_0
# train_mae: 33.68684, test_mae: 33.98631 ↓ position_education_partner_segment_position_1_education_4_partner_1(除去)
# train_mae: 33.67881, test_mae: 33.96992 - position_education_partner_segment_position_2_education_0_partner_1
# train_mae: 33.63163, test_mae: 33.92514 - position_education_partner_segment_position_2_education_1_partner_1
# train_mae: 33.61711, test_mae: 33.91659 - position_education_partner_segment_position_2_education_2_partner_0
# train_mae: 33.59101, test_mae: 33.89444 - position_education_partner_segment_position_2_education_2_partner_1
# train_mae: 33.58502, test_mae: 33.89501 ↓ position_education_partner_segment_position_2_education_3_partner_0(除去)
# train_mae: 33.59012, test_mae: 33.89377 ↓ position_education_partner_segment_position_2_education_3_partner_1(除去)
# train_mae: 33.58955, test_mae: 33.90304 ↓ position_education_partner_segment_position_2_education_4_partner_0(除去)
# train_mae: 33.58943, test_mae: 33.8983  ↓ position_education_partner_segment_position_2_education_4_partner_1(除去)
# train_mae: 33.5889,  test_mae: 33.89301 - position_education_partner_segment_position_3_education_0_partner_0
# train_mae: 33.59059, test_mae: 33.89774 ↓ position_education_partner_segment_position_3_education_0_partner_1(除去)
# train_mae: 33.57977, test_mae: 33.88968 - position_education_partner_segment_position_3_education_1_partner_0
# train_mae: 33.57905, test_mae: 33.89395 ↓ position_education_partner_segment_position_3_education_1_partner_1(除去)
# train_mae: 33.57843, test_mae: 33.88923 - position_education_partner_segment_position_3_education_2_partner_1
# train_mae: 33.56006, test_mae: 33.87757 - position_education_partner_segment_position_3_education_3_partner_1
# train_mae: 33.56154, test_mae: 33.87975 ↓ position_education_partner_segment_position_3_education_4_partner_0(除去)
# train_mae: 33.55906, test_mae: 33.88313 ↓ position_education_partner_segment_position_3_education_4_partner_1(除去)
# train_mae: 33.55948, test_mae: 33.87846 ↓ position_education_partner_segment_position_4_education_0_partner_0(除去)
# train_mae: 33.55777, test_mae: 33.88062 ↓ position_education_partner_segment_position_4_education_0_partner_1(除去)
# train_mae: 33.54912, test_mae: 33.86773 - position_education_partner_segment_position_4_education_1_partner_0
# train_mae: 33.54899, test_mae: 33.86946 ↓ position_education_partner_segment_position_4_education_1_partner_1(除去)
# train_mae: 33.53601, test_mae: 33.85009 - position_education_partner_segment_position_4_education_2_partner_1
# train_mae: 33.51753, test_mae: 33.83651 - position_education_partner_segment_position_4_education_3_partner_1
# train_mae: 33.50089, test_mae: 33.81848 - position_education_partner_segment_position_4_education_4_partner_1
# train_mae: 33.51711, test_mae: 33.82981 - ダメそうな変数の除去
# train_mae: 34.03655, test_mae: 34.28747 - 対数変換(salary) まじか・・・(なしよ)



library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("model/LinearRegression_3_lmer/functions.R", encoding = "utf-8")


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
  )

# %>%
#   # 対数変換: salary
#   recipes::step_log(salary, offset = 1)

recipes::prep(recipe) %>% recipes::juice() %>% summary()


future::plan(future::multisession)

system.time(
  {
    set.seed(5963)

    df.results <-

      # クロスバリデーションの分割ごとにループ
      purrr::map(df.cv$splits, model = model.applied, function(df.split, model) {

        # 前処理済データの作成
        df.train.baked <- recipe %>%
          recipes::prep() %>%
          recipes::bake(rsample::training(df.split))
        df.train <- df.train.baked %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(., .) %>%
          # 標準化
          scaling()
        df.test <- recipe %>%
          recipes::prep() %>%
          recipes::bake(rsample::testing(df.split)) %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(df.train.baked) %>%
          # 標準化
          scaling()

        create_lmer_model(df.train) %>%

          # 学習済モデルによる予測
          {
            model <- (.)
            list(
              train = predict(model, df.train),
              test  = predict(model, df.test)
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
                truth    = salary,
                estimate = predicted
                # truth    = salary    %>% exp() %>% { (.) - 1 },
                # estimate = predicted %>% exp() %>% { (.) - 1 }
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
                truth    = salary,
                estimate = predicted
                # truth    = salary    %>% exp() %>% { (.) - 1 },
                # estimate = predicted %>% exp() %>% { (.) - 1 }
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
  }
)
