# TODO
# - area x partner x child の完全組み合わせの low〜high フラグの作成(OK)
# Target Encoding の LLO 対応 # https://mikebird28.hatenablog.jp/entry/2018/06/14/172132 これは test_data に適用するイメージが掴めない・・・
# age と service_length の比 同じ経験期間なら若い方が良いとか？(OK)
# position x education x partner の検討

# ここから下は次回に回す
# partner を使う
# age など他の数量のセグメント平均とかを検討
# 平均だけじゃなくて最大と最小も追加
# 残渣の大きいレコードの調査


# baseline: train_mae: 14.35628, test_mae: 24.33401

# train_mae: 14.2581,  test_mae: 24.3088  ↑ clipping: overtime
# train_mae: 14.24623, test_mae: 24.40136 ↓ clipping: study_time (除去)

# train_mae: 14.22013, test_mae: 24.11858 ↑ area_partner_child_segment
# train_mae: 14.2691,  test_mae: 24.17938 ↓flg_newbie (除去)
# train_mae: 14.86234, test_mae: 23.80893 ↑ position_education_segment

# train_mae: 14.65587, test_mae: 23.49142 ↑ flg_partner_area_commute_high (high + extra_high のやつだね)
# train_mae: 14.58164, test_mae: 23.26663 ↑ flg_partner_area_commute_low
# train_mae: 14.56684, test_mae: 23.21752 ↑ flg_partner_area_commute_extra_low

# train_mae: 14.54691, test_mae: 23.16739 ↑ flg_staff
# train_mae: 14.58257, test_mae: 23.16964 ↓ d (除去)

# train_mae: 14.58242, test_mae: 23.19773 - 順序入れ替え

# train_mae: 14.86219, test_mae: 23.88103 - flg_xxx_commute 系の全消去(ベースライン)
# train_mae: 14.86734, test_mae: 23.75695 ↑ fl_area_partner_child_commute_extra_high
# train_mae: 14.86734, test_mae: 23.75695 ↑ flg_area_partner_child_commute_high
# train_mae: 14.90767, test_mae: 23.7492  ↑ flg_area_partner_child_commute_low
# train_mae: 14.88286, test_mae: 23.76741 ↑ flg_area_partner_child_commute_extra_low
# あとで high & extra_high を混ぜたやつを試してみたい
# train_mae: 14.59396, test_mae: 23.12996 ↑ high & extra_high (1 フラグ)
# train_mae: 14.63208, test_mae: 23.15376 ↑ high & extra_high (2 フラグ)
# 結論: まとめたほうが良さげ
# train_mae: 14.59705, test_mae: 23.05618 ↑ flg_area_partner_child_commute_high + flg_area_partner_child_commute_low + flg_area_partner_child_commute_extra_low
# train_mae: 14.6572,  test_mae: 23.10552 ↑ flg_area_partner_child_commute_extra_high + flg_area_partner_child_commute_high + flg_area_partner_child_commute_low + flg_area_partner_child_commute_extra_low
# やはりまとめる方で
# low と extra_low もまとめてみる？
# train_mae: 14.90396, test_mae: 23.73328 ↑ low & extra_low (2 つ)
# train_mae: 14.82206, test_mae: 23.68759 ↑ low & extra_low (1 つ)
# まとめたほうが良かったw
# train_mae: 14.48375, test_mae: 23.1497 - flg_area_partner_child_commute_high(high & extra_high) & flg_area_partner_child_commute_low(low & extra_low)
# 意外w 最終的には high はまとめて low は個別という形になります
# train_mae: 14.59705, test_mae: 23.05618 ↑ high + low + extra_low

# train_mae: 14.6268,  test_mae: 23.11198 ↓ area_segment やはりw

# train_mae: 14.48677, test_mae: 22.91943 ↑ position_avg_salary
# train_mae: 14.53921, test_mae: 22.95977 ↓ position_avg_commute
# train_mae: 13.82747, test_mae: 22.18418 ↑ diff_position_avg_commute 結果として上がる
# train_mae: 13.70029, test_mae: 22.06952 ↑ ratio_position_avg_commute いいね!
# train_mae: 13.65474, test_mae: 22.02526 ↑ area_partner_child_segment_avg_salary
# train_mae: 13.74213, test_mae: 22.03422 ↓ area_partner_child_segment_avg_commute
# train_mae: 13.39698, test_mae: 21.6364  ↑ diff_area_partner_child_segment_avg_commute いいね！
# train_mae: 13.27108, test_mae: 21.58278 ↑ ratio_area_partner_child_segment_avg_commute

# position を position_education に変更
# train_mae: 13.51355, test_mae: 21.93705 - position の関連指標(salary, commute x avg, diff, ratio) の除去
# train_mae: 13.43786, test_mae: 21.89242 ↑ position_education_segment_avg_salary
# train_mae: 13.40308, test_mae: 21.89913 ↓ position_education_segment_avg_commute
# train_mae: 13.29376, test_mae: 21.76603 ↑ diff_position_education_segment_avg_commute
# train_mae: 13.29708, test_mae: 21.78839 ↓ ratio_position_education_segment_avg_commute
# position_education を外して position に戻す ↓
# train_mae: 13.27108, test_mae: 21.58278 - OK
# 結論: position_education よりは position の方が効く
# education 単体でも追加してみたらどうだろう？？？

# train_mae: 13.21735, test_mae: 21.56504 ↑ education_avg_salary
# train_mae: 13.10158, test_mae: 21.48505 ↑ eposition_ducation__sgment_avg_commute だいぶ向上！間違えとった 急上昇w (除去)
# train_mae: 13.18762, test_mae: 21.52044 ↑ education_avg_commute うーん・・・
# train_mae: 13.155,   test_mae: 21.49326 - position_education_segment_avg_commute 後で再検討します
# train_mae: 13.21664, test_mae: 21.51207 ↑ diff_education_avg_commute
# train_mae: 13.16899, test_mae: 21.52763 ↓ ratio_education_avg_commute (除去)

# position_education_segment_avg_xxx 追加の検討
# education_avg_commute の削除もありかも
# train_mae: 13.18948, test_mae: 21.50824 ↑ education_avg_commute の除去
# train_mae: 13.10671, test_mae: 21.47897 ↑ position_education_segment_avg_salary
# train_mae: 13.07023, test_mae: 21.55704 ↓ position_education_segment_avg_commute
# train_mae: 13.06742, test_mae: 21.5077  - diff_position_education_segment_avg_commute
# train_mae: 13.01755, test_mae: 21.5241  - ratio_position_education_segment_avg_commute
# 結論: position_education_segment_avg_commute 系は全削除
# train_mae: 13.10671, test_mae: 21.47897

# train_mae: 13.05958, test_mae: 21.51226 ↓ position_avg_commute の除去 (やめ！)
# train_mae: 13.10671, test_mae: 21.47897

# train_mae: 13.13009, test_mae: 21.55489 ↓ other_company_term
# train_mae: 13.12552, test_mae: 21.47403 ↑ other_company_term / service_length まじかw
# train_mae: 13.08907, test_mae: 21.54632 ↓ other_company_term / (other_company_term + service_length)
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - (service_length - other_company_term) / service_length

# train_mae: 13.10671, test_mae: 21.47897 - baseline(11/19)
# train_mae: 13.02677, test_mae: 21.50872 ↓ working_years
# train_mae: 13.07769, test_mae: 21.51399 ↓ ratio_trueborn
# train_mae: 13.08385, test_mae: 21.52085 ↓ ratio_trueborn
# train_mae: 13.0704,  test_mae: 21.48325 ↓ ratio_trueborn
# train_mae: 13.12552, test_mae: 21.47403 ↑ (working_years - service_length) / service_length
# train_mae: 13.13844, test_mae: 21.5065  ↓ flg_newbie
# train_mae: 13.12529, test_mae: 21.57136 ↓ flg_newbie_avg_salary
# train_mae: 13.12529, test_mae: 21.57136 - flg_newbie_avg_commute どういうこと？
# train_mae: 13.12469, test_mae: 21.45117 ↑ flg_newbie_avg_salary & flg_newbie_avg_commute
# 下がる/上がる は全体で見ないとわからんのかもなー
# train_mae: 13.16984, test_mae: 21.53187 ↓ diff_flg_newbie_avg_commute
# train_mae: 13.14185, test_mae: 21.55321 ↓ ratio_flg_newbie_avg_commute

# train_mae: 13.12469, test_mae: 21.45117 - baseline
# train_mae: 13.06272, test_mae: 21.56043 ↓ ratio_service_length_per_age orz (除去)

# train_mae: 13.19028, test_mae: 21.53418 ↓ flg_trueborne
# train_mae: 13.14202, test_mae: 21.51661 ↓ flg_trueborne_avg_salary
# train_mae: 13.19805, test_mae: 21.48821 ↓ flg_trueborne_avg_commute
# train_mae: 13.1908,  test_mae: 21.51688 ↓ diff_flg_trueborne_avg_commute
# train_mae: 13.18889, test_mae: 21.56372 ↓ ratio_flg_trueborne_avg_commute あかんw

# 変数を一気に作成して後から重要度で除去していく方が良いのではという感じがプンプンする
# やったからこそ分かるというか実感があるね

# train_mae: 13.12469, test_mae: 21.45117 - baseline
# train_mae: 13.58142, test_mae: 21.47802 ↓ position_education_partner_segment に変更
# train_mae: 13.67566, test_mae: 21.50779 - commute 関連を外す
# train_mae: 13.67738, test_mae: 21.49301 - position_education_partner_segment_avg_commute を追加
# train_mae: 13.53289, test_mae: 21.50794 - diff_position_education_partner_segment_avg_commute を追加
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - ratio_position_education_partner_segment_avg_commute を追加 -> baseline
# train_mae: 13.53596, test_mae: 21.44049 ↑ position_education_partner_segment_avg_salary を外す
# train_mae: 13.71203, test_mae: 21.4895  ↓ position_education_partner_segment_avg_commute のみ残す
# train_mae: 13.58063, test_mae: 21.39846 ↑ position_education_partner_segment_avg_commute のみ外す

# train_mae: 13.80893, test_mae: 21.40762 - position_education_segment を追加(復活)
# train_mae: 13.69386, test_mae: 21.46505 - position_education_segment_avg_salary
# だめや・・・

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - xxx
# ハイパラチューニング
# train_mae: 13.58063, test_mae: 21.39846 - baseline
# train-mae:17.181912+0.068799	test-mae:21.310270+0.339175 trees = 175
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  4, loss_reduction: 0.02682696, mtry: 60, sample_size: 0.8, train_mae: 16.37787, test_mae: 21.23359 ↑
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  5, loss_reduction: 0.02682696, mtry: 80, sample_size: 0.8, train_mae: 16.28601, test_mae: 21.25911 ↓
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  5, loss_reduction: 0.02682696, mtry: 90, sample_size: 0.8, train_mae: 16.23664, test_mae: 21.25863 ↓
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  5, loss_reduction: 0.02682696, mtry: 83, sample_size: 0.8, train_mae: 16.23602, test_mae: 21.30790 ↓
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  5, loss_reduction: 0.02682696, mtry: 82, sample_size: 0.9, train_mae: 16.17102, test_mae: 21.26282 ↓
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  5, loss_reduction: 0.02682696, mtry: 82, sample_size: 0.825, train_mae: 16.25867, test_mae: 21.23389 ↓
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  5, loss_reduction: 0.02682696, mtry: 82, sample_size: 0.8285714, train_mae: 16.27303, test_mae: 21.25965 ↓

# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  6, loss_reduction: 0.02682696, mtry: 82, sample_size: 0.8285714, train_mae: 16.28714, test_mae: 21.23734 ↓
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  7, loss_reduction: 0.02682696, mtry: 82, sample_size: 0.8285714, train_mae: 16.53598, test_mae: 21.32608 ↓

# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  6, loss_reduction: 0.02682696, mtry: 82, sample_size: 0.8285714, train_mae: 16.37831, test_mae: 21.28333 ↓
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  8, loss_reduction: 0.02682696, mtry: 82, sample_size: 0.8285714, train_mae: 16.61775, test_mae: 21.27985 ↓

# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  8, loss_reduction: 0.02511886, mtry: 82, sample_size: 0.8285714, train_mae: 16.65909, test_mae: 21.28437 ↑
# trees: 175, learn_rate: 0.1, tree_depth:  7, min_n:  8, loss_reduction: 0.02772408, mtry: 82, sample_size: 0.8285714, train_mae: 16.67904, test_mae: 21.25601

# [1618]	train-mae:16.574905+0.096108	test-mae:21.026807+0.467536
# trees: 1618, learn_rate: 0.01, tree_depth:  7, min_n:  8, loss_reduction: 0.02772408, mtry: 82, sample_size: 0.8285714, train_mae: 16.65909, test_mae: 21.28437



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
  )

recipes::prep(recipe) %>% recipes::juice() %>% 
  summary()



# Hyper Parameter ---------------------------------------------------------

# 1618 => trees
# recipe %>%
#   recipes::prep() %>%
#   recipes::juice() %>%
#   add_features_per_category(., .) %>%
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
#       colsample_bytree = 0.5985401,
# 
#       max_depth = 7,
#       min_child_weight = 8,
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
  dials::trees(c(1618, 1618)),
  dials::learn_rate(c(-2, -2)),

  dials::mtry(c(82, 82)),

  dials::tree_depth(c(7, 7)),
  dials::min_n(c(8, 8)),

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
    trees = 1618,
    learn_rate = 0.01,
    tree_depth = 7,
    min_n = 8,
    loss_reduction = 0.02772407,
    mtry = 82,
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
      y = predict(fit, df.test, type = "numeric")[[1]]
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


