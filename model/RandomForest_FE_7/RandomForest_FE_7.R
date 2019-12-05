# TODO
# - cv を変更(OK)
# - 集約処理を area_partner_child_segment + 対象カテゴリ となるように変更(OK)
#   - バグ対応 どうするかな・・・(OK)
# - 対数変換: salary(OK)
# - 変数の追加を検討(OK)
# - ハイパラチューニング


# - パラメータフィッティング
# - ハイパラ複数 + seedAverage
# - 提出

# mtry: 11, min_n:  7, trees:  550, max.depth: 15, oob_rmse: 32.04987, train_mae: 13.04984, test_mae: 21.46758
# - 特徴量として線形回帰による予測を追加: 21.73353 orz...

# train_mae: 13.35872, test_mae: 21.67567 ↑ cv を変更
# train_mae: 13.28166, test_mae: 21.08633 ↑ 集約処理を変更 & cv を変更 & 線形ほげほげ除外

# train_mae: 13.66858, test_mae: 21.22374 - 変数を若干修正(XGBoost に合わせた)
# train_mae: 13.5882,  test_mae: 21.19604 ↑ ratio_othercompany_years_per_service_length を除去
# train_mae: 13.632,   test_mae: 21.19199 ↑ working_years の除去
# train_mae: 13.56976, test_mae: 21.18001 ↑ flg_staff の除去
# train_mae: 13.61866, test_mae: 21.33868 ↓ position_education_partner_segment の除去(なしよ)
# train_mae: 13.54269, test_mae: 21.17899 ↑ position_education_segment に入れ替え(むむむ…なしよ)
# train_mae: 13.41626, test_mae: 21.16165 ↑ flg_service_length_0 の除去(上がったw)
# train_mae: 13.34677, test_mae: 21.15885 ↑ flg_overtime_0 の除去
# Validation を変えてまともになったからなのかな・・・
# train_mae: 13.1123,  test_mae: 21.09445 ↑ flg_area_partner_child_commute_xxx 系(3つ)を除去(そうなのwww)
# train_mae: 13.22922, test_mae: 21.14952 ↓ flg_area_partner_child_commute_high を追加(大幅ダウンw 除去)
# train_mae: 13.15875, test_mae: 21.0725  ↑ flg_area_partner_child_commute_low の追加
# train_mae: 13.20982, test_mae: 21.0767  ↓ flg_area_partner_child_commute_extra_low の追加(除去)
# train_mae: 13.04578, test_mae: 21.09978 ↓ flg_newbie の除去(やめ)
# train_mae: 13.11119, test_mae: 21.10588 ↓ flg_newbie だけ(カテゴリ統計量なし)

# train_mae: xxxxxxxx, test_mae: xxxxxxxx - xxx
# train_mae: 13.51548, test_mae: 22.34135 - カテゴリ統計量を全部削除 baseline
# train_mae: 13.06158, test_mae: 22.24953 ↑ 線型モデルの投入 上がりましたな！
# train_mae: 13.17792, test_mae: 21.11883 ↑ position 周りの統計量を追加
# train_mae: 12.92464, test_mae: 21.04921 ↑ 対数変換: salary
# train_mae: 12.94785, test_mae: 21.02629 ↑ education 周りの統計量を追加
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - position_education_partner_segment 周りの統計量はダメでした
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - area_partner_child_segment 周りの統計量もダメでした
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - flg_newbie 周りの統計量もダメ orz
# train_mae: 13.00115, test_mae: 21.02477 ↑ flg_area_partner_child_commute_low 周りの統計量を追加
# train_mae: 13.09062, test_mae: 21.42811 ↓ position および education を position_education_partner_segment に置き換え 激悪化orz(戻す)
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - sex 周りの統計量はダメ
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - partner 周りの統計量はダメ
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - age はダメぽ
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - service_length もダメポ
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - overtime もダメポ
# train_mae: 13.11495, test_mae: 21.01301 ↑ study_time☆
# train_mae: xxxxxxxx, test_mae: xxxxxxxx - num_child 周りの特徴量もダメポだったorz

# mtry: xx, min_n: xx, trees: xxx, max.depth: xx, train_mae: xxxxxxxx, test_mae: xxxxxxxx -

# mtry: 10, min_n:  7, trees: 550, max.depth: 15, train_mae: 13.30467, test_mae: 21.03246 -
# mtry: 11, min_n:  7, trees: 550, max.depth: 15, train_mae: 13.11495, test_mae: 21.01301 -
# mtry: 11, min_n:  7, trees: 550, max.depth: 14, train_mae: 13.96423, test_mae: 21.01411 -
# mtry: 11, min_n:  7, trees: 550, max.depth: 13, train_mae: 14.96786, test_mae: 21.04218 ↓


# min_n x max.depth
# mtry: 11, min_n:  7, trees: 550, max.depth: 15, train_mae: 13.11495, test_mae: 21.01301 -
# mtry: 11, min_n:  7, trees: 550, max.depth: 14, train_mae: 13.96423, test_mae: 21.01411 -
# mtry: 11, min_n:  8, trees: 550, max.depth: 14, train_mae: 14.17526, test_mae: 21.02369 -

# trees
# mtry: 11, min_n:  7, trees:  750, max.depth: 15, train_mae: 13.11948, test_mae: 21.00827 -
# mtry: 11, min_n:  7, trees: 1000, max.depth: 15, train_mae: 13.11936, test_mae: 21.00950 -
# mtry: 11, min_n:  7, trees: 1500, max.depth: 15, train_mae: 13.11450, test_mae: 21.01182 -
# mtry: 11, min_n:  7, trees: 1250, max.depth: 15, train_mae: 13.11412, test_mae: 21.01189 -
# mtry: 11, min_n:  7, trees:  875, max.depth: 15, train_mae: 13.11731, test_mae: 21.00880 -
# mtry: 11, min_n:  7, trees:  625, max.depth: 15, train_mae: 13.11742, test_mae: 21.01169 -
# mtry: 11, min_n:  7, trees: xxxx, max.depth: 15, train_mae: xxxxxxxx, test_mae: xxxxxxxx -

# trees x min_n x max.depth: 最終！
# mtry: 11, min_n:  7, trees:  750, max.depth: 15, train_mae: 13.11948, test_mae: 21.00827 ☆
# mtry: 11, min_n:  7, trees:  875, max.depth: 15, train_mae: 13.11731, test_mae: 21.00880
# mtry: 11, min_n:  7, trees:  750, max.depth: 14, train_mae: 13.96895, test_mae: 21.01143
# mtry: 11, min_n:  7, trees:  875, max.depth: 14, train_mae: 13.96747, test_mae: 21.01225
# mtry: 11, min_n:  8, trees:  750, max.depth: 15, train_mae: 13.38048, test_mae: 21.01674
# mtry: 11, min_n:  8, trees:  750, max.depth: 14, train_mae: 14.17915, test_mae: 21.01834
# mtry: 11, min_n:  8, trees:  875, max.depth: 15, train_mae: 13.37939, test_mae: 21.02027
# mtry: 11, min_n:  8, trees:  875, max.depth: 14, train_mae: 14.17885, test_mae: 21.02244

# mtry: 11, min_n:  7, trees: 550, max.depth: 13, train_mae: 14.96786, test_mae: 21.04218 -


library(tidyverse)
library(tidymodels)
library(furrr)

#set.seed(1025)

source("model/RandomForest_FE_7/functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")

#df.cv <- create_cv(df.train_data, v = 5)
df.cv <- create_cv(df.train_data)


# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  # clipping
  recipes::step_mutate(
    overtime = ifelse(overtime < 25.301, overtime, 25.301),
    study_time = ifelse(study_time <= 13, study_time, 13)
  ) %>%

  recipes::step_mutate(

    # 新卒フラグ
    base_age = dplyr::case_when(
      education == 0 ~ 18,
      education == 1 ~ 20,
      education == 2 ~ 22,
      education == 3 ~ 24,
      education == 4 ~ 27
    ),
    working_years = age - base_age,

    # 生え抜きフラグ
    flg_newbie = (working_years - service_length <= 1),

    # position x education x partner
    position_education_partner_segment = stringr::str_c(
      "position",  position,
      "education", education,
      "partner",   partner,
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

    flg_area_partner_child_commute_low = ifelse(
      (area_partner_child_segment == "country_family_child")     & dplyr::between(commute, 1, 1.5)
      | (area_partner_child_segment == "bigcity_single_nochild") & dplyr::between(commute, 0.5, 1)
      | (area_partner_child_segment == "bigcity_family_nochild") & dplyr::between(commute, 1.5, 2)
      | (area_partner_child_segment == "bigcity_family_child")   & dplyr::between(commute, 2, 2.5),
      1,
      0
    ),
  ) %>%

  # 対数変換(salary)
  recipes::step_log(salary) %>%

  recipes::step_rm(
    id,

    # 新卒フラグ(flg_newbie)関連
    base_age,
    working_years
  )

recipes::prep(recipe) %>% recipes::juice() %>% summary()


# Model Definition --------------------------------------------------------

model <- parsnip::rand_forest(
  mode = "regression",
  mtry = parsnip::varying(),
  min_n = parsnip::varying(),
  trees = parsnip::varying()
) %>%
  parsnip::set_engine(
    engine = "ranger",
    num.threads = 2,
    seed = 1234
  )


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::mtry(c(11, 11)),
  dials::min_n(c(7, 7)),
  dials::trees(c(750, 750)),
  levels = 1
) %>%
  tidyr::crossing(max.depth = c(15))
df.grid.params


# Tuning ------------------------------------------------------------------

# 並列処理
furrr::future_options(seed = F)
future::plan(future::multisession(workers = 4))
#future::plan(future::sequential)


system.time(

  df.results <-

    # ハイパーパラメータをモデルに適用
    purrr::pmap(df.grid.params, function(mtry, min_n, trees, max.depth) {
      parsnip::set_args(
        model,
        mtry = mtry,
        min_n = min_n,
        trees = trees,

        max.depth = max.depth
      )
    }) %>%

    # ハイパーパラメータの組み合わせごとにループ
    purrr::map_dfr(function(model.applied) {
#    furrr::future_map_dfr(function(model.applied) {

      # クロスバリデーションの分割ごとにループ
      furrr::future_map_dfr(df.cv$splits, model = model.applied, function(df.split, model) {
#      purrr::map_dfr(df.cv$splits, model = model.applied, function(df.split, model) {

        # 前処理済データの作成
        df.train <- recipe %>%
          recipes::prep() %>%
          recipes::bake(rsample::analysis(df.split)) %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(., .) %>%
         # 線形モデルによる予測値を追加
         add_linear_model_predictions(., .)
        df.test <- recipe %>%
          recipes::prep() %>%
          recipes::bake(rsample::assessment(df.split)) %>%
          # 訓練/検証 データに代表値を付与
          add_features_per_category(df.train) %>%
          # 線形モデルによる予測値を追加
          add_linear_model_predictions(df.train)

        model %>%

          # モデルの学習
          {
            model <- (.)
            parsnip::fit(model, (salary) ~ ., df.train)
          } %>%

          # 学習済モデルによる予測
          {
            fit <- (.)
            list(
              oob_rmse = sqrt(fit$fit$prediction.error),
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

            # split 単位の結果を格納
            dplyr::bind_cols(
              oob_rmse = lst.predicted$oob_rmse,
              df.result.train,
              df.result.test
            )
          }
      }) %>%

        # CV 分割全体の平均値を評価スコアとする
        dplyr::summarise_all(mean)
    }) %>%

    # 評価結果とパラメータを結合
    dplyr::bind_cols(df.grid.params) %>%

    # 評価スコアの順にソート(昇順)
    dplyr::arrange(
      test_mae
    ) %>%

    dplyr::select(
      mtry,
      min_n,
      trees,

      max.depth,

      oob_rmse,
      train_mae,
      test_mae
    )
)
# xxx

# Importances -------------------------------------------------------------

model %>%

  {
    model <- (.)

    # 前処理済データの作成
    df.train <- recipe %>%
      recipes::prep() %>%
      recipes::juice() %>%
      add_features_per_category(., .) %>%
      # 線形モデルによる予測値を追加
      add_linear_model_predictions(., .)

    # 学習の実施
    parsnip::set_args(
      model,
      mtry = 11,
      min_n = 7,
      trees = 750,
      max.depth = 15,
      importance = "permutation",
      num.threads = 8,
      seed = 1234
    ) %>%
      parsnip::fit(salary ~ ., df.train)
  } %>%

  .$fit %>%
  ranger::importance() %>%
  tibble::enframe(name = "feature", value = "importance") %>%
  dplyr::mutate(feature = forcats::fct_reorder(feature, importance)) %>%

  ggplot(aes(feature, importance)) +
    geom_col() +
    coord_flip() +
    theme_gray(base_family = "Osaka")


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
    # 線形モデルによる予測値を追加
    add_linear_model_predictions(., .)

  # 学習の実施
  model.fitted <- parsnip::set_args(
    model,
    mtry = 11,
    min_n = 7,
    trees = 750,
    max.depth = 15,
    num.threads = 8,
    seed = 1234
  ) %>%
    parsnip::fit(salary ~ ., df.train)

  list(
#    df.train.baked = df.train.baked,
    df.train= df.train,
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
      add_features_per_category(lst.results$df.train) %>%
      # 線形モデルによる予測値を追加
      add_linear_model_predictions(lst.results$df.train)

    # 予測結果データセット
    tibble(
      id = 0:(nrow(df.test)-1),
      y = predict(fit, df.test, type = "numeric")[[1]] %>% exp()
    )
  } %>%

  {
    df.result <- (.)

    # ファイル名
    filename <- stringr::str_c(
      "RandomForest",
      lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
      sep = "_"
    ) %>%
      stringr::str_c("csv", sep = ".")

    # 出力ファイルパス
    filepath <- stringr::str_c("data/output", filename, sep = "/")

    # 書き出し
    readr::write_csv(df.result, filepath, col_names = T)
  }
