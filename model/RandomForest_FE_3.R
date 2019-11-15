# OOB RMSE
# - 33.11167

library(tidyverse)
library(tidymodels)
library(furrr)

set.seed(1025)

source("functions.R", encoding = "utf-8")


# Data Load ---------------------------------------------------------------

df.train_data <- load_train_data("data/input/train_data.csv")


# Feature Engineering -----------------------------------------------------

recipe <- recipes::recipe(salary ~ ., data = df.train_data) %>%

  recipes::step_mutate(
    # 子供ありフラグ
    flg_child = ifelse(num_child > 0, 1, 0),

    # エリアセグメント
    area_segment = dplyr::case_when(
      area %in% c("東京都", "大阪府") ~ "bigcity",
      T                               ~ "country"
    ) %>%
      factor(),

    # partner & num child
    partner_child = ifelse(partner == 0, "no_partner", stringr::str_c("child", num_child, sep = "_")) %>%
      factor(levels = c(
        "no_partner",
        stringr::str_c("child", 0:9, sep = "_")
      )),

    # child & area(segemnt)
    child_area_segment = dplyr::case_when(
      flg_child & (area_segment == "bigcity") ~ "child_bigcity",
      !flg_child & (area_segment == "bigcity") ~ "nochild_bigcity",
      flg_child & (area_segment == "country") ~ "child_country",
      !flg_child & (area_segment == "country") ~ "nochild_country"
    ) %>%
      factor(),

    # area(segment) & partner
    area_segment_partner = dplyr::case_when(
      (area_segment == "bigcity") & (partner == 0) ~ "bigcity_single",
      (area_segment == "bigcity") & (partner == 1) ~ "bigcity_family",
      (area_segment == "country") & (partner == 0) ~ "country_single",
      (area_segment == "country") & (partner == 1) ~ "country_family"
    ) %>%
      factor(),

    # child & area segment & partner
    child_area_segment_partner = dplyr::case_when(
      flg_child & (area_segment == "bigcity") & (partner == 0) ~ "child_bigcity_single",
      flg_child & (area_segment == "bigcity") & (partner == 1) ~ "child_bigcity_family",
      flg_child & (area_segment == "country") & (partner == 0) ~ "child_country_single",
      flg_child & (area_segment == "country") & (partner == 1) ~ "child_country_family",

      !flg_child & (area_segment == "bigcity") & (partner == 0) ~ "nochild_bigcity_single",
      !flg_child & (area_segment == "bigcity") & (partner == 1) ~ "nochild_bigcity_family",
      !flg_child & (area_segment == "country") & (partner == 0) ~ "nochild_country_single",
      !flg_child & (area_segment == "country") & (partner == 1) ~ "nochild_country_family"
    ) %>%
      factor(),

    # child & commute
      child_commute =   flg_child  * commute,
    nochild_commute = (!flg_child) * commute,

    # area(egment) & commute
    bigcity_commute = (area_segment == "bigcity") * commute,
    country_commute = (area_segment == "country") * commute,

    # partner & commute
    single_commute = (partner == 0) * commute,
    family_commute = (partner == 1) * commute,

    # child & area(segment) & commute
    child_bigcity_commute =  flg_child * (area_segment == "bigcity") * commute,
    child_country_commute =  flg_child * (area_segment == "country") * commute,
    nochild_bigcity_commute = (!flg_child) * (area_segment == "bigcity") * commute,
    nochild_country_commute = (!flg_child) * (area_segment == "country") * commute,

    # area(segment) & partner & commute
    bigcity_single_commute = (area_segment == "bigcity") * (partner == 0) * commute,
    bigcity_family_commute = (area_segment == "bigcity") * (partner == 1) * commute,
    country_single_commute = (area_segment == "country") * (partner == 0) * commute,
    country_family_commute = (area_segment == "country") * (partner == 1) * commute,

    # partner & child & commute
    single_child_commute   = (partner == 0) *  flg_child * commute,
    single_nochild_commute = (partner == 0) * (!flg_child) * commute,
    family_child_commute   = (partner == 1) *  flg_child * commute,
    family_nochild_commute = (partner == 1) * (!flg_child) * commute,

    # child & area(segment) & partner & commute
    child_bigcity_single_commute =  flg_child * (area_segment == "bigcity") * (partner == 0) * commute,
    child_bigcity_family_commute =  flg_child * (area_segment == "bigcity") * (partner == 1) * commute,
    child_country_single_commute =  flg_child * (area_segment == "country") * (partner == 0) * commute,
    child_country_family_commute =  flg_child * (area_segment == "country") * (partner == 1) * commute,
    nochild_bigcity_single_commute = (!flg_child) * (area_segment == "bigcity") * (partner == 0) * commute,
    nochild_bigcity_family_commute = (!flg_child) * (area_segment == "bigcity") * (partner == 1) * commute,
    nochild_country_single_commute = (!flg_child) * (area_segment == "country") * (partner == 0) * commute,
    nochild_country_family_commute = (!flg_child) * (area_segment == "country") * (partner == 1) * commute,

    flg_partner_area_commute_high = ifelse(
      ((partner == 0) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1, 2))
      | ((partner == 0) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 2, 3))
      | ((partner == 1) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 2, 3))
      | ((partner == 1) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 4, 5)),
      1,
      0
    ),
    flg_partner_area_commute_low = ifelse(
      ((partner == 0) & (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 0.5, 1))
      | ((partner == 1) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1, 1.5))
      | ((partner == 1) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1.5, 2.5)),
      1,
      0
    ),
    flg_partner_area_commute_extra_low = ifelse(
      ((partner == 0) & dplyr::between(commute, 0, 0.5))
      | ((partner == 1) & !(area %in% c("東京都", "大阪府")) & dplyr::between(commute, 0, 1))
      | ((partner == 1) &  (area %in% c("東京都", "大阪府")) & dplyr::between(commute, 1, 1.5)),
      1,
      0
    ),
  ) %>%

  recipes::step_rm(
    id
  ) %>%

  recipes::step_dummy(recipes::all_nominal(), one_hot = T)

recipes::prep(recipe) %>% recipes::juice() %>% summary()


# Model Definition --------------------------------------------------------

model <- parsnip::rand_forest(
  mode = "regression",
  mtry = parsnip::varying(),
  min_n = parsnip::varying(),
  trees = parsnip::varying()
) %>%
  parsnip::set_engine(engine = "ranger", num.threads = 8, seed = 1234)


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::mtry(c(37, 37)),
  dials::min_n(c(3, 3)),
  dials::trees(c(1000, 1000)),
  levels = 5
)
df.grid.params


# Tuning ------------------------------------------------------------------

# 並列処理
#future::plan(future::multisession(workers = 4))
future::plan(future::sequential)

system.time(
  df.results <-

    # ハイパーパラメータをモデルに適用
    # merge(df.grid.params, model) %>%
    purrr::pmap(df.grid.params, function(mtry, min_n, trees) {
      parsnip::set_args(
        model,
        mtry = mtry,
        min_n = min_n,
        trees = trees
      )
    }) %>%

    # ハイパーパラメータの組み合わせごとにループ
    #  purrr::map(function(model.applied) {
    furrr::future_map_dfr(function(model.applied) {

      # 前処理済データの作成
      df.train <- recipe %>%
        recipes::prep() %>%
        recipes::bake(df.train_data)

      fit <- parsnip::fit(model.applied, salary ~ ., df.train)
      tibble(oob_rmse = sqrt(fit$fit$prediction.error))
    }) %>%

    # 評価結果とパラメータを結合
    dplyr::bind_cols(df.grid.params) %>%

    # 評価スコアの順にソート(昇順)
    dplyr::arrange(
      oob_rmse
    ) %>%

    dplyr::select(
      mtry,
      min_n,
      trees,

      oob_rmse
    )
)


# Predict by Test Data ----------------------------------------------------

# モデルの学習
{
  # 前処理済データの作成
  df.train <- recipe %>%
    recipes::prep() %>%
    recipes::bake(df.train_data)

  # 学習の実施
  parsnip::set_args(
    model,
    mtry = 37,
    min_n = 2,
    trees = 1125,
    num.threads = 8,
    seed = 1234
  ) %>%
    parsnip::fit(salary ~ ., df.train)
} %>%

  # テストデータを用いた予測
  {
    fit <- (.)

    # 前処理済データの作成
    df.test <- recipe %>%
      recipes::prep() %>%
      recipes::bake(load_test_data("data/input/test_data.csv"))

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