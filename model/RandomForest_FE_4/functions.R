source("functions.R", encoding = "utf-8")


add_features_per_category <- function(target_data, train_data) {

  target_data %>%

    # position ごとの平均 salary を追加
    add_feature_per_category(train_data, position, salary) %>%

    # position ごとの commute 平均・差・比の追加
    add_feature_per_category(train_data, position, commute) %>%
    dplyr::mutate(
      diff_position_avg_commute  = commute - position_avg_commute,
      ratio_position_avg_commute = commute / position_avg_commute
    ) %>%

    # area_segment ごとの commute 平均・差・比の追加
    add_feature_per_category(train_data, area_segment, commute) %>%
    dplyr::mutate(
      diff_area_segment_avg_commute  = commute - area_segment_avg_commute,
      ratio_area_segment_avg_commute = commute / area_segment_avg_commute
    )
}
#recipes::prep(recipe) %>% juice() %>% add_features_per_category(., .) %>% View

