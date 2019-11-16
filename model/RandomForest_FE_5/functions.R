source("functions.R", encoding = "utf-8")

add_features_per_category <- function(target_data, train_data) {

  target_data %>%

    # position ごとの平均 salary を追加
    add_feature_per_category(train_data, position, salary) %>%

    # position ごとの commute 平均・差・比の追加
    add_feature_per_category(train_data, position, commute) %>%
    dplyr::mutate(
      diff_position_avg_commute  = commute - position_avg_commute
      # ,
      # ratio_position_avg_commute = commute / position_avg_commute
    ) %>%

    # area x partner x num_child ごとの commute 平均・差・比
    add_feature_per_category(train_data, area_partner_child_segment, commute) %>%
    dplyr::mutate(
      diff_area_partner_child_segment_avg_commute  = commute - area_partner_child_segment_avg_commute
      # ,
      # ratio_area_partner_child_segment_avg_commute = commute / area_partner_child_segment_avg_commute
    ) %>%

    # area x partner x num_child ごとの salary 平均
    add_feature_per_category(train_data, area_partner_child_segment, salary) %>%

    # flg_newbie ごとの salary 平均
    add_feature_per_category(train_data, flg_newbie, salary) %>%

    # flg_service_length_0 ごとの salary 平均
    add_feature_per_category(train_data, flg_service_length_0, salary)
}
#recipes::prep(recipe) %>% juice() %>% add_features_per_category(., .) %>% View

