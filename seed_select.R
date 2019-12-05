# v: 7, seed: xxxx, score: xxxxxxxx
# v: 7, seed:  823, score: 1.667539
# v: 7, seed: 4630, score: 1.616755


library(tidyverse)
library(furrr)

future::plan(future::multisession)
system.time(
  df.seed_results <- furrr::future_map_dfr(1001:5000, function(seed, data, recipe) {

    score <- create_cv(data, v = 7, seed = seed) %>%

      dplyr::mutate(
        hoge = purrr::map(splits, function(split, recipe) {
          df.train <- rsample::training(split) %>%
            {
              data <- (.)
              recipes::prep(recipe) %>%
                recipes::bake(data)
            }
          df.train %>%
            dplyr::group_by(area_partner_child_segment) %>%
            dplyr::summarise(avg_salary = mean(salary))
        }, recipe = recipe)
      ) %>%
      dplyr::select(-splits) %>%
      tidyr::unnest(hoge) %>%

      dplyr::group_by(area_partner_child_segment, id) %>%
      dplyr::summarize(segment_avg_salary = mean(avg_salary)) %>%
      dplyr::mutate(diff = mean(segment_avg_salary) - segment_avg_salary) %>%
      dplyr::ungroup() %>%
      dplyr::summarise(score = sqrt(mean(diff^2))) %>%
      .$score

    tibble(
      seed = seed,
      score = score
    )
  }, data = df.train_data, recipe = recipe) %>%

    dplyr::arrange(score)
)


create_cv(df.train_data, v = 7, seed = 4630) %>%
  dplyr::mutate(
    hoge = purrr::map(splits, function(split, recipe) {
      df.train <- rsample::training(split) %>%
        {
          data <- (.)
          recipes::prep(recipe) %>%
            recipes::bake(data)
        }
      df.train %>%
        dplyr::group_by(area_partner_child_segment) %>%
        dplyr::summarise(avg_salary = mean(salary))
    }, recipe = recipe)
  ) %>%
  dplyr::select(-splits) %>%
  tidyr::unnest(hoge) %>%

  ggplot(aes(area_partner_child_segment, avg_salary)) +
    geom_line(aes(group = id, colour = id))
