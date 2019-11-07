library(tidyverse)


# Data Load ---------------------------------------------------------------

# Train Data
load_train_data <- function(path) {
  readr::read_csv(
    path,
    col_types = cols(
      id = col_integer(),
      position = readr::col_factor(levels = 0:4),
      age = col_integer(),
      area = readr::col_factor(),
      sex = readr::col_factor(levels = 1:2),
      partner = readr::col_factor(levels = 0:1),
      num_child = col_integer(),
      education = readr::col_factor(levels = 0:4),
      service_length = col_integer(),
      study_time = col_double(),
      commute = col_double(),
      overtime = col_double(),
      salary = col_double()
    )
  )
}
#df.train <- load_train_data("data/input/train_data.csv")

# Test Data
load_test_data <- function(path) {
  readr::read_csv(
    path,
    col_types = cols(
      id = col_integer(),
      position = readr::col_factor(levels = 0:4),
      age = col_integer(),
      area = readr::col_factor(),
      sex = readr::col_factor(levels = 1:2),
      partner = readr::col_factor(levels = 0:1),
      num_child = col_integer(),
      education = readr::col_factor(levels = 0:4),
      service_length = col_integer(),
      study_time = col_double(),
      commute = col_double(),
      overtime = col_double()
    )
  )
}
#load_test_data("data/input/test_data.csv")

