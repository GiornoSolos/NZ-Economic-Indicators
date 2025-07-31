
# Required Packages
install.packages(c("dplyr", "lubridate", "tidyr", "zoo", "readr"))
library(dplyr)
library(lubridate)
library(tidyr)
library(zoo)
library(readr)

# Read CSVs 
cpi_raw <- read_csv("Consumers-price-index,-annual-percentage-change,-June-2021–June-2025-quarters.csv")
ocr_raw <- read_csv("Offical Cash Rate (OCR) & 90 day bank bill rate (2).csv")
b1_raw <- read_csv("Trade Weighted Index (TWI)  17 Currency Basket - B1.csv")
b3_raw <- read_csv("Floating mortgage rate vs Six-month term deposit rate - B3.csv")

# Clean CPI (From Quarterly to Monthly data)
cpi_df <- cpi_raw %>%
  rename(
    Period = `Period ended`,
    CPI_pct = `Percentage change`
  ) %>%
  mutate(
    Date = my(Period),  # robust to locale issues
    month = floor_date(Date, "month")
  ) %>%
  select(month, CPI_pct) %>%
  complete(month = seq.Date(from = min(month), to = max(month), by = "month")) %>%
  arrange(month) %>%
  mutate(CPI_pct = zoo::na.locf(CPI_pct, na.rm = FALSE))

# Clean OCR (Monthly)
ocr_df <- ocr_raw %>%
  rename(Date = 1, Value = 2) %>%
  mutate(
    Date = as.Date(Date, format = "%d-%m-%Y"),
    month = floor_date(Date, "month")
  ) %>%
  group_by(month) %>%
  summarise(OCR = dplyr::last(Value, na_rm = TRUE)) %>%  # Correct spelling: na_rm, not na.rm
  ungroup()

# Clean B1 (TWI)
b1_df <- b1_raw %>%
  rename(TWI = `TWI - 17 currency basket`) %>%
  mutate(
    Date = as.Date(Date, format = "%d-%m-%Y"),
    month = floor_date(Date, "month")
  ) %>%
  group_by(month) %>%
  summarise(TWI = last(TWI)) %>%
  ungroup()

# (Mortgage and Deposit Rates)
b3_df <- b3_raw %>%
  rename(
    FloatingMortgage = `Floating mortgage rate`,
    TermDeposit6M = `Six-month term deposit rate`
  ) %>%
  mutate(
    Date = as.Date(Date, format = "%d-%m-%Y"),
    month = floor_date(Date, "month")
  ) %>%
  group_by(month) %>%
  summarise(
    FloatingMortgage = last(FloatingMortgage),
    TermDeposit6M = last(TermDeposit6M)
  ) %>%
  ungroup()

# Merge Everything Together
full_data <- ocr_df %>%
  left_join(cpi_df, by = "month") %>%
  left_join(b1_df, by = "month") %>%
  left_join(b3_df, by = "month") %>%
  arrange(month)

# Trim to 2021 Onward and Drop NAs
full_data <- full_data %>%
  filter(month >= as.Date("2021-01-01")) %>%
  drop_na()

# Create Target Variables
full_data <- full_data %>%
  mutate(
    OCR_next = lead(OCR),
    OCR_direction = case_when(
      OCR_next > OCR ~ "up",
      OCR_next < OCR ~ "down",
      TRUE ~ "same"
    )
  )

# View Final Dataset
print(head(full_data, 12))
summary(full_data)
