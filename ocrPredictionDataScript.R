# Required Packages
install.packages(c("dplyr", "lubridate", "tidyr", "zoo", "readr", "readxl", "stringr"))
library(dplyr)
library(lubridate)
library(tidyr)
library(zoo)
library(readr)
library(readxl)
library(stringr)

print("=== STARTING DATA INTEGRATION ===")

# Read CSVs with progress tracking
print("Reading CSV files...")
cpi_raw <- read_csv("Consumers-price-index,-annual-percentage-change,-June-2021–June-2025-quarters.csv")
print("✓ CPI data loaded")

ocr_raw <- read_csv("Offical Cash Rate (OCR) & 90 day bank bill rate (2).csv")
print("✓ OCR data loaded")

b1_raw <- read_csv("Trade Weighted Index (TWI)  17 Currency Basket - B1.csv")
print("✓ TWI data loaded")

b3_raw <- read_csv("Floating mortgage rate vs Six-month term deposit rate - B3.csv")
print("✓ Mortgage/deposit data loaded")

hm1_raw <- read_csv("hm1.csv", skip = 4, show_col_types = FALSE)  # Skip header rows
print("✓ HM1 CSV data loaded (skipped 4 header rows)")

print("\n=== CLEANING INDIVIDUAL DATASETS ===")

# Clean CPI (From Quarterly to Monthly data)
print("Cleaning CPI data...")
cpi_df <- cpi_raw %>%
  rename(
    Period = `Period ended`,
    CPI_pct = `Percentage change`
  ) %>%
  mutate(
    Date = my(Period),
    month = floor_date(Date, "month")
  ) %>%
  select(month, CPI_pct) %>%
  complete(month = seq.Date(from = min(month), to = max(month), by = "month")) %>%
  arrange(month) %>%
  mutate(CPI_pct = zoo::na.locf(CPI_pct, na.rm = FALSE))

print(paste("CPI data: ", nrow(cpi_df), " rows"))

# Clean OCR (Monthly)
print("Cleaning OCR data...")
ocr_df <- ocr_raw %>%
  rename(Date = 1, Value = 2) %>%
  mutate(
    Date = as.Date(Date, format = "%d-%m-%Y"),
    month = floor_date(Date, "month")
  ) %>%
  group_by(month) %>%
  summarise(OCR = dplyr::last(Value, na_rm = TRUE)) %>%
  ungroup()

print(paste("OCR data: ", nrow(ocr_df), " rows"))

# Clean B1 (TWI)
print("Cleaning TWI data...")
b1_df <- b1_raw %>%
  rename(TWI = `TWI - 17 currency basket`) %>%
  mutate(
    Date = as.Date(Date, format = "%d-%m-%Y"),
    month = floor_date(Date, "month")
  ) %>%
  group_by(month) %>%
  summarise(TWI = last(TWI)) %>%
  ungroup()

print(paste("TWI data: ", nrow(b1_df), " rows"))

# Clean B3 (Mortgage and Deposit Rates)
print("Cleaning mortgage/deposit data...")
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

print(paste("Mortgage/deposit data: ", nrow(b3_df), " rows"))

# HM1 (House Prices)
hm1_df <- hm1_raw %>%
  
  select(
    Date = 1,  # First column should be dates
    CoreInflation = 31,  # Sectoral factor model
    HousePriceGrowth = 34  # House price index 
  ) %>%
  # Handle date parsing more robustly
  mutate(
    # Convert to character first
    Date = as.character(Date),
    # Parse dates
    Date = case_when(
      # Skip rows that are not dates
      is.na(Date) | Date == "" | Date == "NA" ~ as.Date(NA),
      # If it's already a date format
      !is.na(as.Date(Date, tryFormats = c("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"))) ~ 
        as.Date(Date, tryFormats = c("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y")),
      # If it's an Excel date number
      str_detect(Date, "^\\d{5}$") ~ as.Date(as.numeric(Date), origin = "1899-12-30"),
      # If it's in format like "Mar 2023"
      str_detect(Date, "^\\w{3}\\s\\d{4}$") ~ as.Date(paste0("01 ", Date), format = "%d %b %Y"),
      # Default to NA for unparseable dates
      TRUE ~ as.Date(NA)
    ),
    month = floor_date(Date, "month")
  ) %>%
  # Filter for valid dates from 2021 onwards
  filter(!is.na(Date), year(Date) >= 2021) %>%
  select(month, CoreInflation, HousePriceGrowth) %>%
  # Convert to numeric
  mutate(
    CoreInflation = as.numeric(CoreInflation),
    HousePriceGrowth = as.numeric(HousePriceGrowth)
  ) %>%
  # Remove rows where both values are missing
  filter(!(is.na(CoreInflation) & is.na(HousePriceGrowth))) %>%
  # Convert quarterly to monthly using interpolation
  complete(month = seq.Date(from = min(month, na.rm = TRUE), 
                            to = max(month, na.rm = TRUE), 
                            by = "month")) %>%
  arrange(month) %>%
  mutate(
    CoreInflation = zoo::na.approx(CoreInflation, na.rm = FALSE),
    HousePriceGrowth = zoo::na.approx(HousePriceGrowth, na.rm = FALSE)
  )

print(paste("HM1 CSV data: ", nrow(hm1_df), " rows"))

# Clean HLF (Unemployment Rate)
print("Cleaning unemployment data...")
unemployment_df <- read_csv(
  "unemployment_rate.csv", 
  skip = 3,
  col_names = c("Period", "UnemploymentRate"),
  col_types = cols(
    Period = col_character(),
    UnemploymentRate = col_double()
  ),
  show_col_types = FALSE
) %>%
  filter(!is.na(UnemploymentRate), 
         !is.na(Period),
         UnemploymentRate > 0,
         UnemploymentRate < 20) %>%
  mutate(
    Date = yq(Period),
    month = floor_date(Date, "month")
  ) %>%
  filter(!is.na(Date), year(Date) >= 2021) %>%
  select(month, UnemploymentRate) %>%
  complete(month = seq.Date(from = min(month, na.rm = TRUE), 
                            to = max(month, na.rm = TRUE), 
                            by = "month")) %>%
  arrange(month) %>%
  mutate(UnemploymentRate = zoo::na.approx(UnemploymentRate, na.rm = FALSE))

print(paste("Unemployment data: ", nrow(unemployment_df), " rows"))


print("\n Merging all Datasets")

# Use hm1_df
full_data <- ocr_df %>%
  left_join(cpi_df, by = "month") %>%
  left_join(unemployment_df, by = "month") %>%
  left_join(hm1_df, by = "month") %>%  # FIXED: was excel_df, now hm1_df
  left_join(b1_df, by = "month") %>%
  left_join(b3_df, by = "month") %>%
  arrange(month)

print(paste("After initial merge: ", nrow(full_data), " rows"))

# Trim to 2021 Onward and Drop NAs
full_data <- full_data %>%
  filter(month >= as.Date("2021-01-01")) %>%
  drop_na()

print(paste("After filtering and dropping NAs: ", nrow(full_data), " rows"))

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


# Save the dataset
write_csv(full_data, "ocr_prediction_enhanced.csv")
# View Final Dataset
print(head(full_data, 12))
summary(full_data)
