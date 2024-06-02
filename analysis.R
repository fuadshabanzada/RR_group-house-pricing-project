# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(tidyr)


file_path <- "C:/Users/Oybek/OneDrive/Рабочий стол/Machine Learning/ML 2 Project Fuad-Oybek/American_Housing_Data_20231209.csv"

# Alternatively, you can use double backslashes
# file_path <- "C:\\Users\\Oybek\\OneDrive\\Рабочий стол\\Machine Learning\\ML 2 Project Fuad-Oybek\\American_Housing_Data_20231209.csv"

# Load the dataset
housing_data <- read_csv(file_path)

# Display the first few rows of the dataset
head(housing_data)

# Summary statistics of the dataset
summary(housing_data)

# Checking for missing values
colSums(is.na(housing_data))

# Handling missing values (if any)
housing_data <- housing_data %>%
  drop_na()

# Visualizing the distribution of house prices
ggplot(housing_data, aes(x = Price)) +
  geom_histogram(binwidth = 10000, fill = "blue", color = "black") +
  labs(title = "Distribution of House Prices", x = "Price", y = "Frequency")

# Splitting the data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(housing_data$Price, p = 0.8, list = FALSE)
train_data <- housing_data[train_index, ]
test_data <- housing_data[-train_index, ]

# Simplified linear regression model
linear_model_simplified <- lm(Price ~ Beds + Baths + `Living Space`, data = train_data)

# Summary of the simplified model
summary(linear_model_simplified)

# Making predictions with the simplified model
predictions_simplified <- predict(linear_model_simplified, newdata = test_data)

# Evaluate the simplified model
rmse_simplified <- sqrt(mean((predictions_simplified - test_data$Price)^2))
r_squared_simplified <- summary(linear_model_simplified)$r.squared

cat("Simplified RMSE:", rmse_simplified, "\n")
cat("Simplified R-squared:", r_squared_simplified, "\n")

# Cross-Validation
train_control <- trainControl(method = "cv", number = 10)
cv_model <- train(Price ~ Beds + Baths + `Living Space`, data = train_data, method = "lm", trControl = train_control)
cv_results <- cv_model$results
cat("Cross-Validation Results:", cv_results, "\n")

# Residual Analysis
residuals_simplified <- residuals(linear_model_simplified)
qqnorm(residuals_simplified)
qqline(residuals_simplified)

# Expanded linear regression model with more predictors
expanded_model <- lm(Price ~ Beds + Baths + `Living Space` + `Zip Code Population` + `Zip Code Density`, data = train_data)

# Summary of the expanded model
summary(expanded_model)

# Making predictions with the expanded model
predictions_expanded <- predict(expanded_model, newdata = test_data)

# Evaluate the expanded model
rmse_expanded <- sqrt(mean((predictions_expanded - test_data$Price)^2))
r_squared_expanded <- summary(expanded_model)$r.squared

cat("Expanded Model RMSE:", rmse_expanded, "\n")
cat("Expanded Model R-squared:", r_squared_expanded, "\n")

# Further expanded linear regression model with additional predictors
further_expanded_model <- lm(Price ~ Beds + Baths + `Living Space` + `Zip Code Population` + `Zip Code Density` + `Median Household Income` + Latitude + Longitude, data = train_data)

# Summary of the further expanded model
summary(further_expanded_model)

# Making predictions with the further expanded model
predictions_further_expanded <- predict(further_expanded_model, newdata = test_data)

# Evaluate the further expanded model
rmse_further_expanded <- sqrt(mean((predictions_further_expanded - test_data$Price)^2))
r_squared_further_expanded <- summary(further_expanded_model)$r.squared

cat("Further Expanded Model RMSE:", rmse_further_expanded, "\n")
cat("Further Expanded Model R-squared:", r_squared_further_expanded, "\n")

# Random Forest Model for Comparison
rf_model <- randomForest(Price ~ Beds + Baths + `Living Space` + `Zip Code Population` + `Zip Code Density` + `Median Household Income` + Latitude + Longitude, data = train_data, importance = TRUE)

# Summary of the random forest model
print(rf_model)

# Making predictions with the random forest model
predictions_rf <- predict(rf_model, newdata = test_data)

# Evaluate the random forest model
rmse_rf <- sqrt(mean((predictions_rf - test_data$Price)^2))
cat("Random Forest Model RMSE:", rmse_rf, "\n")
