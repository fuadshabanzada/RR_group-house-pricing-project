# Authors:
# Oybek Ismatov
# Fuad Shabanov
# Qadir Gasimov

# Step 1: Load Data and Libraries
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(tidyr)
library(gbm)      # Gradient Boosting
library(glmnet)   # Linear Boosting
library(class)    # KNN

# Step 2: Data Preprocessing and Exploration
# Load and preprocess the data:

# Dataset file path
file_path <- "C:/Users/Oybek/OneDrive/Рабочий стол/Machine Learning/ML 2 Project Fuad-Oybek/American_Housing_Data_20231209.csv"

# Load the dataset
housing_data <- read_csv(file_path)

# Renaming columns to remove spaces
colnames(housing_data) <- make.names(colnames(housing_data))

# Checking for missing values and handling them
housing_data <- housing_data %>%
  drop_na()

# Display the first few rows of the dataset
head(housing_data)

# Summary statistics of the dataset
summary(housing_data)

# Step 3: Visualization
# Create necessary visualizations:

# Visualizing the distribution of house prices
p1 <- ggplot(housing_data, aes(x = Price)) +
  geom_histogram(binwidth = 10000, fill = "blue", color = "black") +
  labs(title = "Distribution of House Prices", x = "Price", y = "Frequency")
print(p1)
ggsave("distribution_of_house_prices.png", plot = p1)

# Scatter plot of Living.Space vs Price
p2 <- ggplot(housing_data, aes(x = Living.Space, y = Price)) +
  geom_point(color = "blue") +
  labs(title = "Living Space vs Price", x = "Living Space", y = "Price")
print(p2)
ggsave("living_space_vs_price.png", plot = p2)

# Box plot of Price by Number of Beds
p3 <- ggplot(housing_data, aes(x = factor(Beds), y = Price)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Price by Number of Beds", x = "Number of Beds", y = "Price")
print(p3)
ggsave("price_by_number_of_beds.png", plot = p3)

# Additional scatter plots
p4 <- ggplot(housing_data, aes(x = Latitude, y = Price)) +
  geom_point(color = "blue") +
  labs(title = "Latitude vs Price", x = "Latitude", y = "Price")
print(p4)
ggsave("latitude_vs_price.png", plot = p4)

p5 <- ggplot(housing_data, aes(x = Longitude, y = Price)) +
  geom_point(color = "blue") +
  labs(title = "Longitude vs Price", x = "Longitude", y = "Price")
print(p5)
ggsave("longitude_vs_price.png", plot = p5)

# Step 4: Model Training and Evaluation
# Implement the models (Linear Regression, KNN, Random Forest, Gradient Boosting, Linear Boosting):

# Splitting the data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(housing_data$Price, p = 0.8, list = FALSE)
train_data <- housing_data[train_index, ]
test_data <- housing_data[-train_index, ]

# Helper function to calculate RMSE
calculate_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Linear Regression
start_time <- Sys.time()
linear_model <- lm(Price ~ Beds + Baths + Living.Space + Zip.Code.Population + Zip.Code.Density + Median.Household.Income + Latitude + Longitude, data = train_data)
predictions_linear <- predict(linear_model, newdata = test_data)
rmse_linear <- calculate_rmse(test_data$Price, predictions_linear)
cat("Linear Regression RMSE:", rmse_linear, "\n")
cat("Linear Regression Time:", Sys.time() - start_time, "\n")

# KNN Regression
start_time <- Sys.time()
numeric_cols <- sapply(train_data, is.numeric)
train_data_scaled <- scale(train_data[, numeric_cols])
test_data_scaled <- scale(test_data[, numeric_cols])
knn_model <- knn(train_data_scaled, test_data_scaled, train_data$Price, k = 5)
predictions_knn <- as.numeric(as.character(knn_model))
rmse_knn <- calculate_rmse(test_data$Price, predictions_knn)
cat("KNN Regression RMSE:", rmse_knn, "\n")
cat("KNN Regression Time:", Sys.time() - start_time, "\n")

# Random Forest
start_time <- Sys.time()
rf_model <- randomForest(Price ~ Beds + Baths + Living.Space + Zip.Code.Population + Zip.Code.Density + Median.Household.Income + Latitude + Longitude, data = train_data, importance = TRUE)
predictions_rf <- predict(rf_model, newdata = test_data)
rmse_rf <- calculate_rmse(test_data$Price, predictions_rf)
cat("Random Forest RMSE:", rmse_rf, "\n")
cat("Random Forest Time:", Sys.time() - start_time, "\n")

# Gradient Boosting
start_time <- Sys.time()
gbm_model <- gbm(Price ~ Beds + Baths + Living.Space + Zip.Code.Population + Zip.Code.Density + Median.Household.Income + Latitude + Longitude, data = train_data, distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.01, cv.folds = 5)
best_iter <- gbm.perf(gbm_model, method = "cv")
predictions_gbm <- predict(gbm_model, newdata = test_data, n.trees = best_iter)
rmse_gbm <- calculate_rmse(test_data$Price, predictions_gbm)
cat("Gradient Boosting RMSE:", rmse_gbm, "\n")
cat("Gradient Boosting Time:", Sys.time() - start_time, "\n")

# Linear Boosting (Elastic Net)
start_time <- Sys.time()
x_train <- model.matrix(Price ~ Beds + Baths + Living.Space + Zip.Code.Population + Zip.Code.Density + Median.Household.Income + Latitude + Longitude, train_data)[, -1]
y_train <- train_data$Price
x_test <- model.matrix(Price ~ Beds + Baths + Living.Space + Zip.Code.Population + Zip.Code.Density + Median.Household.Income + Latitude + Longitude, test_data)[, -1]
linear_boosting_model <- cv.glmnet(x_train, y_train, alpha = 0.5)
predictions_linear_boosting <- predict(linear_boosting_model, s = "lambda.min", newx = x_test)
rmse_linear_boosting <- calculate_rmse(test_data$Price, predictions_linear_boosting)
cat("Linear Boosting RMSE:", rmse_linear_boosting, "\n")
cat("Linear Boosting Time:", Sys.time() - start_time, "\n")

# Step 5: Model Comparison
# Compare model performances:

model_performance <- data.frame(
  Model = c("Linear Regression", "KNN", "Random Forest", "Gradient Boosting", "Linear Boosting"),
  RMSE = c(rmse_linear, rmse_knn, rmse_rf, rmse_gbm, rmse_linear_boosting)
)

print(model_performance)

# Plot model performance
p6 <- ggplot(model_performance, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  labs(title = "Model Performance Comparison", x = "Model", y = "RMSE")
print(p6)
ggsave("model_performance_comparison.png", plot = p6)

# Step 6: Conclusion

# Update the Conclusion section based on the results:
cat("Conclusion:\n")
cat("Linear Regression RMSE:", rmse_linear, "\n")
cat("KNN Regression RMSE:", rmse_knn, "\n")
cat("Random Forest RMSE:", rmse_rf, "\n")
cat("Gradient Boosting RMSE:", rmse_gbm, "\n")
cat("Linear Boosting RMSE:", rmse_linear_boosting, "\n")

