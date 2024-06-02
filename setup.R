# Authors:
# Oybek Ismatov
# Fuad Shabanov
# Qadir Gasimov

# Install necessary packages
required_packages <- c("readr", "dplyr", "ggplot2", "caret", "randomForest",
                       "e1071", "tidyr", "gbm", "glmnet", "class")

# Function to check and install missing packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# Install and load required packages
install_if_missing(required_packages)

# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(tidyr)
library(gbm)
library(glmnet)
library(class)

cat("Setup complete. All required packages are installed and loaded.\n")
