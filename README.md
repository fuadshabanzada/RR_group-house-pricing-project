# American House Prices Analysis

## Introduction
This project focuses on predicting American house prices using a comprehensive dataset obtained from Kaggle. We employ various machine learning models to explore and predict housing prices based on a variety of features.

## Setup Instructions

### Prerequisites
- R (version 4.3.1 or later)
- RStudio (optional but recommended for ease of use)
- Git

### Clone the Repository
1. Clone the repository:
   ```sh
   git clone https://github.com/fuadshabanzada/RR_group-house-pricing-project.git
   cd RR_group-house-pricing-project

## Conclusion

After training and evaluating multiple models on the dataset, we observed the following RMSE values:

  - Linear Regression: 913,352
- KNN Regression: 398,974.2
- Random Forest: 671,596.9
- Gradient Boosting: 682,216.8
- Linear Boosting: 915,010.8

### Insights:

1. **Linear Regression**:
  - RMSE: 913,352
- Provides a baseline for model performance.
- Can handle linear relationships well but may struggle with more complex interactions.

2. **KNN Regression**:
  - RMSE: 398,974.2
- Simple and intuitive model.
- Performance heavily dependent on the choice of k and scaling of features.
- Achieved the lowest RMSE, indicating a strong performance for this dataset.

3. **Random Forest**:
  - RMSE: 671,596.9
- Robust and powerful ensemble method.
- Handles non-linear relationships and interactions well.
- Performs better than Linear Regression and Linear Boosting.

4. **Gradient Boosting**:
  - RMSE: 682,216.8
- Ensemble method that builds models sequentially.
- Can achieve high performance but requires careful tuning.
- Comparable performance to Random Forest.

5. **Linear Boosting (Elastic Net)**:
  - RMSE: 915,010.8
- Combines linear regression with L1 and L2 regularization.
- Balances between bias and variance, performing feature selection and regularization.
- Did not perform as well as other models for this dataset.

### Best Model:
Among these models, the KNN Regression demonstrated the best performance with the lowest RMSE. This indicates that for this particular dataset, the KNN model is able to capture the underlying patterns more effectively than the other models.

### Future Work:
Further improvements can be made by:
  - Exploring more advanced models and techniques.
- Fine-tuning hyperparameters more thoroughly.
- Including additional relevant features in the dataset.
- Cross-validation to ensure the robustness of the model performance.

### Visualizations:
The visualizations provided insights into the distribution and relationships of different features with the house prices. The scatter plots and histograms helped us understand the data distribution and potential patterns.

Overall, the project demonstrates the application of various machine learning models to predict house prices and highlights the importance of model selection and evaluation.
