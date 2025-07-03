## Sales Prediction using XGBoost Regression

_This project predicts sales revenue using regression techniques, with a focus on the XGBoost Regressor._

It follows the complete machine learning lifecycle – from data cleaning and feature selection to model training and evaluation – and aims to demonstrate effective preprocessing and tuning strategies.

#### Problem Statement: 
To predict the sales of retail stores using structured data, enabling better inventory planning and strategic decisions. The dataset contains features like Item Weight, Item Fat Content, Item MRP, Outlet Identifier,Outlet Establishment Year,Outlet Size,Outlet Location Type,Outlet Type,Item Outlet Sales

#### ML Workflow: 
1. Importing Libraries
    1. `numpy`, `pandas` for data manipulation
    2. `matplotlib`, `seaborn` for EDA
    3. `xgboost` for model training
    4. `sklearn` for preprocessing and evaluation
2. Data Loading
    1. Loaded dataset from CSV
    2. Basic checks for shape, missing values, and data types
3. Data Preprocessing
    1. Filled missing values using mean/mode
    2. Handled outliers and inconsistent categories
4. Exploratory Data Analysis (EDA)
    1. Sales distribution per store type
    2. Relationship between sales and promo, holidays, and store size
5. Label Encoding
    1. Used `LabelEncoder` to convert categorical variables into numeric values
6. Feature Selection
    1. Dropped irrelevant features
    2. Retained features with high predictive power
7. Model Training
    1. Used `XGBRegressor` with initial parameters
    2. Trained on training data split using 80:20 ratio
8. Model Evaluation
    1. Metrics used: R² Score (Coefficient of Determination)
    2. Visualized predicted vs actual sales

#### Visualizations
![Screenshot 2025-07-03 152956](https://github.com/user-attachments/assets/bc56523a-0f45-43ab-aa9d-aaa73faaed57)

![Screenshot 2025-07-03 153045](https://github.com/user-attachments/assets/c09c87c8-cf4c-4813-98ee-740ed53a808c)

#### Results
R² Score: 0.87

#### What I Learned
1. Building an end-to-end ML pipeline from scratch
2. How to clean, encode, and structure real-world retail datasets
3. Working with XGBoost and interpreting feature importance
4. Evaluation using R² and RMSE in regression tasks
