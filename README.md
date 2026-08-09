# Automobile MPG Prediction using Polynomial Regression

## 📌 Project Overview

This project focuses on predicting the fuel efficiency of automobiles using Machine Learning.

The dataset contains information about different automobiles, such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin.

The main objective is to preprocess the real-world dataset, perform Exploratory Data Analysis (EDA), and build a Machine Learning model to predict the **Miles Per Gallon (MPG)** of a car.

---

## 🎯 Objective

The goal of this project is to:

- Explore and understand the automobile dataset
- Identify and handle missing values
- Perform data cleaning and preprocessing
- Analyze the distribution of features
- Detect relationships between features using correlation
- Encode categorical variables
- Apply feature scaling
- Create polynomial features
- Train a regression model
- Evaluate the model
- Predict the MPG of a new automobile

---

## 📊 Dataset

The dataset contains information about automobiles and includes the following features:

| Feature | Description |
|---|---|
| `mpg` | Miles Per Gallon (Target Variable) |
| `cylinders` | Number of cylinders |
| `displacement` | Engine displacement |
| `horsepower` | Engine horsepower |
| `weight` | Vehicle weight |
| `acceleration` | Acceleration performance |
| `model_year` | Model year |
| `origin` | Origin of the automobile |
| `name` | Name of the automobile |

The `name` column was removed during preprocessing because it is an identifier/name rather than a useful numerical feature for the model.

---

## 🎯 Target Variable

The target variable is:

```text
mpg
