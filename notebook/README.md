# Project Title: Steam Games Recommendation System With Deep Learning

## Overview
This project involves building a machine learning model to predict house prices based on various features such as location, square footage, number of bedrooms, and amenities. The primary goal was to create an accurate and efficient model that can assist potential buyers and sellers in understanding property values.

## Problem Statement
Real estate prices can fluctuate significantly based on various factors. Accurately predicting house prices helps stakeholders make informed decisions. This project explores a dataset to identify patterns and build a predictive model.

## Dataset
I used the [Steam Games, Reviews, and Rankings.t](https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings), which includes:

- **Two datasets**: `game_description.csv` and `steam_game_reviews.csv`
- **Features**: Over 20 attributes such as `game_name`, `user_name`, `houred_play`, and `more`.
- **Target Variable**:  Game recommend by user.

## Approach
1. **Data Exploration**: Conducted exploratory data analysis (EDA) to understand the dataset and visualize relationships between features and the target variable.
2. **Data Preprocessing**: Handled missing values, performed feature encoding, and scaled the data to prepare it for modeling.
3. **Model Selection**: Compared `GMF`, `HybMLP` and `NHybF` deep learning models to identify the best performer.
4. **Model Evaluation**: Used metrics like `ROC Curve`, `Precision Score` and `Confuse Matrix` to evaluate model performance.

## Results
The final selected `HybMLP` model achieved:
-  precision: 93 (approx)
- F1 score: 73 (approx)

## Deployed App
Please!, click here: [Steam Game Recommendation App](https://gamerecommendationapp-z6czighdujsd8q8kauvqcc.streamlit.app/) 

