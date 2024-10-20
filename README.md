# Project Title: Steam Games Recommendation System With Deep Learning

## Overview
 In an era where player engagement is critical to the success of online games, our company seeks to enhance user experience through personalized recommendations. Currently, our recommendation system relies on basic algorithms that do not effectively capture user preferences and behaviors, leading to suboptimal game suggestions. To improve player retention and increase monetization opportunities, we aim to develop a Neural Collaborative Filtering approach model powered by deep learning.
## Problem Statement
The existing recommendation system struggles to provide relevant game suggestions tailored to individual users, resulting in low engagement rates and reduced player satisfaction. The limitations of traditional approaches hinder our ability to leverage vast amounts of user interaction data, including gameplay patterns, in-game purchases, and social interactions.
## Dataset
I used the [Steam Games, Reviews, and Rankings.t](https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings), which includes:

- **Two datasets**: `game_description.csv` and `steam_game_reviews.csv`
- **Features**: Over 20 attributes such as `game_name`, `user_name`, `houred_play`, and `more`.
- **Target Variable**:  Game recommend by user.

## Approach
1. **Data Exploration**: Conducted exploratory data analysis (EDA) to understand the dataset and visualize relationships between features and the target variable.
2. **Data Preprocessing**: Handled missing values, performed feature encoding, and scaled the data to prepare it for modeling.
3. **Model Selection**: Build  `GMF`, `HybMLP` and `NHybF` deep learning models from scratch for classification problem, train and compared to identify the best performer.
4. **Model Evaluation**: Used metrics like `ROC Curve`, `Precision Score` and `Confuse Matrix` to evaluate model performance.

## Results
The final selected `HybMLP` model achieved on test data:
- accuracy: 0.6432
-  precision: 0.94 (approx)
- recall: 0.94 (approx)
- F1 score: 73.74 (approx)

## Deployed App
Please!, click here to check out my model : [Steam Game Recommendation App](https://gamerecommendationapp-z6czighdujsd8q8kauvqcc.streamlit.app/) 

