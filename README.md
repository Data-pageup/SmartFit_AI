# SmartFit AI

[Streamlit App] (https://smartfit-ai.streamlit.app/)  

[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SmartFit AI is a comprehensive machine-learning-powered system that analyzes workout patterns, dietary habits, and health indicators to deliver personalized insights, predictions, and recommendations. It combines supervised, unsupervised, and deep-learning techniques to model calorie burn, cluster fitness profiles, and suggest optimal workouts and diet plans._

![Dashboard Preview]
<img width="1348" height="562" alt="image" src="https://github.com/user-attachments/assets/90087d81-ba34-42a2-9f2e-7f84e09c0674" />

## Key Highlights
-  Predict calories burned, BMI, and fat percentage
-  Identify user fitness/diet archetypes using clustering
-  Build neural-network models for health profiling
-  Recommend personalized diet and workout routines
-  Interactive Streamlit dashboard for live exploration

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dashboard Sections](#dashboard-sections)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [ licensed Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Overview
SmartFit AI leverages a dataset of user fitness metrics to provide actionable insights. Using techniques like PCA for dimensionality reduction, K-Means clustering for profile segmentation, and neural networks for predictions, it helps users optimize their health journeys. The interactive Streamlit app visualizes data, predicts outcomes, and generates recommendations in real-time.

## Features
- **Predictions**: Real-time calorie burn, BMI, body fat analysis, and workout impact projections.
- **Clustering**- Unsupervised learning to group users into 5 fitness archetypes (Elite Athletes, Strength Builders, Enthusiasts, Beginners, Health Focus).
- **Recommendations**: AI-driven workout schedules and diet plans tailored to user stats, goals, and equipment.
- **Visualizations**: Radar charts, pie charts, heatmaps, scatter plots, and more for intuitive data exploration.
- **Data Analysis**: Correlation heatmaps, PCA visualizations, and distribution analyses.

## Dashboard Sections
The Streamlit app is organized into intuitive sections:

1. ** Dashboard**
   - System overview with key metrics.
   - Cluster distribution visualization.
   - Calorie burn by workout type.
   - BMI distribution analysis.

2. ** Predictions**
   - Calorie Burn Calculator: Real-time prediction based on intensity, duration, and user stats.
   - BMI & Body Fat Analyzer: With visual gauge charts.
   - Workout Impact Predictor: Project weight changes over time with timeline charts.

3. ** Fitness Profiles**
   - 5 Fitness archetypes (Elite Athletes, Strength Builders, Enthusiasts, Beginners, Health Focus).
   - Interactive profile matching based on user input.
   - Radar charts showing fitness attributes.
   - Detailed cluster characteristics.

4. ** Diet Planner**
   - Personalized macronutrient calculations.
   - Sample meal plans for different goals.
   - Macronutrient distribution pie charts.
   - Weekly shopping lists.

5. ** Workout Recommender**
   - AI-generated workout schedules.
   - Detailed strength, cardio, and recovery sessions.
   - Progress tracking metrics.
   - Customized based on experience and equipment.

6. ** Data Explorer**
   - Correlation heatmaps.
   - Interactive distribution visualizations.
   - 2D/3D scatter plots.
   - PCA cluster visualization with explained variance.

## Dataset
- **Shape**: (20,000, 62) – 20,000 rows with 62 features.
- **Key Columns and Datatypes**:
  - `age`: float64
  - `gender`: object
  - `weight_kg`: float64
  - `height_m`: float64
  - `max_bpm`: float64
  - ... (additional features like `pct_fats`: float64, `difficulty_level_enc`: int32, `cluster`: int32, `pca1`: float64, `pca2`: float64)
- **Missing Values (%)**: 0.0 across all columns (e.g., `age`: 0.0, `protein_per_kg`: 0.0, `sets`: 0.0, `benefit`: 0.0, `sodium_mg`: 0.0, `cholesterol_mg`: 0.0, `serving_size_g`: 0.0,0.0).
- The dataset is fully clean with no missing values, making it ideal for direct modeling.

## Model Performance
- **Clustering**: K-Means applied on PCA-reduced features (`pca1`, `pca2`).
- **Classification Report** (on 4,000 test samples):
