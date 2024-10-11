# Final Project: Fetal Health Prediction Using CTG Analysis

## Project Goal
To predict fetal health using Cardiotocography (CTG) results through machine learning (ML).

## Background
Cardiotocography (CTG) is a non-invasive procedure that continuously monitors the fetal heart rate (FHR) and uterine contractions using an ultrasound transducer on the mother's abdomen. CTG is widely used, especially in high-risk pregnancies, to assess fetal well-being. During CTG, several metrics are monitored, including:

- **Uterine Contractions**
- **Baseline Heart Rate**
- **Variability**
- **Accelerations** and **Decelerations**

The standard duration of CTG monitoring is 30 minutes, though this may extend if patterns appear suspicious. FHR classifications include:

- **Baseline FHR**
- **Oscillations** (oscillation rate and amplitude)

Various factors influence FHR, including maternal, fetoplacental, fetal, and external elements.

## Problem Statement
Interpreting CTG data requires trained professionals, posing challenges in regions with limited healthcare resources. This project aims to address this gap by developing a machine learning model to assist in CTG interpretation.

## Dataset
The dataset used in this project can be accessed on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/discussion/194303).

## Feature Information
- **Baseline Value**: FHR in beats per minute
- **Accelerations**: Accelerations per second
- **Fetal Movement**: Fetal movements per second
- **Uterine Contractions**: Uterine contractions per second
- **Light, Severe, and Prolonged Decelerations**: Rates of different deceleration types
- **ASTV / MSTV**: Short-term variability (abnormal percentage and mean)
- **ALTV / MLTV**: Long-term variability (abnormal percentage and mean)
- **FHR Histogram**: Width, minimum, maximum, mode, median, peaks, zeroes, variance, and tendency
- **Target**: Fetal health status (Normal, Suspected, Pathological)

## Data Preprocessing
- Renamed columns for clarity
- Removed 13 duplicate entries

## Exploratory Data Analysis (EDA)
- Analyzed target distribution with value counts and proportions
- Visualized feature distributions using box plots and histograms
- Computed a correlation matrix for numerical features
- Analyzed feature importance

## Machine Learning Process
- Addressed class imbalance with SMOTE
- Applied Robust Scaling to features
- Encoded labels for the XGBoost model, with decoded interpretations
- Trained and evaluated five models:
  - **K-Nearest Neighbors (KNN)**
  - **AdaBoost**
  - **Gradient Boosting**
  - **Random Forest (RF)**
  - **XGBoost**

## Additional Resources
- **Slides**: [Project Slides](https://github.com/AleUrzi/Final_project_Ironhack/blob/main/Slides/final_project_AU.pdf)
- **Tableau Dashboard**: [Tableau Visualization](https://public.tableau.com/app/profile/urzi.alessia/viz/final_project_Ironhack/Dashboard12)
- **Streamlit App**: [Deployed Application](https://aleurzi-final-project-ironhack-streamlitapp-ln2uq4.streamlit.app/)
