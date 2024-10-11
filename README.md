# Final_project_Ironhack

AIM OF PROJECT 
predicting fetal health by CTG analysis results with ML

BACKGROUND
Cardiotocography (CTG) is a continuous recording of the fetal heart rate obtained via an ultrasound transducer placed on the mother's abdomen. CTG is widely used in pregnancy as a method of assessing fetal well-being,predominantly in pregnancies with increased risk of complications. What we find using CTG and when we use CTG:-

A cardiotocography (CTG) test is a medical procedure that monitors a pregnant woman's fetal heartbeat and uterine contractions. It's often performed during the third trimester, and is used to assess the baby's well-being, identify problems, and monitor the baby's response during labor. The CTG monitors several different measures, including: Uterine contractions, Baseline heart rate, Variability, Accelerations, and Decelerations. The CTG is typically recommended for pregnancies with an increased risk of complications. The recommended duration of CTG monitoring is 30 minutes, but the duration can be prolonged if the FHR pattern looks suspicious. The fetal heart rate (FHR) is classified into: Baseline fetal heart rate, Oscillations, Oscillation amplitude (range), and Long-term oscillations (oscillation rate). Factors that can affect FHR include: Maternal, Fetoplacental, Fetal, and Exogenous.

PROBLEM 

Analysing CTGs and drawing conclusions is challenging, especially in underdeveloped countries due to a shortage of skilled medical professionals

In this study, a ML model was developed to solve this issue 

DATASET
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/discussion/194303

METADATA
Baseline value: This feature indicates the FHR (fetal heart rate) per BPM (Beats Per Minute). Accelerations: Number of accelerations per second. Fetal movement: Number of fetal movements per second. Uterine contractions: Number of uterine contractions per second. Light decelerations: Number of LDs (light decelerations) per second. Severe decelerations: Number of SDs (severe decelerations) per second. Prolongued decelerations: Number of PDs (prolonged decelerations) per second. ASTV: Percentage of time with abnormal short-term variability. MSTV: Mean value of short-term variability. ALTV: Percentage of time with abnormal long-term variability. MLTV: Mean value of long-term variability. Width: Width of FHR histogram. Min: Minimum of FHR histogram. Max: Maximum of FHR histogram. Nmax: Number of histogram peaks. Nzeros: Number of histogram zeroes. Mode: Histogram mode. Medianc: Histogram median. Variance: Histogram variance. Tendency: Histogram Tendency. Target: Fetal_Health :- Fetal state class code (N=Normal, S=Suspected, P=Pathological)


DATA PRE-PROCESSING:
- changed column names
- dropped 13 duplicated values

EDA
- Target distribution of value counts and proportions
- Features distribution with box plots and histograms
- correlation matrix for numerical features
- Feature importance

MACHINE LEARNING
- Solved target imbalance with SMOTE
- Applied Robust Scaler
- Label encoder (only for XGBoost Model, then decodified for interpretation)
- Trained 5 models: KNN, AdaBoost, GradientBoost, RF, and XGBoost

SLIDES:
https://github.com/AleUrzi/Final_project_Ironhack/blob/main/Slides/final_project_AU.pdf

TABLEAU DASHBOARD:
https://public.tableau.com/app/profile/urzi.alessia/viz/final_project_Ironhack/Dashboard12

STREAMLIT APP
https://aleurzi-final-project-ironhack-streamlitapp-ln2uq4.streamlit.app/
