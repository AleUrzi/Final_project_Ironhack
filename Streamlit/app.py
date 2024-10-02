import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your dataset
@st.cache_resource()
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'cleaned_df_fetal_health.csv')
    return pd.read_csv(file_path)


# Load the trained model and preprocessing functions
@st.cache_resource()
def load_model():    
    model_path = os.path.join(os.path.dirname(__file__), 'xgb_model.pkl')
    return joblib.load(model_path)

@st.cache_resource()
def load_scaler():
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    return joblib.load(scaler_path)

@st.cache_resource()
def load_label_encoder():
    econder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
    return joblib.load(econder_path)

# User input for CTG parameters
def user_input_features():
    baseline_value_FHR = st.number_input('Baseline FHR (BPM)', min_value=100, max_value=160, value=130)
    accelerations = st.number_input('Accelerations', min_value=0.0, max_value=0.02, value=0.001, format="%.6f")
    fetal_movement = st.number_input('Fetal Movement', min_value=0.0, max_value=0.5, value=0.01, format="%.6f")
    uterine_contractions = st.number_input('Uterine Contractions', min_value=0.0, max_value=0.02, value=0.005, format="%.6f")
    light_decelerations = st.number_input('Light Decelerations', min_value=0.0, max_value=0.02, value=0.001, format="%.6f")
    severe_decelerations = st.number_input('Severe Decelerations', min_value=0.0, max_value=0.001, value=0.000, format="%.6f")
    prolongued_decelerations = st.number_input('Prolonged Decelerations', min_value=0.0, max_value=0.01, value=0.0001, format="%.6f")
    abnormal_short_term_variability = st.number_input('Abnormal Short Term Variability', min_value=10, max_value=100, value=50)
    mean_abnormal_short_term_variability = st.number_input('Mean Abnormal Short Term Variability', min_value=0.2, max_value=7.0, value=1.5)
    percentage_of_time_with_abnormal_long_term_variability = st.number_input('Percentage of Time with Abnormal Long Term Variability', min_value=0, max_value=100, value=10)
    mean_abnormal_long_term_variability = st.number_input('Mean Abnormal Long Term Variability', min_value=0.0, max_value=60.0, value=8.0)
    FHR_width = st.number_input('FHR Width', min_value=5, max_value=200, value=70)
    FHR_min = st.number_input('FHR Min', min_value=50, max_value=160, value=90)
    FHR_max = st.number_input('FHR Max', min_value=120, max_value=240, value=165)
    FHR_n_of_peaks = st.number_input('FHR Number of Peaks', min_value=0, max_value=20, value=4)
    FHR_n_of_zeros = st.number_input('FHR Number of Zeros', min_value=0, max_value=10, value=0)
    FHR_mode = st.number_input('FHR Mode', min_value=60, max_value=190, value=135)
    FHR_mean = st.number_input('FHR Mean', min_value=70, max_value=190, value=135)
    FHR_median = st.number_input('FHR Median', min_value=70, max_value=190, value=138)
    FHR_variance = st.number_input('FHR Variance', min_value=0, max_value=300, value=20)
    FHR_tendency = st.number_input('FHR Tendency', min_value=-1.0, max_value=1.0, value=0.3)

    data = {
        'baseline_value_FHR(BPM)': baseline_value_FHR,
        'accelerations': accelerations,
        'fetal_movement': fetal_movement,
        'uterine_contractions': uterine_contractions,
        'light_decelerations': light_decelerations,
        'severe_decelerations': severe_decelerations,
        'prolongued_decelerations': prolongued_decelerations,
        'abnormal_short_term_variability': abnormal_short_term_variability,
        'mean_abnormal_short_term_variability': mean_abnormal_short_term_variability,
        'percentage_of_time_with_abnormal_long_term_variability': percentage_of_time_with_abnormal_long_term_variability,
        'mean_abnormal_long_term_variability': mean_abnormal_long_term_variability,
        'FHR_width': FHR_width,
        'FHR_min': FHR_min,
        'FHR_max': FHR_max,
        'FHR_n_of_peaks': FHR_n_of_peaks,
        'FHR_n_of_zeros': FHR_n_of_zeros,
        'FHR_mode': FHR_mode,
        'FHR_mean': FHR_mean,
        'FHR_median': FHR_median,
        'FHR_variance': FHR_variance,
        'FHR_tendency': FHR_tendency
    }

    return pd.DataFrame(data, index=[0])

# Prediction function
def predict_fetal_health(input_data):
    model = load_model()
    scaler = load_scaler()
    label_encoder = load_label_encoder()
    
    # Preprocessing: scale the input
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)
    
    # Decode the prediction
    return label_encoder.inverse_transform(prediction)[0]

# Visualization function for box plots
# Visualization function for box plots
def plot_boxplots(data):
    features = data.columns.drop('fetal_health')  # Exclude the fetal_health column
    for feature in features:
        # Create a new figure and axes
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create a box plot with specified order of categories
        sns.boxplot(x='fetal_health', y=feature, data=data, palette="Set2", order=["Normal", "Suspect", "Pathological"], ax=ax)
        
        # Calculate and annotate min and max values for each category
        min_values = data.groupby('fetal_health')[feature].min()
        max_values = data.groupby('fetal_health')[feature].max()
        
        for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
            ax.text(i, min_val, f'Min: {min_val:.2f}', horizontalalignment='center', color='blue', fontsize=10, weight='bold')
            ax.text(i, max_val, f'Max: {max_val:.2f}', horizontalalignment='center', color='red', fontsize=10, weight='bold')

        ax.set_title(f'Box Plot of {feature} by Fetal Health Classification')
        ax.set_xlabel('Fetal Health')
        ax.set_ylabel(feature)
        ax.set_xticklabels(['Normal', 'Suspect', 'Pathological'])  # Correct labels
        ax.grid(axis='y')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Clear the current figure to avoid overlap
        plt.clf()
# Main Streamlit interface
def main():
    st.title("FetalHealth")
    
    # Intro text
    st.markdown("### Welcome to the Fetal Health Prediction App!")
    st.markdown("This application is designed to assist in identifying high-risk fetuses, even without trained medical professionals. By inputting values from a CTG examination below, you can obtain predictions regarding fetal health status.")
    
    # Load dataset
    data = load_data()
    
    # Get user input
    input_df = user_input_features()

    if st.button("Predict"):
        prediction = predict_fetal_health(input_df)
        
        # Normalize prediction output to lowercase
        normalized_prediction = prediction.lower().strip()  # Normalize the string for consistent key matching
        
        # Styling prediction output
        prediction_color = {
            'normal': 'green',
            'suspect': 'orange',
            'pathological': 'red'
        }
        
        # Check if normalized_prediction exists in the prediction_color keys
        if normalized_prediction in prediction_color:
            st.markdown(f"<h2 style='color: {prediction_color[normalized_prediction]}; font-size: 30px;'>Prediction: {normalized_prediction.capitalize()}</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: black; font-size: 30px;'>Prediction: Unknown</h2>", unsafe_allow_html=True)

    # Data visualization
    plot_boxplots(data)

# Run the app
if __name__ == '__main__':
    main()
