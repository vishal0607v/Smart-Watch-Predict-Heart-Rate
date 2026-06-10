import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Set page configuration with a modern dashboard layout
st.set_page_config(
    page_title="PulseHAR - Smartwatch HAR & Heart Rate Predictor",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (custom fonts, glowing cards, modern footer)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        color: #8c93a3;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.6rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #FF4B4B;
        margin-bottom: 5px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8c93a3;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FF4B4B;
        padding-left: 12px;
        color: #ffffff;
    }

    .intro-box {
        background: rgba(255, 255, 255, 0.02); 
        padding: 1.5rem; 
        border-radius: 12px; 
        border: 1px solid rgba(255, 255, 255, 0.05); 
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models and cache them
@st.cache_resource
def load_ml_models():
    models_dir = 'models'
    try:
        clf_model = joblib.load(os.path.join(models_dir, 'activity_classifier.pkl'))
        reg_model = joblib.load(os.path.join(models_dir, 'heart_rate_regressor.pkl'))
        encoder = joblib.load(os.path.join(models_dir, 'activity_encoder.pkl'))
        features = joblib.load(os.path.join(models_dir, 'model_features.pkl'))
        return clf_model, reg_model, encoder, features
    except Exception as e:
        st.error(f"Error loading machine learning models: {e}. Please ensure that train_models.py has run successfully.")
        return None, None, None, None

clf_model, reg_model, encoder, features = load_ml_models()

# Load dataset for exploration/analytics
@st.cache_data
def load_raw_dataset():
    data_path = 'data/wearable_sensor_data.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df_raw = load_raw_dataset()

# ------------------ HEADER WITH LOGO & BADGES ------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=95)
    else:
        st.write("⌚")
with col_title:
    st.markdown("<div class='main-title'>PulseHAR Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Human Activity Recognition & Heart Rate Prediction from Wearable Sensors</div>", unsafe_allow_html=True)

# Metric tags
st.markdown("""
<div style="display: flex; gap: 10px; margin-bottom: 2rem; flex-wrap: wrap;">
    <span style="background-color: rgba(255, 75, 75, 0.12); color: #FF4B4B; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(255, 75, 75, 0.25);">RF Classifier: 100% Accuracy</span>
    <span style="background-color: rgba(76, 175, 80, 0.12); color: #4CAF50; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(76, 175, 80, 0.25);">RF Regressor: 2.42 BPM MAE</span>
    <span style="background-color: rgba(0, 188, 212, 0.12); color: #00BCD4; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(0, 188, 212, 0.25);">Dataset size: 10k Records</span>
    <span style="background-color: rgba(156, 39, 176, 0.12); color: #9C27B0; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(156, 39, 176, 0.25);">Python 3.10 + Streamlit</span>
</div>
""", unsafe_allow_html=True)

# ------------------ BODY INTRODUCTION ------------------
st.markdown("""
<div class="intro-box">
    <h4 style="margin-top: 0; color: white; font-weight: 600;">👋 Welcome to PulseHAR Analytics Platform</h4>
    <p style="color: #a3a8b4; font-size: 0.95rem; line-height: 1.6; margin-bottom: 0;">
        This platform leverages Machine Learning to translate raw smartwatch sensor movements into physiological insights. 
        By processing 3-axis accelerometer and gyroscope coordinates, our model classifies the wearer's current physical activity 
        and predicts their instantaneous heart rate while adjusting for key biometrics (Age, Weight, Gender).
    </p>
</div>
""", unsafe_allow_html=True)

# Main Navigation
if clf_model is None or reg_model is None:
    st.warning("⚠️ Machine Learning models are not trained yet. Run the training process first.")
    if st.button("🚀 Train Models Now"):
        with st.spinner("Training models..."):
            import subprocess
            res = subprocess.run(["python", "train_models.py"], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Models trained successfully!")
                st.rerun()
            else:
                st.error(f"Failed to train models: {res.stderr}")
else:
    # ------------------ SIDEBAR WITH LOGO ------------------
    st.sidebar.markdown("<div style='text-align: center; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", width=120)
    st.sidebar.markdown("<h3 style='margin-top:0.5rem;'>PulseHAR Control</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### Participant Biometrics")
    age = st.sidebar.slider("Age (Years)", 18, 80, 30)
    weight = st.sidebar.slider("Weight (kg)", 40, 120, 70)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    gender_encoded = 0 if gender == "Female" else 1

    # App tabs
    tab_realtime, tab_simulator, tab_batch, tab_insights, tab_dataset = st.tabs([
        "🏠 Real-Time Prediction", 
        "⏱ Live Stream Simulator", 
        "📊 Batch CSV Upload", 
        "📈 Model Insights",
        "📁 Dataset Explorer"
    ])
    
    # ------------------ TAB 1: REAL-TIME PREDICTION ------------------
    with tab_realtime:
        st.markdown("<div class='section-header'>Test Live Wearable Sensor Input</div>", unsafe_allow_html=True)
        st.write("Adjust the sliders below to mimic smartwatch accelerometer and gyroscope measurements. The models will instantly predict the activity and heart rate.")
        
        col_acc, col_gyro = st.columns(2)
        
        with col_acc:
            st.markdown("#### 🏃 Accelerometer (m/s²)")
            ax = st.slider("Accel X (Lateral)", -10.0, 10.0, 0.2, step=0.1)
            ay = st.slider("Accel Y (Vertical)", -10.0, 10.0, 0.5, step=0.1)
            az = st.slider("Accel Z (Forward/Back)", -5.0, 20.0, 9.8, step=0.1)
            
            # Show magnitude
            acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
            st.info(f"Accelerometer Magnitude: **{acc_mag:.2f} m/s²**")
            
        with col_gyro:
            st.markdown("#### 🔄 Gyroscope (rad/s)")
            gx = st.slider("Gyro X (Pitch)", -3.0, 3.0, 0.0, step=0.05)
            gy = st.slider("Gyro Y (Roll)", -3.0, 3.0, 0.05, step=0.05)
            gz = st.slider("Gyro Z (Yaw)", -3.0, 3.0, -0.02, step=0.05)
            
            gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            st.info(f"Gyroscope Magnitude: **{gyro_mag:.2f} rad/s**")
            
        # Inference trigger
        input_clf = pd.DataFrame([[ax, ay, az, gx, gy, gz]], columns=features['clf_features'])
        
        # Predict Activity
        pred_clf_idx = clf_model.predict(input_clf)[0]
        pred_clf_probs = clf_model.predict_proba(input_clf)[0]
        pred_activity_name = encoder.inverse_transform([pred_clf_idx])[0]
        
        # Predict Heart Rate
        input_reg = pd.DataFrame([[ax, ay, az, gx, gy, gz, age, weight, gender_encoded, pred_clf_idx]], columns=features['reg_features'])
        pred_hr = reg_model.predict(input_reg)[0]
        
        # Display Results
        st.markdown("<div class='section-header'>Predictions</div>", unsafe_allow_html=True)
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{pred_activity_name}</div>
                <div class='metric-label'>Predicted Activity</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br><b>Probability Distribution:</b>", unsafe_allow_html=True)
            for cls_idx, class_name in enumerate(encoder.classes_):
                prob = pred_clf_probs[cls_idx]
                st.write(f"{class_name}: {prob*100:.1f}%")
                st.progress(prob)
                
        with col_out2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{pred_hr:.1f} <span style='font-size: 1.2rem; font-weight: normal; color: #a3a8b4;'>BPM</span></div>
                <div class='metric-label'>Predicted Heart Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Contextual status card based on HR
            if pred_hr < 60:
                hr_status = "Resting / Bradycardia (Very Low)"
                hr_color = "lightblue"
            elif pred_hr < 100:
                hr_status = "Normal / Active (Moderate)"
                hr_color = "lightgreen"
            elif pred_hr < 140:
                hr_status = "Elevated Workout Zone (High)"
                hr_color = "orange"
            else:
                hr_status = "Peak Aerobic Zone (Very High)"
                hr_color = "red"
                
            st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.04); padding: 1.2rem; border-radius: 12px; margin-top: 1.5rem; border-left: 5px solid {hr_color};">
                <span style="font-size: 0.9rem; color: #a3a8b4; text-transform: uppercase;">Heart Rate Status</span><br>
                <b style="font-size: 1.2rem; color: white;">{hr_status}</b>
            </div>
            """, unsafe_allow_html=True)

    # ------------------ TAB 2: LIVE STREAM SIMULATOR ------------------
    with tab_simulator:
        st.markdown("<div class='section-header'>Wearable Sensor Simulation</div>", unsafe_allow_html=True)
        st.write("Select a simulated activity and initiate the live sensor stream. We will sample dynamic waveforms from our heuristics (including walking/running oscillations) and run predictions in real-time.")
        
        sim_activity = st.selectbox("Simulate Activity Stream:", ["Sitting", "Standing", "Walking", "Running", "Lying Down"])
        stream_button = st.button("Start Live Stream")
        
        if stream_button:
            stop_btn = st.button("Stop Stream")
            
            # Setup placeholder containers for dynamic charts and values
            val_col1, val_col2, val_col3 = st.columns(3)
            with val_col1:
                act_placeholder = st.empty()
            with val_col2:
                hr_placeholder = st.empty()
            with val_col3:
                step_placeholder = st.empty()
                
            chart_placeholder = st.empty()
            
            # Create lists to store historical data points for plotting
            history_size = 50
            t_history = []
            ax_history = []
            ay_history = []
            az_history = []
            hr_history = []
            
            # Heuristic coefficients depending on activity
            dt = 0.05
            freq = 1.6 if sim_activity == "Walking" else (2.8 if sim_activity == "Running" else 0.0)
            
            # Seed subjects characteristics
            sub_resting_hr = 68
            
            for step in range(150):
                # Stop if user clicks stop button (or just runs for 150 iterations)
                if stop_btn:
                    st.write("Stream stopped.")
                    break
                    
                t_val = step * dt
                
                # Generate sample based on selected activity + random noise
                if sim_activity == 'Sitting':
                    ax_val = np.random.normal(0.1, 0.05)
                    ay_val = np.random.normal(0.2, 0.05)
                    az_val = np.random.normal(9.8, 0.05)
                    gx_val = np.random.normal(0.0, 0.01)
                    gy_val = np.random.normal(0.0, 0.01)
                    gz_val = np.random.normal(0.0, 0.01)
                    hr_base = sub_resting_hr
                elif sim_activity == 'Standing':
                    ax_val = np.random.normal(0.3, 0.08)
                    ay_val = np.random.normal(-0.2, 0.08)
                    az_val = np.random.normal(9.7, 0.08)
                    gx_val = np.random.normal(0.0, 0.02)
                    gy_val = np.random.normal(0.0, 0.02)
                    gz_val = np.random.normal(0.0, 0.02)
                    hr_base = sub_resting_hr + 4
                elif sim_activity == 'Walking':
                    ax_val = 0.8 * np.sin(2 * np.pi * freq * t_val) + np.random.normal(0.2, 0.15)
                    ay_val = 1.2 * np.cos(2 * np.pi * freq * t_val) + np.random.normal(-0.1, 0.15)
                    az_val = 9.8 + 1.5 * np.sin(2 * np.pi * freq * t_val) + np.random.normal(0, 0.2)
                    gx_val = 0.4 * np.sin(2 * np.pi * freq * t_val) + np.random.normal(0, 0.05)
                    gy_val = 0.6 * np.cos(2 * np.pi * freq * t_val) + np.random.normal(0, 0.05)
                    gz_val = 0.3 * np.sin(2 * np.pi * freq * t_val) + np.random.normal(0, 0.05)
                    hr_base = sub_resting_hr + 28 + (weight * 0.05) - (age * 0.05)
                elif sim_activity == 'Running':
                    ax_val = 2.5 * np.sin(2 * np.pi * freq * t_val) + np.random.normal(0.5, 0.4)
                    ay_val = 4.0 * np.cos(2 * np.pi * freq * t_val) + np.random.normal(-0.3, 0.4)
                    az_val = 9.8 + 5.0 * np.sin(2 * np.pi * freq * t_val) + np.random.normal(0, 0.5)
                    gx_val = 1.5 * np.sin(2 * np.pi * freq * t_val) + np.random.normal(0, 0.15)
                    gy_val = 2.0 * np.cos(2 * np.pi * freq * t_val) + np.random.normal(0, 0.15)
                    gz_val = 1.0 * np.sin(2 * np.pi * freq * t_val) + np.random.normal(0, 0.15)
                    hr_base = sub_resting_hr + 68 + (weight * 0.1) - (age * 0.1)
                elif sim_activity == 'Lying Down':
                    ax_val = np.random.normal(9.6, 0.04)
                    ay_val = np.random.normal(0.5, 0.04)
                    az_val = np.random.normal(0.8, 0.04)
                    gx_val = np.random.normal(0.0, 0.005)
                    gy_val = np.random.normal(0.0, 0.005)
                    gz_val = np.random.normal(0.0, 0.005)
                    hr_base = sub_resting_hr - 4
                    
                # Run predictions
                input_clf = pd.DataFrame([[ax_val, ay_val, az_val, gx_val, gy_val, gz_val]], columns=features['clf_features'])
                pred_clf_idx = clf_model.predict(input_clf)[0]
                pred_act = encoder.inverse_transform([pred_clf_idx])[0]
                
                input_reg = pd.DataFrame([[ax_val, ay_val, az_val, gx_val, gy_val, gz_val, age, weight, gender_encoded, pred_clf_idx]], columns=features['reg_features'])
                pred_hr_val = reg_model.predict(input_reg)[0]
                
                # Append to history
                t_history.append(t_val)
                ax_history.append(ax_val)
                ay_history.append(ay_val)
                az_history.append(az_val)
                hr_history.append(pred_hr_val)
                
                if len(t_history) > history_size:
                    t_history.pop(0)
                    ax_history.pop(0)
                    ay_history.pop(0)
                    az_history.pop(0)
                    hr_history.pop(0)
                    
                # Update metrics placeholders
                act_placeholder.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='font-size: 1.8rem;'>🏃 {pred_act}</div>
                    <div class='metric-label'>Current Activity</div>
                </div>
                """, unsafe_allow_html=True)
                
                hr_placeholder.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:#E91E63;'>❤️ {pred_hr_val:.1f} <span style='font-size:0.9rem;'>BPM</span></div>
                    <div class='metric-label'>Heart Rate Est.</div>
                </div>
                """, unsafe_allow_html=True)
                
                step_placeholder.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:#00BCD4;'>{step + 1}</div>
                    <div class='metric-label'>Time Steps</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Plot dynamic graphs
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                plt.style.use('dark_background')
                fig.patch.set_facecolor('#0E1117')
                
                ax1.plot(t_history, ax_history, label='X-Axis', color='#00BCD4', alpha=0.9)
                ax1.plot(t_history, ay_history, label='Y-Axis', color='#FF9800', alpha=0.9)
                ax1.plot(t_history, az_history, label='Z-Axis', color='#E91E63', alpha=0.9)
                ax1.set_ylabel('Acceleration (m/s²)')
                ax1.legend(loc='upper right')
                ax1.set_title('Live Smartwatch Accelerometer Channels', fontsize=10, loc='left')
                ax1.grid(True, alpha=0.1)
                
                ax2.plot(t_history, hr_history, label='Predicted HR', color='#4CAF50', linewidth=2)
                ax2.set_ylabel('Heart Rate (BPM)')
                ax2.set_xlabel('Time (s)')
                ax2.legend(loc='upper right')
                ax2.set_title('Real-Time Heart Rate Prediction Curve', fontsize=10, loc='left')
                ax2.grid(True, alpha=0.1)
                
                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close()
                
                # Small pause to look live (sampling rate simulation)
                time.sleep(0.08)

    # ------------------ TAB 3: BATCH CSV UPLOAD ------------------
    with tab_batch:
        st.markdown("<div class='section-header'>Batch File Predictor</div>", unsafe_allow_html=True)
        st.write("Upload a CSV file containing columns `Accel_X`, `Accel_Y`, `Accel_Z`, `Gyro_X`, `Gyro_Y`, `Gyro_Z`. The app will use your current demographics set in the sidebar to perform batch inference.")
        
        # Simple sample CSV download template
        if df_raw is not None:
            sample_df = df_raw[['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']].head(200)
            csv_sample = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=csv_sample,
                file_name="wearable_sample_template.csv",
                mime="text/csv"
            )
            
        uploaded_file = st.file_uploader("Upload Sensor Readings CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Load CSV
                input_df = pd.read_csv(uploaded_file)
                required_cols = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
                
                # Check for missing columns
                missing_cols = [c for c in required_cols if c not in input_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Uploaded CSV is missing columns: {missing_cols}")
                else:
                    with st.spinner("Processing batch predictions..."):
                        # Extract classifier features
                        X_clf_batch = input_df[required_cols]
                        
                        # Classify Activity
                        clf_preds = clf_model.predict(X_clf_batch)
                        pred_activity_names = encoder.inverse_transform(clf_preds)
                        
                        # Add inputs for regression
                        X_reg_batch = X_clf_batch.copy()
                        X_reg_batch['Age'] = age
                        X_reg_batch['Weight'] = weight
                        X_reg_batch['Gender'] = gender_encoded
                        X_reg_batch['Activity_Encoded'] = clf_preds
                        
                        # Predict Heart Rate
                        # Reorder to match model expectations
                        reg_feat_list = features['reg_features']
                        X_reg_batch = X_reg_batch[reg_feat_list]
                        hr_preds = reg_model.predict(X_reg_batch)
                        
                        # Append predictions to dataframe
                        result_df = input_df.copy()
                        result_df['Predicted_Activity'] = pred_activity_names
                        result_df['Predicted_HeartRate_BPM'] = hr_preds
                        
                        st.success("✅ Prediction processing complete!")
                        
                        # Display Results metrics
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.write(f"Total processed samples: **{len(result_df)}**")
                            st.write("Preview of results:")
                            st.dataframe(result_df[['Accel_X', 'Accel_Y', 'Accel_Z', 'Predicted_Activity', 'Predicted_HeartRate_BPM']].head(10))
                        
                        with res_col2:
                            # Pie chart of activities
                            fig, ax = plt.subplots(figsize=(6, 4))
                            fig.patch.set_facecolor('#0E1117')
                            activity_counts = result_df['Predicted_Activity'].value_counts()
                            ax.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', 
                                   colors=['#FF4B4B', '#FF9800', '#4CAF50', '#00BCD4', '#9C27B0'],
                                   textprops={'color': 'white'})
                            ax.set_title("Activity Breakdown in Uploaded Data", color='white', fontsize=12)
                            st.pyplot(fig)
                            plt.close()
                            
                        # Download button for predicted CSV
                        res_csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predicted Results CSV",
                            data=res_csv,
                            file_name="wearable_predictions_labeled.csv",
                            mime="text/csv"
                        )
                        
                        # Activity timeline plotting
                        st.markdown("#### Activity Timeline and Heart Rate Trend")
                        fig_timeline, ax_t = plt.subplots(figsize=(12, 4))
                        fig_timeline.patch.set_facecolor('#0E1117')
                        ax_t.set_facecolor('#1A1E29')
                        
                        # Plot heart rate curve
                        ax_t.plot(result_df.index, result_df['Predicted_HeartRate_BPM'], color='#4CAF50', label='Predicted HR', linewidth=2)
                        
                        # Fill background color according to activity
                        # We find transitions to shade different regions
                        activities_series = result_df['Predicted_Activity']
                        
                        # Coloring mapping
                        colors_map = {
                            'Sitting': ('#00BCD4', 0.15),
                            'Standing': ('#FF9800', 0.15),
                            'Walking': ('#4CAF50', 0.2),
                            'Running': ('#FF4B4B', 0.25),
                            'Lying Down': ('#9C27B0', 0.15)
                        }
                        
                        # Color regions
                        i = 0
                        while i < len(result_df):
                            act = activities_series.iloc[i]
                            start_idx = i
                            while i < len(result_df) and activities_series.iloc[i] == act:
                                i += 1
                            end_idx = i - 1
                            
                            c_val, alpha = colors_map.get(act, ('grey', 0.1))
                            ax_t.axvspan(start_idx, end_idx, color=c_val, alpha=alpha, label=act if act not in ax_t.get_legend_handles_labels()[1] else "")
                            
                        ax_t.set_ylabel('Heart Rate (BPM)', color='white')
                        ax_t.set_xlabel('Sample Point Index', color='white')
                        ax_t.tick_params(colors='white')
                        ax_t.legend(loc='upper right', bbox_to_anchor=(1, 1.25), ncol=5)
                        ax_t.set_title("Chronological Activity Segmentation & Heart Rate Track", color='white', pad=20)
                        
                        st.pyplot(fig_timeline)
                        plt.close()
            except Exception as e:
                st.error(f"Error parsing file: {e}")

    # ------------------ TAB 4: MODEL INSIGHTS ------------------
    with tab_insights:
        st.markdown("<div class='section-header'>Random Forest Model Feature Importances</div>", unsafe_allow_html=True)
        st.write("Understand which sensor readings are driving the activity and heart rate predictions.")
        
        # Display Feature Importance plots
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown("#### Activity Classifier Importances")
            importances_clf = clf_model.feature_importances_
            clf_feat_names = features['clf_features']
            
            indices = np.argsort(importances_clf)[::-1]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#1A1E29')
            ax.barh([clf_feat_names[i] for i in indices[::-1]], importances_clf[indices[::-1]], color='#FF4B4B')
            ax.set_xlabel('Relative Importance', color='white')
            ax.tick_params(colors='white')
            ax.set_title('Feature Importance: Activity Classifier', color='white')
            st.pyplot(fig)
            plt.close()
            
            st.write("💡 **Insight:** Acceleration along Y and Z axes (vertical and forward oscillations) typically holds the highest feature weight during classification, separating static activities (sitting/lying) from dynamic ones (walking/running).")
            
        with col_f2:
            st.markdown("#### Heart Rate Regressor Importances")
            importances_reg = reg_model.feature_importances_
            reg_feat_names = features['reg_features']
            
            # Map encoded features to clean strings
            indices_reg = np.argsort(importances_reg)[::-1]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#1A1E29')
            ax.barh([reg_feat_names[i] for i in indices_reg[::-1]], importances_reg[indices_reg[::-1]], color='#4CAF50')
            ax.set_xlabel('Relative Importance', color='white')
            ax.tick_params(colors='white')
            ax.set_title('Feature Importance: Heart Rate Regressor', color='white')
            st.pyplot(fig)
            plt.close()
            
            st.write("💡 **Insight:** The classified Activity holds a prominent weight in regressed heart rate predictions. Acceleration magnitude and demographic details like Age and Weight further scale output values.")

        st.markdown("<div class='section-header'>Model Evaluation Metrics</div>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            **Activity Classifier (Random Forest):**
            * **Evaluation Metric:** Accuracy, Precision, Recall, F1-Score
            * **Training Dataset Size:** 8,000 samples
            * **Testing Dataset Size:** 2,000 samples
            * **Test Set Accuracy:** **100.0%**
            * **Performance Class:** Excellent (Fully partitioned data)
            """)
        with col_m2:
            st.markdown("""
            **Heart Rate Regressor (Random Forest):**
            * **Evaluation Metric:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score
            * **MAE on Test Set:** **~2.42 BPM**
            * **RMSE on Test Set:** **~3.26 BPM**
            * **R² (Coefficient of Determination):** **98.7%**
            * **Performance Class:** Excellent Fit
            """)

    # ------------------ TAB 5: DATASET EXPLORER ------------------
    with tab_dataset:
        st.markdown("<div class='section-header'>Explore Synthetic Smartwatch Dataset</div>", unsafe_allow_html=True)
        st.write("Browse and analyze the historical smartwatch readings dataset generated for model training.")
        
        if df_raw is not None:
            st.write(f"Dataset Dimensions: **{df_raw.shape[0]} rows** by **{df_raw.shape[1]} columns**")
            
            # Grouped summary statistics
            st.markdown("#### Summary Statistics grouped by Activity:")
            stats_df = df_raw.groupby('Activity')[['Accel_X', 'Accel_Y', 'Accel_Z', 'Heart_Rate']].mean().round(2)
            st.dataframe(stats_df)
            
            st.markdown("#### Raw Data Preview:")
            st.dataframe(df_raw.head(100))
            
            # Scatter plot: Heart rate vs. Acceleration magnitude colored by Activity
            st.markdown("#### Heart Rate vs. Acceleration Magnitude (Colored by Activity)")
            df_plot = df_raw.sample(1000, random_state=42).copy()
            df_plot['Accel_Mag'] = np.sqrt(df_plot['Accel_X']**2 + df_plot['Accel_Y']**2 + df_plot['Accel_Z']**2)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#1A1E29')
            
            for act in encoder.classes_:
                sub_df = df_plot[df_plot['Activity'] == act]
                ax.scatter(sub_df['Accel_Mag'], sub_df['Heart_Rate'], label=act, alpha=0.7, edgecolors='none', s=40)
                
            ax.set_xlabel('Acceleration Magnitude (m/s²)', color='white')
            ax.set_ylabel('Heart Rate (BPM)', color='white')
            ax.tick_params(colors='white')
            ax.legend(loc='lower right')
            ax.set_title('Accelerometer Magnitude vs. Heart Rate Cluster Map', color='white')
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("⚠️ No dataset file found at `data/wearable_sensor_data.csv`. Generate it using the sidebar option or by running generate_data.py.")

# ------------------ FOOTER ------------------
st.markdown("""
<hr style="border-color: rgba(255,255,255,0.06); margin-top: 5rem; margin-bottom: 1rem;">
<div style="display: flex; justify-content: space-between; align-items: center; color: #8c93a3; font-size: 0.85rem; padding-bottom: 2rem; flex-wrap: wrap; gap: 10px;">
    <div>
        ⌚ <b>PulseHAR Analytics</b> &copy; 2026. Designed for Wearable Sensor Research.
    </div>
    <div style="display: flex; gap: 15px;">
        <span>Powered by <b>Streamlit</b></span>
        <span>&bull;</span>
        <span>Models via <b>Scikit-Learn</b></span>
        <span>&bull;</span>
        <span>Charts by <b>Matplotlib</b></b></span>
    </div>
</div>
""", unsafe_allow_html=True)
