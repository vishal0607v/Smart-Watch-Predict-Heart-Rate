import os
import numpy as np
import pandas as pd

def generate_smartwatch_data(output_path, num_subjects=20, samples_per_activity=100):
    np.random.seed(42)
    
    # Define activities
    activities = ['Sitting', 'Standing', 'Walking', 'Running', 'Lying Down']
    
    # Create subjects with biometrics
    subjects = []
    for sub_id in range(1, num_subjects + 1):
        subjects.append({
            'subject_id': f'Sub_{sub_id:02d}',
            'age': int(np.random.randint(18, 65)),
            'weight': float(np.round(np.random.uniform(50, 95), 1)),
            'gender': int(np.random.choice([0, 1])), # 0 = Female, 1 = Male
            'resting_hr': float(np.random.randint(60, 76))
        })
    
    # 20 Hz sampling rate (dt = 0.05s)
    dt = 0.05
    data_records = []
    
    for sub in subjects:
        sub_id = sub['subject_id']
        age = sub['age']
        weight = sub['weight']
        gender = sub['gender']
        resting_hr = sub['resting_hr']
        
        for activity in activities:
            # Generate continuous-ish time index
            t = np.arange(samples_per_activity) * dt
            
            # Accelerometer (m/s^2) and Gyroscope (rad/s) generation based on activity
            if activity == 'Sitting':
                ax = np.random.normal(0.1, 0.05, samples_per_activity)
                ay = np.random.normal(0.2, 0.05, samples_per_activity)
                az = np.random.normal(9.8, 0.05, samples_per_activity)
                gx = np.random.normal(0.0, 0.01, samples_per_activity)
                gy = np.random.normal(0.0, 0.01, samples_per_activity)
                gz = np.random.normal(0.0, 0.01, samples_per_activity)
                
                # Heart rate: resting HR plus slight fluctuation
                hr = np.random.normal(resting_hr, 1.5, samples_per_activity)
                
            elif activity == 'Standing':
                ax = np.random.normal(0.3, 0.08, samples_per_activity)
                ay = np.random.normal(-0.2, 0.08, samples_per_activity)
                az = np.random.normal(9.7, 0.08, samples_per_activity)
                gx = np.random.normal(0.0, 0.02, samples_per_activity)
                gy = np.random.normal(0.0, 0.02, samples_per_activity)
                gz = np.random.normal(0.0, 0.02, samples_per_activity)
                
                # Heart rate: slightly higher than sitting
                hr = np.random.normal(resting_hr + 4, 2.0, samples_per_activity)
                
            elif activity == 'Walking':
                # Sine waves for stepping motion (freq = 1.6 Hz)
                freq = 1.6
                ax = 0.8 * np.sin(2 * np.pi * freq * t) + np.random.normal(0.2, 0.15, samples_per_activity)
                ay = 1.2 * np.cos(2 * np.pi * freq * t) + np.random.normal(-0.1, 0.15, samples_per_activity)
                az = 9.8 + 1.5 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.2, samples_per_activity)
                
                gx = 0.4 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05, samples_per_activity)
                gy = 0.6 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.05, samples_per_activity)
                gz = 0.3 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05, samples_per_activity)
                
                # Heart rate increases with age factor and weight
                hr_base = resting_hr + 28 + (weight * 0.05) - (age * 0.05)
                hr = np.random.normal(hr_base, 3.5, samples_per_activity)
                
            elif activity == 'Running':
                # Higher frequency and amplitude sine waves (freq = 2.8 Hz)
                freq = 2.8
                ax = 2.5 * np.sin(2 * np.pi * freq * t) + np.random.normal(0.5, 0.4, samples_per_activity)
                ay = 4.0 * np.cos(2 * np.pi * freq * t) + np.random.normal(-0.3, 0.4, samples_per_activity)
                az = 9.8 + 6.0 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.5, samples_per_activity)
                
                gx = 1.5 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.15, samples_per_activity)
                gy = 2.0 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.15, samples_per_activity)
                gz = 1.0 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.15, samples_per_activity)
                
                # Heart rate spikes during running
                hr_base = resting_hr + 68 + (weight * 0.1) - (age * 0.1)
                hr = np.random.normal(hr_base, 5.0, samples_per_activity)
                
            elif activity == 'Lying Down':
                # Gravity aligned mostly with X-axis (lying on side/back)
                ax = np.random.normal(9.6, 0.04, samples_per_activity)
                ay = np.random.normal(0.5, 0.04, samples_per_activity)
                az = np.random.normal(0.8, 0.04, samples_per_activity)
                gx = np.random.normal(0.0, 0.005, samples_per_activity)
                gy = np.random.normal(0.0, 0.005, samples_per_activity)
                gz = np.random.normal(0.0, 0.005, samples_per_activity)
                
                # Heart rate drops below resting HR
                hr = np.random.normal(resting_hr - 4, 1.2, samples_per_activity)
            
            # Append records
            for j in range(samples_per_activity):
                data_records.append({
                    'Subject_ID': sub_id,
                    'Age': age,
                    'Weight': weight,
                    'Gender': gender,
                    'Accel_X': float(ax[j]),
                    'Accel_Y': float(ay[j]),
                    'Accel_Z': float(az[j]),
                    'Gyro_X': float(gx[j]),
                    'Gyro_Y': float(gy[j]),
                    'Gyro_Z': float(gz[j]),
                    'Activity': activity,
                    'Heart_Rate': float(hr[j])
                })
                
    df = pd.DataFrame(data_records)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully with {len(df)} records at {output_path}")

if __name__ == "__main__":
    generate_smartwatch_data('data/wearable_sensor_data.csv')
