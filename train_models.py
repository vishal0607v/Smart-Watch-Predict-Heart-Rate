import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score

def train_and_save_models(data_path, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load data
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
    
    df = pd.read_csv(data_path)
    
    # 2. Encode Activity labels
    print("Encoding activity labels...")
    activity_encoder = LabelEncoder()
    df['Activity_Encoded'] = activity_encoder.fit_transform(df['Activity'])
    
    # Save the encoder for deployment
    encoder_path = os.path.join(models_dir, 'activity_encoder.pkl')
    joblib.dump(activity_encoder, encoder_path)
    print(f"Activity encoder saved to {encoder_path}")
    print(f"Classes: {activity_encoder.classes_}")
    
    # 3. Prepare features & targets
    # Classifier features: raw smartwatch sensor data
    classifier_features = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    X_clf = df[classifier_features]
    y_clf = df['Activity_Encoded']
    
    # Regressor features: raw smartwatch sensor data + demographic info + activity label
    regressor_features = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 
                          'Age', 'Weight', 'Gender', 'Activity_Encoded']
    X_reg = df[regressor_features]
    y_reg = df['Heart_Rate']
    
    # 4. Train Activity Classifier
    print("\nTraining Activity Classifier (Random Forest)...")
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # n_estimators=100 and max_depth=12 for fast training and avoidance of overfitting
    clf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf_model.fit(X_train_clf, y_train_clf)
    
    # Evaluate Classifier
    y_pred_clf = clf_model.predict(X_test_clf)
    clf_acc = accuracy_score(y_test_clf, y_pred_clf)
    print(f"Classifier Accuracy: {clf_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_clf, y_pred_clf, target_names=activity_encoder.classes_))
    
    # Save Classifier Model
    clf_model_path = os.path.join(models_dir, 'activity_classifier.pkl')
    joblib.dump(clf_model, clf_model_path)
    print(f"Classifier model saved to {clf_model_path}")
    
    # 5. Train Heart Rate Regressor
    print("\nTraining Heart Rate Regressor (Random Forest)...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    reg_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    reg_model.fit(X_train_reg, y_train_reg)
    
    # Evaluate Regressor
    y_pred_reg = reg_model.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"Regressor Mean Absolute Error (MAE): {mae:.4f} bpm")
    print(f"Regressor Root Mean Squared Error (RMSE): {rmse:.4f} bpm")
    print(f"Regressor R^2 Score: {r2:.4f}")
    
    # Save Regressor Model
    reg_model_path = os.path.join(models_dir, 'heart_rate_regressor.pkl')
    joblib.dump(reg_model, reg_model_path)
    print(f"Regressor model saved to {reg_model_path}")
    
    # Save feature lists so app.py knows what features were used
    joblib.dump({
        'clf_features': classifier_features,
        'reg_features': regressor_features
    }, os.path.join(models_dir, 'model_features.pkl'))
    print("Model feature metadata saved.")

if __name__ == "__main__":
    train_and_save_models('data/wearable_sensor_data.csv', 'models')
