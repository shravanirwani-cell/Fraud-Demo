from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
import google.genai as genai

app = Flask(__name__)
CORS(app)
warnings.filterwarnings('ignore')

client = genai.Client(api_key="AIzaSyCj0KCwlwYBRbIMc739yrrx83thu-I2fTk")
MODEL_PATH = 'fraud_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURE_COLS_PATH = 'feature_cols.pkl'

def preprocess_data(df):
    """Fixed preprocessing - always 25 features"""
    df_processed = df.copy()
    
    merchants = ['Walmart','Amazon','Target','BestBuy','GasStation','HotelChain','OnlineStore']
    categories = ['groceries','electronics','travel','clothing','fuel','dining','online']
    
    for m in merchants:
        df_processed[f'merchant_{m}'] = (df_processed['merchant'] == m).astype(int)
    for c in categories:
        df_processed[f'category_{c}'] = (df_processed['category'] == c).astype(int)
    
    feature_cols = ['amount','time_of_day','day_of_week','location_lat','location_long',
                   'user_age','user_account_age_days','transaction_count_last_30d','avg_transaction_amount'] + \
                  [f'merchant_{m}' for m in merchants] + \
                  [f'category_{c}' for c in categories]
    
    X = df_processed[feature_cols].fillna(0)
    return X, df_processed['is_fraud'], feature_cols

@app.route('/train-from-csv', methods=['POST'])
def train_from_csv():
    """Train model from UPLOADED CSV"""
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        print(f"📊 Training from {len(df)} transactions")
        
        X, y, feature_cols = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        # Save model
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(list(feature_cols), FEATURE_COLS_PATH)
        
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return jsonify({
            'status': 'success',
            'dataset_size': len(df),
            'fraud_rate': f"{df['is_fraud'].mean():.2%}",
            'accuracy': f"{report['accuracy']:.3f}",
            'features': len(feature_cols)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
