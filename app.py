from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
import warnings
from flask import render_template
from dotenv import load_dotenv
load_dotenv()




# ------------------ Flask setup ------------------

app = Flask(__name__)
CORS(app)
warnings.filterwarnings('ignore')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        # Use existing CSV in repo
        df = pd.read_csv("transactions_1000.csv")

        X, y, feat_cols = preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler_local = StandardScaler()
        X_train_scaled = scaler_local.fit_transform(X_train)
        X_test_scaled = scaler_local.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train)

        report = classification_report(
            y_test, clf.predict(X_test_scaled), output_dict=True
        )

        joblib.dump(clf, MODEL_PATH)
        joblib.dump(scaler_local, SCALER_PATH)
        joblib.dump(feat_cols, FEATURE_COLS_PATH)

        global model, scaler, feature_cols
        model, scaler, feature_cols = clf, scaler_local, feat_cols

        return jsonify({
            "status": "success",
            "features": len(feat_cols),
            "fraud_rate": f"{float(df['is_fraud'].mean())*100:.2f}%",
            "accuracy": report["accuracy"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ Paths & globals ------------------

MODEL_PATH = 'fraud_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURE_COLS_PATH = 'feature_cols.pkl'

model = None
scaler = None
feature_cols = None


# ------------------ Preprocessing ------------------

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess transaction dataframe:
    - one-hot encode merchant & category (fixed lists)
    - return X, y, feature_cols
    """
    df_processed = df.copy()

    merchants = ['Walmart', 'Amazon', 'Target', 'BestBuy',
                 'GasStation', 'HotelChain', 'OnlineStore']
    categories = ['groceries', 'electronics', 'travel',
                  'clothing', 'fuel', 'dining', 'online']

    # one‑hot with fixed columns
    for m in merchants:
        df_processed[f'merchant_{m}'] = (df_processed['merchant'] == m).astype(int)
    for c in categories:
        df_processed[f'category_{c}'] = (df_processed['category'] == c).astype(int)

    numeric_cols = [
        'amount', 'time_of_day', 'day_of_week', 'location_lat',
        'location_long', 'user_age', 'user_account_age_days',
        'transaction_count_last_30d', 'avg_transaction_amount'
    ]

    feature_cols = numeric_cols + \
        [f'merchant_{m}' for m in merchants] + \
        [f'category_{c}' for c in categories]

    X = df_processed[feature_cols].fillna(0)
    y = df_processed['is_fraud'].astype(int)

    return X, y, feature_cols



def get_fraud_reasons(transaction_data: dict, prob: float) -> str:
    reasons = []

    amt = transaction_data.get("amount", 0)
    avg_amt = transaction_data.get("avg_transaction_amount", 1)
    acct_age = transaction_data.get("user_account_age_days", 0)
    txn_count = transaction_data.get("transaction_count_last_30d", 0)

    if amt > 3 * avg_amt:
        reasons.append("Transaction amount is significantly higher than user's usual spending.")

    if acct_age < 30:
        reasons.append("User account is relatively new.")

    if txn_count > 30:
        reasons.append("High transaction frequency observed in recent period.")

    if not reasons:
        reasons.append("Transaction deviates from typical behavioral patterns.")

    return "\n".join(f"• {r}" for r in reasons)





def load_model():
    global model, scaler, feature_cols
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

# ------------------ Routes ------------------

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_ready': os.path.exists(MODEL_PATH)
    })

# ---- Train from uploaded CSV ----

@app.route('/api/train-from-csv', methods=['POST'])
def train_from_csv():
    """
    Upload a CSV with the columns you used before (including is_fraud)
    and train / overwrite the model.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        if 'is_fraud' not in df.columns:
            return jsonify({'error': 'CSV must contain is_fraud column'}), 400

        X, y, feat_cols = preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler_local = StandardScaler()
        X_train_scaled = scaler_local.fit_transform(X_train)
        X_test_scaled = scaler_local.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train)

        # evaluate
        report = classification_report(
            y_test, clf.predict(X_test_scaled), output_dict=True
        )

        # persist
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(scaler_local, SCALER_PATH)
        joblib.dump(feat_cols, FEATURE_COLS_PATH)

        # refresh globals
        global model, scaler, feature_cols
        model, scaler, feature_cols = clf, scaler_local, feat_cols

        return jsonify({
            'status': 'success',
            'dataset_size': int(len(df)),
            'fraud_rate': f"{float(df['is_fraud'].mean())*100:.2f}%",
            'accuracy': f"{report['accuracy']:.3f}",
            'features': len(feat_cols)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---- Single prediction ----

@app.route('/api/predict', methods=['POST'])
def predict():
    global model, scaler, feature_cols
    if model is None:
        load_model()

    data = request.json or {}
    X_pred = np.zeros((1, len(feature_cols)))

    numeric_features = [
        'amount', 'time_of_day', 'day_of_week', 'location_lat',
        'location_long', 'user_age', 'user_account_age_days',
        'transaction_count_last_30d', 'avg_transaction_amount'
    ]

    for i, feature in enumerate(feature_cols):
        if feature in numeric_features:
            X_pred[0, i] = data.get(feature, 0)

    X_scaled = scaler.transform(X_pred)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    explanation = "Legitimate transaction" if pred == 0 else \
        get_fraud_reasons(data, prob)

    return jsonify({
        'is_fraud': bool(pred),
        'fraud_probability': float(prob),
        'confidence': float(max(model.predict_proba(X_scaled)[0])),
        'risk_score': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW',
        'explanation': explanation
    })

# ---- Batch analyze uploaded CSV ----

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    global model, scaler, feature_cols
    if model is None:
        load_model()

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    X, _, _ = preprocess_data(df)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    fraud_mask = predictions == 1
    total_fraud = int(fraud_mask.sum())
    fraud_rate = total_fraud / len(df) * 100

    fraud_details = []
    # limit explanations to first 20 frauds to save tokens/time
    for idx in df[fraud_mask].index[:20]:
        row_dict = df.iloc[idx].to_dict()
        row_dict['fraud_probability'] = float(probabilities[idx])
        row_dict['risk_score'] = 'HIGH' if probabilities[idx] > 0.7 else 'MEDIUM' if probabilities[idx] > 0.3 else 'LOW'
        row_dict['gemini_explanation'] = get_fraud_reasons(row_dict, probabilities[idx])
        fraud_details.append(row_dict)

    return jsonify({
        'total_transactions': int(len(df)),
        'total_fraud': total_fraud,
        'fraud_rate': f"{fraud_rate:.2f}%",
        'fraud_details': fraud_details,
        'summary': f"{total_fraud} frauds detected out of {len(df)} ({fraud_rate:.2f}% fraud rate) with Gemini explanations for first {len(fraud_details)}."
    })

# ---- Batch download fraud report ----

@app.route('/api/batch-download', methods=['POST'])
def batch_download():
    global model, scaler, feature_cols
    if model is None:
        load_model()

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    X, _, _ = preprocess_data(df)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df['predicted_fraud'] = predictions
    df['fraud_probability'] = probabilities
    df['risk_score'] = np.where(
        probabilities > 0.7, 'HIGH',
        np.where(probabilities > 0.3, 'MEDIUM', 'LOW')
    )

    fraud_report = df[df['predicted_fraud'] == 1].copy()
    filename = 'fraud_report.csv'
    fraud_report.to_csv(filename, index=False)

    return send_file(filename, as_attachment=True, download_name='fraud_report.csv')


# ------------------ main ------------------

if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )


