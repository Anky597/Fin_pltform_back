# --- Flask API for Loan Approval & Segmentation (Render Deployment Version) ---

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys
import warnings

# Suppress specific warnings if needed
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---

# --- Assume Models are in the SAME directory as this script (relative paths) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Loan Approval Model ---
LOAN_MODEL_FILENAME = 'stacking_loan_approval_pipeline.joblib'
LOAN_MODEL_PATH = os.path.join(BASE_DIR, LOAN_MODEL_FILENAME)

# --- Segmentation Models ---
SCALER_FILENAME = 'segmentation_scaler.joblib'
KMEANS_FILENAME = 'segmentation_kmeans_model.joblib'
DBSCAN_FILENAME = 'segmentation_dbscan_model.joblib'

SCALER_PATH = os.path.join(BASE_DIR, SCALER_FILENAME)
KMEANS_PATH = os.path.join(BASE_DIR, KMEANS_FILENAME)
DBSCAN_PATH = os.path.join(BASE_DIR, DBSCAN_FILENAME)

# Features for segmentation (MUST match segmentation training)
SEGMENTATION_FEATURES = [
    'income_annum',
    'loan_amount',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
]

# --- Global Variables ---
loan_pipeline = None
loan_expected_columns = None
loan_model_loaded_successfully = False
segmentation_scaler = None
kmeans_model = None
dbscan_model = None
segmentation_models_loaded = False

# --- Flask App Initialization ---
app = Flask(__name__)
# Initialize CORS - For Render, you might restrict origins later in settings
# Or use environment variables for allowed origins
CORS(app)

# --- Optional: Define Segment Names (Customize!) ---
kmeans_segment_names = {
    0: "Low Value",
    1: "Mid Income/Loan",
}
dbscan_segment_names = {
    -1: "Noise/Outlier",
     0: "Dense Core 1",
}

# --- Helper Function to Load LOAN APPROVAL Model ---
def load_loan_model_and_metadata():
    """Loads the loan pipeline and extracts expected column order."""
    global loan_pipeline, loan_expected_columns, loan_model_loaded_successfully
    print(f"--- Loading Loan Approval Model ---")
    if not os.path.exists(LOAN_MODEL_PATH):
        print(f"CRITICAL ERROR: Loan Model file not found at {LOAN_MODEL_PATH}", file=sys.stderr)
        loan_model_loaded_successfully = False
        return
    try:
        print(f"Attempting to load loan model from: {LOAN_MODEL_PATH}")
        loan_pipeline = joblib.load(LOAN_MODEL_PATH)
        print("Loan Pipeline loaded successfully.")
        # Extract expected features (ensure this logic works for your specific pipeline)
        preprocessor = loan_pipeline.named_steps.get('preprocessor')
        if preprocessor is None: raise ValueError("Preprocessor step 'preprocessor' not found.")
        cols = []
        transformers = getattr(preprocessor, 'transformers_', None)
        if transformers:
            for name, _, columns_list in transformers:
                 if isinstance(columns_list, list) and all(isinstance(c, str) for c in columns_list): cols.extend(columns_list)
                 elif isinstance(columns_list, str): cols.append(columns_list)
        else: raise ValueError("Could not find 'transformers_' attribute.")
        seen = set()
        loan_expected_columns = [x for x in cols if x not in seen and not seen.add(x)]
        if not loan_expected_columns: raise ValueError("Could not extract expected columns.")
        print(f"Loan Model expects columns: {loan_expected_columns}")
        loan_model_loaded_successfully = True
    except Exception as e:
        print(f"CRITICAL ERROR loading loan model: {e}", file=sys.stderr); print(traceback.format_exc(), file=sys.stderr)
        loan_pipeline, loan_expected_columns, loan_model_loaded_successfully = None, None, False

# --- Helper Function to Load SEGMENTATION Models ---
def load_segmentation_models():
    """Loads the segmentation models."""
    global segmentation_scaler, kmeans_model, dbscan_model, segmentation_models_loaded
    print(f"--- Loading Segmentation Models ---")
    scaler_ok, kmeans_ok, dbscan_ok = False, False, False
    try:
        if os.path.exists(SCALER_PATH): segmentation_scaler = joblib.load(SCALER_PATH); print(f"Scaler loaded."); scaler_ok = True
        else: print(f"ERROR: Scaler not found at {SCALER_PATH}", file=sys.stderr)
        if os.path.exists(KMEANS_PATH): kmeans_model = joblib.load(KMEANS_PATH); print(f"KMeans loaded."); kmeans_ok = True
        else: print(f"ERROR: KMeans not found at {KMEANS_PATH}", file=sys.stderr)
        if os.path.exists(DBSCAN_PATH): dbscan_model = joblib.load(DBSCAN_PATH); print(f"DBSCAN loaded."); dbscan_ok = True
        else: print(f"ERROR: DBSCAN not found at {DBSCAN_PATH}", file=sys.stderr)
        segmentation_models_loaded = scaler_ok and (kmeans_ok or dbscan_ok) # Need scaler and at least one model
    except Exception as e:
        print(f"ERROR loading segmentation models: {e}", file=sys.stderr); print(traceback.format_exc(), file=sys.stderr)
        segmentation_scaler, kmeans_model, dbscan_model, segmentation_models_loaded = None, None, None, False
    if not segmentation_models_loaded: print("Segmentation endpoint will not function.", file=sys.stderr)

# --- Load ALL models when the application starts ---
# Gunicorn will typically run this loading process for each worker it starts
print("--- Executing Model Loading at Startup ---")
load_loan_model_and_metadata()
load_segmentation_models()
print("--- Model Loading Attempt Complete ---")


# --- API Endpoint for LOAN APPROVAL Prediction (Functionally Unchanged) ---
@app.route('/predict', methods=['POST'])
def predict():
    """Handles loan approval predictions."""
    print("\n--- Request received for /predict (Loan Approval) ---") # Log request
    if not loan_model_loaded_successfully or not loan_pipeline or not loan_expected_columns:
        return jsonify({'error': 'Loan Approval service unavailable.', 'status': 'error'}), 503
    try:
        input_data = request.get_json()
        if not input_data: return jsonify({'error': 'No input JSON data provided.', 'status': 'error'}), 400
        # print(f"Received loan input data: {input_data}") # Sensitive - log carefully

        missing_keys = [key for key in loan_expected_columns if key not in input_data]
        if missing_keys:
            return jsonify({'error': f'Missing fields: {", ".join(missing_keys)}', 'status': 'error', 'expected': loan_expected_columns}), 400

        data_for_df = {key: [input_data[key]] for key in loan_expected_columns}
        sample_df = pd.DataFrame(data_for_df)[loan_expected_columns] # Ensure order

        prediction_raw = loan_pipeline.predict(sample_df)
        prediction_proba = loan_pipeline.predict_proba(sample_df)
        prediction_label = 'Approved' if prediction_raw[0] == 1 else 'Rejected'

        # Safely extract probabilities
        prob_rej, prob_app = 0.0, 0.0
        try:
            classes = getattr(loan_pipeline, 'classes_', [0, 1])
            idx_rej = np.where(classes == 0)[0][0]
            idx_app = np.where(classes == 1)[0][0]
            prob_rej = prediction_proba[0][idx_rej]
            prob_app = prediction_proba[0][idx_app]
        except Exception as e:
             print(f"Warning: Prob extraction failed: {e}. Using defaults.", file=sys.stderr)
             prob_rej = prediction_proba[0][0]
             prob_app = prediction_proba[0][1]

        print(f"Loan Prediction successful: {prediction_label}")
        return jsonify({
            'prediction': prediction_label,
            'probability_rejected': round(float(prob_rej), 4),
            'probability_approved': round(float(prob_app), 4),
            'status': 'success'
        }), 200
    except Exception as e:
        print(f"ERROR during loan prediction: {e}", file=sys.stderr); print(traceback.format_exc(), file=sys.stderr)
        return jsonify({'error': 'Internal server error during loan prediction.', 'status': 'error'}), 500


# --- API Endpoint for SEGMENTATION Prediction (Functionally Unchanged) ---
@app.route('/segment', methods=['POST'])
def segment():
    """Handles customer segmentation predictions."""
    print("\n--- Request received for /segment (Customer Segmentation) ---") # Log request
    if not segmentation_models_loaded or not segmentation_scaler:
        return jsonify({'error': 'Segmentation service unavailable.', 'status': 'error'}), 503
    if not kmeans_model and not dbscan_model: # Check if at least one clustering model is loaded
        return jsonify({'error': 'Segmentation service unavailable (no clustering models).', 'status': 'error'}), 503

    try:
        input_data = request.get_json()
        if not input_data: return jsonify({'error': 'No input JSON data provided.', 'status': 'error'}), 400
        # print(f"Received segmentation input data: {input_data}") # Sensitive - log carefully

        missing_keys = [key for key in SEGMENTATION_FEATURES if key not in input_data]
        if missing_keys:
            return jsonify({'error': f'Missing fields: {", ".join(missing_keys)}', 'status': 'error', 'expected': SEGMENTATION_FEATURES}), 400

        data_for_df = {key: [input_data[key]] for key in SEGMENTATION_FEATURES}
        segment_df = pd.DataFrame(data_for_df)
        # Check numeric and convert
        non_numeric = [col for col in SEGMENTATION_FEATURES if not pd.to_numeric(segment_df[col], errors='coerce').notna().all()]
        if non_numeric: return jsonify({"error": f"Features must be numeric: {', '.join(non_numeric)}"}), 400
        segment_df = segment_df.astype(float)[SEGMENTATION_FEATURES] # Ensure type and order

        # Preprocess: Clean assets -> Scale
        asset_cols = [col for col in SEGMENTATION_FEATURES if 'assets' in col]
        for col in asset_cols: segment_df[col] = segment_df[col].clip(lower=0)
        X_scaled = segmentation_scaler.transform(segment_df)
        # print(f"Scaled data for segmentation: {X_scaled}") # Debug only

        results = {"input_data": input_data}
        # K-Means Prediction
        if kmeans_model:
            try:
                pred = kmeans_model.predict(X_scaled)
                label = int(pred[0])
                results['kmeans_segment_pred'] = label
                results['kmeans_segment_name'] = kmeans_segment_names.get(label, f"Unknown KMeans ({label})")
            except Exception as e: print(f"ERROR K-Means seg predict: {e}", file=sys.stderr); results['kmeans_prediction_error'] = str(e)
        else: results['kmeans_segment_pred'], results['kmeans_segment_name'] = None, "N/A"
        # DBSCAN Prediction (using fit_predict caveat)
        if dbscan_model:
            try:
                pred = dbscan_model.fit_predict(X_scaled)
                label = int(pred[0])
                results['dbscan_segment_pred'] = label
                results['dbscan_segment_name'] = dbscan_segment_names.get(label, f"Unknown DBSCAN ({label})")
            except Exception as e: print(f"ERROR DBSCAN seg predict: {e}", file=sys.stderr); results['dbscan_prediction_error'] = str(e)
        else: results['dbscan_segment_pred'], results['dbscan_segment_name'] = None, "N/A"

        print("Segmentation prediction successful.")
        results['status'] = 'success'
        return jsonify(results), 200

    except Exception as e:
        print(f"ERROR during segmentation: {e}", file=sys.stderr); print(traceback.format_exc(), file=sys.stderr)
        return jsonify({'error': 'Internal server error during segmentation.', 'status': 'error'}), 500


# --- Health Check Endpoint (Functionally Unchanged) ---
@app.route('/health', methods=['GET'])
def health_check():
    loan_status = 'ok' if loan_model_loaded_successfully else 'error'
    segmentation_status = 'ok' if segmentation_models_loaded else 'error'
    overall_status_code = 200 if loan_status == 'ok' and segmentation_status == 'ok' else 503
    message = "OK" if overall_status_code == 200 else "Service Unavailable (Model Loading Issue)"
    return jsonify({
        'status': 'ok' if overall_status_code == 200 else 'error', 'message': message,
        'services': {
            'loan_approval': {'status': loan_status, 'loaded': loan_model_loaded_successfully},
            'segmentation': {'status': segmentation_status, 'loaded': segmentation_models_loaded,
                             'scaler': segmentation_scaler is not None, 'kmeans': kmeans_model is not None, 'dbscan': dbscan_model is not None}
        }}), overall_status_code

# --- Basic Root Route (Functionally Unchanged) ---
@app.route('/')
def home():
     loan_msg = "Loan OK" if loan_model_loaded_successfully else "Loan FAILED"
     seg_msg = "Segmentation OK" if segmentation_models_loaded else "Segmentation FAILED/Partial"
     return f"<h1>API Ready</h1><p>Status: {loan_msg}; {seg_msg}.</p><p>Endpoints: POST /predict, POST /segment, GET /health</p>", 200

# --- Remove the __main__ block that calls app.run() ---
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5001))
#     print(f"--- Starting Flask Development Server on http://0.0.0.0:{port} ---")
#     app.run(host='0.0.0.0', port=port, debug=True) # DEBUG should be False for production
