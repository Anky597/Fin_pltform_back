import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import traceback # For detailed error logging
import sys # For potentially writing critical errors to stderr

# --- Configuration ---
# Ensure this matches the filename of your saved stacking pipeline
MODEL_FILENAME = 'stacking_loan_approval_pipeline.joblib'
# Creates an absolute path relative to this script file's location
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

# --- Global Variables ---
pipeline = None
expected_columns = None # Will be populated after loading the model
model_loaded_successfully = False # Flag to track loading status

# --- Flask App Initialization ---
# This 'app' object will be imported by the WSGI configuration file
app = Flask(__name__)

# --- Helper Function to Load Model and Get Expected Columns ---
def load_model_and_metadata():
    """Loads the pipeline and extracts expected column order."""
    global pipeline, expected_columns, model_loaded_successfully # Declare globals
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}", file=sys.stderr)
        pipeline = None
        expected_columns = None
        model_loaded_successfully = False
        return # Stop execution of this function

    try:
        print(f"Attempting to load model from: {MODEL_PATH}")
        pipeline = joblib.load(MODEL_PATH)
        print("Pipeline loaded successfully.")

        # --- Extract expected feature names IN ORDER from the preprocessor ---
        # Access the preprocessor step within the main pipeline
        preprocessor = pipeline.named_steps.get('preprocessor')
        if preprocessor is None:
             raise ValueError("Preprocessor step named 'preprocessor' not found in the pipeline.")

        cols = []
        # Iterate through transformers defined in ColumnTransformer
        for name, transformer_obj, columns_list in preprocessor.transformers_:
            if transformer_obj == 'drop' or transformer_obj == 'passthrough':
                 continue # Skip non-feature generating steps here
            # Add the original columns associated with this transformer
            # Ensure columns_list is actually a list of strings
            if isinstance(columns_list, list) and all(isinstance(col, str) for col in columns_list):
                cols.extend(columns_list)
            else:
                 print(f"Warning: Transformer '{name}' associated columns are not a list of strings: {columns_list}", file=sys.stderr)


        # Simple de-duplication while preserving order
        seen = set()
        expected_columns = [x for x in cols if x not in seen and not seen.add(x)] # Slightly safer de-duplication

        if not expected_columns:
             raise ValueError("Could not extract any expected columns from the preprocessor.")

        print(f"Model expects columns (order matters): {expected_columns}")
        print(f"Number of expected columns: {len(expected_columns)}")
        model_loaded_successfully = True # Set flag on success

    except Exception as e:
        print(f"CRITICAL ERROR loading model or extracting columns: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr) # Log full traceback
        pipeline = None
        expected_columns = None
        model_loaded_successfully = False

# --- Load the model immediately when the script is imported ---
# This happens once when the WSGI server starts your web app worker
print("Executing load_model_and_metadata() at script import...")
load_model_and_metadata() # Call the function to load model and set globals
if not model_loaded_successfully:
    print("Model loading failed at import. The application might not function correctly.", file=sys.stderr)

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    print("Received request for /predict") # Log endpoint hits
    if not model_loaded_successfully or pipeline is None or expected_columns is None:
        print("Error: Model not available or failed to load.", file=sys.stderr)
        # Return 503 Service Unavailable if model isn't ready
        return jsonify({'error': 'Model service is not available, cannot make predictions.', 'status': 'error'}), 503

    try:
        # 1. Get data from request
        input_data = request.get_json()
        if not input_data:
            print("Error: No JSON data received.", file=sys.stderr)
            return jsonify({'error': 'No input JSON data provided.', 'status': 'error'}), 400

        # 2. Validate required keys
        missing_keys = [key for key in expected_columns if key not in input_data]
        if missing_keys:
             print(f"Error: Missing fields: {missing_keys}", file=sys.stderr)
             return jsonify({
                 'error': f'Missing required fields: {", ".join(missing_keys)}',
                 'status': 'error',
                 'expected_fields': expected_columns # Help the user debug
                 }), 400

        # 3. Convert to Pandas DataFrame with correct columns and order
        try:
             # Use only the expected columns in the correct order
             data_for_df = {key: [input_data[key]] for key in expected_columns}
             sample_df = pd.DataFrame(data_for_df)
             # Reorder columns to ensure they match the training order exactly
             sample_df = sample_df[expected_columns]
        except Exception as e:
             print(f"Error creating DataFrame from input: {e}", file=sys.stderr)
             print(traceback.format_exc(), file=sys.stderr)
             return jsonify({'error': 'Error processing input data into DataFrame.', 'status': 'error'}), 400

        print(f"DataFrame prepared for prediction:\n{sample_df.to_string()}")

        # 4. Make Prediction using the loaded pipeline
        prediction_raw = pipeline.predict(sample_df)
        prediction_proba = pipeline.predict_proba(sample_df)

        # 5. Format Response
        prediction_label = 'Approved' if prediction_raw[0] == 1 else 'Rejected'

        # Get probability indices reliably based on stored classes
        try:
            class_index_rejected = np.where(pipeline.classes_ == 0)[0][0]
            class_index_approved = np.where(pipeline.classes_ == 1)[0][0]
            probability_rejected = prediction_proba[0][class_index_rejected]
            probability_approved = prediction_proba[0][class_index_approved]
        except (AttributeError, IndexError, ValueError) as e:
             print(f"Error accessing probabilities using classes_: {e}. Falling back to fixed indices.", file=sys.stderr)
             # Fallback (assumes standard 0, 1 order if classes_ attribute fails)
             probability_rejected = prediction_proba[0][0]
             probability_approved = prediction_proba[0][1]


        print(f"Prediction successful: {prediction_label}")

        response = {
            'prediction': prediction_label,
            'probability_rejected': round(float(probability_rejected), 4), # Ensure native float
            'probability_approved': round(float(probability_approved), 4), # Ensure native float
            'status': 'success'
        }
        return jsonify(response), 200

    except Exception as e:
        print(f"An unhandled error occurred during prediction: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr) # Log the full error traceback
        return jsonify({'error': 'An internal server error occurred during prediction.', 'status': 'error'}), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    if model_loaded_successfully:
        return jsonify({'status': 'ok', 'message': 'Model loaded successfully.'}), 200
    else:
        # Return 503 Service Unavailable if model loading failed
        return jsonify({'status': 'error', 'message': 'Model is not loaded or failed to load.'}), 503

# --- Basic Root Route (Optional) ---
@app.route('/')
def home():
     # Simple HTML response to show the app is alive
     status_message = "ready (model loaded)" if model_loaded_successfully else "error (model not loaded)"
     return f"<h1>Loan Approval API</h1><p>Status: {status_message}. Use the /predict endpoint for predictions.</p>", 200

# --- NO `if __name__ == '__main__':` block ---
# The WSGI server (like Gunicorn used by Render/PythonAnywhere) will import the 'app' object directly.