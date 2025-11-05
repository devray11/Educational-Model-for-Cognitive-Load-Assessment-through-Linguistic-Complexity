from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Define the 6-level classification labels
NEW_LABELS = [
    'Grade till 6th',
    'Grade 6-8',
    'Grade 9-10',
    'Grade 11-12',
    'Undergraduate',
    'Postgraduate'
]

# Global variables for models
classifier = None
model_loaded = False
MODEL_IMPORT_SUCCESS = False

# Try to import the Model components with error handling
try:
    from Model import AdvancedTextClassifier
    MODEL_IMPORT_SUCCESS = True
    logger.info("Model import successful")
except Exception as e:
    logger.error(f"Failed to import Model components: {e}")
    MODEL_IMPORT_SUCCESS = False

def initialize_models():
    """Initialize and load the trained models"""
    global classifier, model_loaded
    
    if not MODEL_IMPORT_SUCCESS:
        logger.error("Cannot initialize models due to import errors")
        model_loaded = False
        return
    
    try:
        classifier = AdvancedTextClassifier()
        
        # Try to load existing model
        model_path = 'cognitive_load_model.pkl'
        if os.path.exists(model_path):
            try:
                classifier.load_model(model_path)
                model_loaded = True
                logger.info("Model loaded successfully from existing file")
            except Exception as e:
                logger.error(f"Failed to load existing model: {e}")
                model_loaded = False
        else:
            # Train new model if file doesn't exist
            logger.info("Model file not found. Training new model...")
            
            # Check for the correct dataset filename - prioritize Book_Dataset.csv
            dataset_files = [
                'Book_Dataset.csv',
                'book_dataset.csv',
                'educational_text_corpus.csv'
            ]
            
            dataset_found = None
            for dataset_file in dataset_files:
                if os.path.exists(dataset_file):
                    dataset_found = dataset_file
                    break
            
            if dataset_found:
                try:
                    logger.info(f"Found dataset: {dataset_found}")
                    classifier.train(dataset_found)
                    classifier.save_model(model_path)
                    model_loaded = True
                    logger.info("New model trained and saved successfully")
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    model_loaded = False
            else:
                logger.error(f"No dataset file found. Checked: {dataset_files}")
                model_loaded = False
                
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        model_loaded = False

@app.route('/')
def index():
    """Serve the main HTML page (Index.html)"""
    try:
        with open('Index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "<h1>Error: Index.html not found</h1><p>Please ensure Index.html is in the same directory as App.py</p>", 404
    except Exception as e:
        return f"<h1>Error loading Index.html</h1><p>{str(e)}</p>", 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_import_success': MODEL_IMPORT_SUCCESS,
        'available_labels': NEW_LABELS,
        'dataset_files_checked': ['Book_Dataset.csv', 'book_dataset.csv', 'educational_text_corpus.csv']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with enhanced response"""
    global classifier, model_loaded
    
    if not MODEL_IMPORT_SUCCESS:
        return jsonify({
            'error': 'Model import failed. Please check dependencies and installation.',
            'model_loaded': False,
            'details': 'Failed to import required modules. Check if all dependencies are installed.'
        }), 500
    
    if not model_loaded or classifier is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'model_loaded': False,
            'details': 'The ML model could not be loaded or trained. Check if Book_Dataset.csv exists.'
        }), 500
    
    try:
        # Get text input from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please include "text" field in JSON request.',
                'example': {'text': 'Your text to analyze here...'}
            }), 400
        
        input_text = data['text'].strip()
        
        if len(input_text) == 0:
            return jsonify({
                'error': 'Empty text provided. Please provide some text to analyze.'
            }), 400
        
        if len(input_text) < 10:
            return jsonify({
                'error': 'Text too short. Please provide at least 10 characters for meaningful analysis.'
            }), 400
        
        # Make prediction
        result = classifier.predict(input_text)
        
        # Enhanced response with all required data
        response = {
            'prediction': result['prediction'],
            'confidence': float(result['confidence']),
            'features': {
                'avg_sentence_length': float(result['features']['avg_sentence_length']),
                'type_token_ratio': float(result['features']['type_token_ratio']),
                'lexical_sophistication': float(result['features']['lexical_sophistication']),
                'word_concreteness': float(result['features']['word_concreteness']),
                'mean_dependency_distance': float(result['features']['mean_dependency_distance']),
                'subordination_ratio': float(result['features']['subordination_ratio']),
                'flesch_kincaid_grade_level': float(result['features']['flesch_kincaid_grade_level']),
                'smog_index': float(result['features']['smog_index']),
                'connective_frequency': float(result['features']['connective_frequency']),
                'referential_cohesion': float(result['features']['referential_cohesion'])
            },
            'probabilities': result['probabilities'],
            'text_stats': {
                'character_count': len(input_text),
                'word_count': len(input_text.split()),
                'sentence_count': max(1, len([s for s in input_text.split('.') if s.strip()]))
            },
            'success': True
        }
        
        logger.info(f"Prediction successful: {result['prediction']} (confidence: {result['confidence']:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False,
            'details': 'An internal error occurred during text analysis. Please check server logs.'
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Alternative endpoint name for backward compatibility"""
    return predict()

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_loaded': model_loaded,
        'model_import_success': MODEL_IMPORT_SUCCESS,
        'labels': NEW_LABELS,
        'label_count': len(NEW_LABELS),
        'features': [
            'avg_sentence_length',
            'type_token_ratio', 
            'lexical_sophistication',
            'word_concreteness',
            'mean_dependency_distance',
            'subordination_ratio',
            'flesch_kincaid_grade_level',
            'smog_index',
            'connective_frequency',
            'referential_cohesion'
        ],
        'feature_count': 10
    })

@app.route('/debug')
def debug_info():
    """Debug endpoint to check system status"""
    import sys
    import pandas as pd
    
    debug_data = {
        'python_version': sys.version,
        'model_import_success': MODEL_IMPORT_SUCCESS,
        'model_loaded': model_loaded,
        'files_in_directory': os.listdir('.'),
        'dataset_files_exist': {}
    }
    
    # Check dataset files
    dataset_files = ['Book_Dataset.csv', 'book_dataset.csv', 'educational_text_corpus.csv']
    for file in dataset_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                debug_data['dataset_files_exist'][file] = {
                    'exists': True,
                    'rows': len(df),
                    'columns': list(df.columns)
                }
            except Exception as e:
                debug_data['dataset_files_exist'][file] = {
                    'exists': True,
                    'error': str(e)
                }
        else:
            debug_data['dataset_files_exist'][file] = {'exists': False}
    
    return jsonify(debug_data)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/ (main page)',
            '/health',
            '/predict',
            '/analyze', 
            '/model_info',
            '/debug'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please check server logs for details',
        'model_loaded': model_loaded,
        'model_import_success': MODEL_IMPORT_SUCCESS
    }), 500

if __name__ == '__main__':
    # Initialize models on startup
    logger.info("Starting Cognitive Load Assessment API server...")
    logger.info("Looking for Index.html and Book_Dataset.csv...")
    
    # Check if files exist
    if os.path.exists('Index.html'):
        logger.info("✓ Index.html found")
    else:
        logger.error("✗ Index.html not found")
    
    if os.path.exists('Book_Dataset.csv'):
        logger.info("✓ Book_Dataset.csv found")
    else:
        logger.warning("⚠ Book_Dataset.csv not found, will check alternatives")
    
    initialize_models()
    
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Model loaded: {model_loaded}")
    logger.info(f"Model import success: {MODEL_IMPORT_SUCCESS}")
    logger.info(f"Available labels: {NEW_LABELS}")
    logger.info(f"Server starting on http://localhost:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )