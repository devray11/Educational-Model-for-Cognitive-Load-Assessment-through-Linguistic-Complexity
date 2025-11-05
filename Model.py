"""
Cognitive Load Assessment Model
Advanced Text Classification for Educational Level Prediction

This module implements a comprehensive machine learning pipeline for classifying text
into educational levels (school, university, phd) based on linguistic complexity features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import warnings
import pickle
import joblib
import textstat  # Fixed import - use textstat directly
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import spacy
from collections import Counter
import re
from wordfreq import word_frequency
import logging
from datetime import datetime
import os
import json
from scipy import stats
from sklearn.utils.class_weight import compute_class_weight

# Try to import advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Try to import transformer libraries
try:
    from transformers import (BertTokenizer, BertForSequenceClassification, 
                             GPT2Tokenizer, GPT2ForSequenceClassification,
                             T5Tokenizer, T5ForConditionalGeneration,
                             RobertaTokenizer, RobertaForSequenceClassification)
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
    nlp = None

warnings.filterwarnings('ignore')

class TextFeatureExtractor:
    """
    Advanced feature extraction class for linguistic complexity analysis.
    Extracts 10 key features that correlate with cognitive load.
    """
    
    def __init__(self):
        """Initialize the feature extractor with required resources."""
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        logger.info("TextFeatureExtractor initialized")
    
    def extract_features(self, text):
        """
        Extract comprehensive linguistic features from text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Dictionary containing all linguistic features
        """
        if not text or len(text.strip()) == 0:
            return self._get_default_features()
        
        features = {}
        
        try:
            # Basic preprocessing
            text = self._preprocess_text(text)
            
            # Extract all features
            features['avg_sentence_length'] = self._calculate_avg_sentence_length(text)
            features['type_token_ratio'] = self._calculate_type_token_ratio(text)
            features['lexical_sophistication'] = self._calculate_lexical_sophistication(text)
            features['word_concreteness'] = self._calculate_word_concreteness(text)
            features['mean_dependency_distance'] = self._calculate_dependency_distance(text)
            features['subordination_ratio'] = self._calculate_subordination_ratio(text)
            features['flesch_kincaid_grade_level'] = self._calculate_flesch_kincaid(text)
            features['smog_index'] = self._calculate_smog_index(text)
            features['connective_frequency'] = self._calculate_connective_frequency(text)
            features['referential_cohesion'] = self._calculate_referential_cohesion(text)
            
            logger.info(f"Successfully extracted features for text of length {len(text)}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return self._get_default_features()
    
    def _preprocess_text(self, text):
        """Preprocess text for feature extraction."""
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _calculate_avg_sentence_length(self, text):
        """Calculate average sentence length in words."""
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.0
            
            total_words = 0
            for sentence in sentences:
                words = word_tokenize(sentence)
                words = [word for word in words if word.isalnum()]
                total_words += len(words)
            
            return total_words / len(sentences)
        except Exception:
            return 10.0  # Default value
    
    def _calculate_type_token_ratio(self, text):
        """Calculate type-token ratio (lexical diversity)."""
        try:
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            
            if not words:
                return 0.0
            
            unique_words = set(words)
            return len(unique_words) / len(words)
        except Exception:
            return 0.5  # Default value
    
    def _calculate_lexical_sophistication(self, text):
        """Calculate percentage of advanced/sophisticated words."""
        try:
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum()]
            
            if not words:
                return 0.0
            
            advanced_count = 0
            for word in words:
                try:
                    # Consider words with low frequency as advanced
                    freq = word_frequency(word, 'en')
                    if freq < 0.0001:  # Threshold for advanced words
                        advanced_count += 1
                except:
                    # If wordfreq fails, use word length as heuristic
                    if len(word) > 6:
                        advanced_count += 1
            
            return (advanced_count / len(words)) * 100
        except Exception:
            return 15.0  # Default value
    
    def _calculate_word_concreteness(self, text):
        """Calculate average word concreteness score."""
        try:
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum()]
            
            if not words:
                return 3.0
            
            # Simplified concreteness calculation based on POS tags
            pos_tags = pos_tag(words)
            concrete_score = 0
            
            for word, pos in pos_tags:
                if pos.startswith('NN'):  # Nouns tend to be more concrete
                    concrete_score += 4.5
                elif pos.startswith('VB'):  # Verbs
                    concrete_score += 3.5
                elif pos.startswith('JJ'):  # Adjectives
                    concrete_score += 3.0
                else:
                    concrete_score += 2.5
            
            return concrete_score / len(words)
        except Exception:
            return 3.0  # Default value
    
    def _calculate_dependency_distance(self, text):
        """Calculate mean dependency distance using spaCy."""
        if not nlp:
            return 3.0  # Default value
        
        try:
            doc = nlp(text)
            distances = []
            
            for token in doc:
                if token.head != token:  # Avoid root tokens
                    distance = abs(token.i - token.head.i)
                    distances.append(distance)
            
            return np.mean(distances) if distances else 3.0
        except Exception:
            return 3.0  # Default value
    
    def _calculate_subordination_ratio(self, text):
        """Calculate ratio of subordinate clauses."""
        try:
            # Simple approximation using subordinating conjunctions
            subordinating_conjunctions = {
                'although', 'because', 'since', 'while', 'when', 'where', 'if', 'unless',
                'until', 'after', 'before', 'though', 'whereas', 'whether', 'as'
            }
            
            words = word_tokenize(text.lower())
            sentences = sent_tokenize(text)
            
            if not sentences:
                return 0.0
            
            subordinate_count = sum(1 for word in words if word in subordinating_conjunctions)
            return subordinate_count / len(sentences)
        except Exception:
            return 0.3  # Default value
    
    def _calculate_flesch_kincaid(self, text):
        """Calculate Flesch-Kincaid grade level."""
        try:
            return textstat.flesch_kincaid_grade(text)  # Fixed function name
        except Exception:
            return 10.0  # Default value
    
    def _calculate_smog_index(self, text):
        """Calculate SMOG readability index."""
        try:
            return textstat.smog_index(text)
        except Exception:
            return 10.0  # Default value
    
    def _calculate_connective_frequency(self, text):
        """Calculate frequency of connective words per 100 words."""
        try:
            connectives = {
                'and', 'but', 'or', 'so', 'yet', 'for', 'nor', 'however', 'therefore',
                'moreover', 'furthermore', 'nevertheless', 'consequently', 'meanwhile',
                'likewise', 'otherwise', 'hence', 'thus', 'accordingly'
            }
            
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum()]
            
            if not words:
                return 0.0
            
            connective_count = sum(1 for word in words if word in connectives)
            return (connective_count / len(words)) * 100
        except Exception:
            return 5.0  # Default value
    
    def _calculate_referential_cohesion(self, text):
        """Calculate referential cohesion score."""
        try:
            words = word_tokenize(text.lower())
            
            # Pronouns and determiners that create cohesion
            cohesive_elements = {
                'he', 'she', 'it', 'they', 'them', 'his', 'her', 'its', 'their',
                'this', 'that', 'these', 'those', 'the', 'such'
            }
            
            if not words:
                return 0.0
            
            cohesive_count = sum(1 for word in words if word in cohesive_elements)
            return cohesive_count / len(words)
        except Exception:
            return 0.5  # Default value
    
    def _get_default_features(self):
        """Return default feature values for error cases."""
        return {
            'avg_sentence_length': 15.0,
            'type_token_ratio': 0.5,
            'lexical_sophistication': 20.0,
            'word_concreteness': 3.0,
            'mean_dependency_distance': 3.0,
            'subordination_ratio': 0.3,
            'flesch_kincaid_grade_level': 10.0,
            'smog_index': 10.0,
            'connective_frequency': 5.0,
            'referential_cohesion': 0.5
        }


# Transformer-related classes (only if transformers are available)
if TRANSFORMERS_AVAILABLE:
    class TransformerDataset(Dataset):
        """PyTorch Dataset for transformer models."""
        
        def __init__(self, texts, labels, tokenizer, max_length=512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    class TransformerClassifier(nn.Module):
        """Custom transformer classifier."""
        
        def __init__(self, model_name, num_classes=3):
            super(TransformerClassifier, self).__init__()
            
            if 'bert' in model_name.lower():
                self.model = BertForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_classes
                )
            elif 'roberta' in model_name.lower():
                self.model = RobertaForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_classes
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        
        def forward(self, input_ids, attention_mask, labels=None):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class AdvancedTextClassifier:
    """
    Advanced text classification system combining traditional ML and deep learning models.
    """
    
    def __init__(self):
        """Initialize the classifier with all necessary components."""
        self.feature_extractor = TextFeatureExtractor()
        self.models = {}
        self.transformers = {}
        self.tokenizers = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self.performance_metrics = {}
        self.best_model = None
        self.feature_names = [
            'avg_sentence_length', 'type_token_ratio', 'lexical_sophistication',
            'word_concreteness', 'mean_dependency_distance', 'subordination_ratio',
            'flesch_kincaid_grade_level', 'smog_index', 'connective_frequency',
            'referential_cohesion'
        ]
        
        logger.info("AdvancedTextClassifier initialized")
    
    def load_and_prepare_data(self, filepath):
        """
        Load and prepare the dataset for training.
        
        Args:
            filepath (str): Path to the dataset file
            
        Returns:
            tuple: Prepared features and labels
        """
        try:
            # Load dataset
            if filepath.endswith('.xlsx'):
                data = pd.read_excel(filepath, sheet_name='Sheet1')
            else:
                data = pd.read_csv(filepath)
            
            logger.info(f"Loaded dataset with shape: {data.shape}")
            
            # Prepare features
            feature_columns = self.feature_names
            X = data[feature_columns].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Prepare labels
            y = data['difficulty_level'].copy()
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            logger.info(f"Data preparation completed. Features: {X.shape}, Labels: {len(y_encoded)}")
            logger.info(f"Class distribution: {Counter(y)}")
            
            return X, y_encoded, data['paragraph'].values
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """Train traditional machine learning models."""
        logger.info("Training traditional ML models...")
        
        # Calculate class weights for imbalanced dataset
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Define models with hyperparameters
        model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    min_samples_leaf=2, random_state=42, class_weight='balanced'
                ),
                'scaler': None
            },
            'SVM': {
                'model': SVC(
                    kernel='rbf', C=1.0, gamma='scale', probability=True,
                    random_state=42, class_weight='balanced'
                ),
                'scaler': StandardScaler()
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42,
                    class_weight='balanced', solver='liblinear'
                ),
                'scaler': StandardScaler()
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42
                ),
                'scaler': StandardScaler()
            },
            'MLP': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32), max_iter=500,
                    random_state=42, early_stopping=True, validation_fraction=0.1
                ),
                'scaler': StandardScaler()
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            model_configs['XGBoost'] = {
                'model': xgb.XGBClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, eval_metric='mlogloss'
                ),
                'scaler': StandardScaler()
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            model_configs['LightGBM'] = {
                'model': lgb.LGBMClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, verbose=-1, class_weight='balanced'
                ),
                'scaler': StandardScaler()
            }
        
        # Train each model
        for name, config in model_configs.items():
            try:
                logger.info(f"Training {name}...")
                
                # Prepare data
                if config['scaler']:
                    scaler = config['scaler']
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                
                # Train model
                model = config['model']
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val_scaled)
                y_pred_proba = model.predict_proba(X_val_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
                
                # Store results
                self.models[name] = model
                self.performance_metrics[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Calculate feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(self.feature_names, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(self.feature_names, np.abs(model.coef_[0])))
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
    
    def train_transformer_models(self, texts_train, y_train, texts_val, y_val):
        """Train transformer models if available."""
        if not TRANSFORMERS_AVAILABLE:
            logger.info("Transformers not available, skipping transformer training")
            return
            
        logger.info("Training transformer models...")
        
        # Define transformer models to use
        transformer_models = {
            'BERT': 'bert-base-uncased',
            'RoBERTa': 'roberta-base'
        }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        for name, model_name in transformer_models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Initialize tokenizer
                if 'bert' in model_name:
                    tokenizer = BertTokenizer.from_pretrained(model_name)
                elif 'roberta' in model_name:
                    tokenizer = RobertaTokenizer.from_pretrained(model_name)
                else:
                    continue
                
                # Create datasets
                train_dataset = TransformerDataset(texts_train, y_train, tokenizer)
                val_dataset = TransformerDataset(texts_val, y_val, tokenizer)
                
                # Create dataloaders
                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=8)
                
                # Initialize model
                model = TransformerClassifier(model_name)
                model.to(device)
                
                # Training parameters
                optimizer = optim.AdamW(model.parameters(), lr=2e-5)
                criterion = nn.CrossEntropyLoss()
                
                # Train for limited epochs due to computational constraints
                num_epochs = 3
                model.train()
                
                for epoch in range(num_epochs):
                    total_loss = 0
                    for batch_idx, batch in enumerate(train_loader):
                        if batch_idx >= 10:  # Limit batches for demo
                            break
                        
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(input_ids, attention_mask, labels)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                    
                    logger.info(f"{name} Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/min(10, len(train_loader)):.4f}")
                
                # Evaluation
                model.eval()
                predictions = []
                probabilities = []
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        if batch_idx >= 5:  # Limit validation batches
                            break
                        
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        
                        outputs = model(input_ids, attention_mask)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1)
                        
                        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        probabilities.extend(probs.cpu().numpy())
                
                # Calculate metrics (limited due to computational constraints)
                if len(predictions) > 0:
                    val_subset = y_val[:len(predictions)]
                    accuracy = accuracy_score(val_subset, predictions)
                    
                    self.transformers[name] = model
                    self.tokenizers[name] = tokenizer
                    self.performance_metrics[name] = {
                        'accuracy': accuracy,
                        'predictions': np.array(predictions),
                        'probabilities': np.array(probabilities)
                    }
                    
                    logger.info(f"{name} - Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
    
    def perform_feature_analysis(self, X, y):
        """Perform comprehensive feature analysis."""
        logger.info("Performing feature analysis...")
        
        # Statistical tests
        f_stats, f_p_values = f_classif(X, y)
        mutual_info_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature analysis dataframe
        feature_analysis = pd.DataFrame({
            'Feature': self.feature_names,
            'F_Statistic': f_stats,
            'F_P_Value': f_p_values,
            'Mutual_Info': mutual_info_scores
        })
        
        # Add importance from best performing model
        if self.best_model and self.best_model in self.feature_importance:
            feature_analysis['Model_Importance'] = [
                self.feature_importance[self.best_model].get(feat, 0) 
                for feat in self.feature_names
            ]
        
        feature_analysis = feature_analysis.sort_values('Mutual_Info', ascending=False)
        
        logger.info("Feature Analysis Results:")
        logger.info(f"\n{feature_analysis.to_string()}")
        
        return feature_analysis
    
    def create_ensemble_model(self, X_train, y_train):
        """Create an ensemble model from the best performing models."""
        try:
            logger.info("Creating ensemble model...")
            
            # Select top 3 models based on accuracy
            top_models = sorted(
                [(name, metrics['accuracy']) for name, metrics in self.performance_metrics.items() 
                 if name in self.models],
                key=lambda x: x[1], reverse=True
            )[:3]
            
            ensemble_models = []
            for model_name, _ in top_models:
                if self.scalers.get(model_name):
                    # Create a pipeline for models that need scaling
                    from sklearn.pipeline import Pipeline
                    pipeline = Pipeline([
                        ('scaler', self.scalers[model_name]),
                        ('classifier', self.models[model_name])
                    ])
                    ensemble_models.append((model_name, pipeline))
                else:
                    ensemble_models.append((model_name, self.models[model_name]))
            
            # Create voting classifier
            voting_classifier = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'
            )
            
            # Fit ensemble
            voting_classifier.fit(X_train, y_train)
            self.models['Ensemble'] = voting_classifier
            
            logger.info(f"Ensemble created with models: {[name for name, _ in top_models]}")
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
    
    def train(self, filepath):
        """
        Main training function that orchestrates the entire training process.
        
        Args:
            filepath (str): Path to the training dataset
        """
        try:
            logger.info("Starting comprehensive model training...")
            
            # Load and prepare data
            X, y, texts = self.load_and_prepare_data(filepath)
            
            # Split data
            X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
                X, y, texts, test_size=0.2, random_state=42, stratify=y
            )
            
            X_train, X_val, y_train, y_val, texts_train, texts_val = train_test_split(
                X_train, y_train, texts_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            
            # Train traditional models
            self.train_traditional_models(X_train, y_train, X_val, y_val)
            
            # Train transformer models (if available)
            if TRANSFORMERS_AVAILABLE:
                self.train_transformer_models(texts_train, y_train, texts_val, y_val)
            
            # Find best model
            best_accuracy = 0
            for name, metrics in self.performance_metrics.items():
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    self.best_model = name
            
            logger.info(f"Best model: {self.best_model} with accuracy: {best_accuracy:.4f}")
            
            # Create ensemble
            self.create_ensemble_model(X_train, y_train)
            
            # Feature analysis
            feature_analysis = self.perform_feature_analysis(X, y)
            
            # Final evaluation on test set
            self.evaluate_on_test_set(X_test, y_test, texts_test)
            
            # Generate comprehensive report
            self.generate_training_report(feature_analysis)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise
    
    def evaluate_on_test_set(self, X_test, y_test, texts_test):
        """Evaluate all models on the test set."""
        logger.info("Evaluating models on test set...")
        
        for name, model in self.models.items():
            try:
                if name == 'Ensemble':
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                elif self.scalers.get(name):
                    X_test_scaled = self.scalers[name].transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate test metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Update performance metrics with test results
                if name not in self.performance_metrics:
                    self.performance_metrics[name] = {}
                
                self.performance_metrics[name]['test_accuracy'] = accuracy
                self.performance_metrics[name]['test_precision'] = precision
                self.performance_metrics[name]['test_recall'] = recall
                self.performance_metrics[name]['test_f1'] = f1
                
                logger.info(f"{name} Test - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name} on test set: {e}")
    
    def generate_training_report(self, feature_analysis):
        """Generate comprehensive training report."""
        logger.info("Generating comprehensive training report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'best_model': self.best_model,
            'performance_metrics': self.performance_metrics,
            'feature_analysis': feature_analysis.to_dict('records'),
            'feature_importance': self.feature_importance
        }
        
        # Save report
        with open('training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Training report saved as 'training_report.json'")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE TRAINING REPORT")
        print("="*80)
        
        print(f"\nBest Performing Model: {self.best_model}")
        if self.best_model in self.performance_metrics:
            metrics = self.performance_metrics[self.best_model]
            print(f"Best Model Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"Best Model F1 Score: {metrics.get('f1', 'N/A'):.4f}")
        
        print(f"\nTotal Models Trained: {len(self.models)}")
        print(f"Traditional ML Models: {len([m for m in self.models if m not in self.transformers and m != 'Ensemble'])}")
        print(f"Transformer Models: {len(self.transformers)}")
        
        print(f"\nTop 5 Most Important Features:")
        for i, row in feature_analysis.head().iterrows():
            print(f"  {i+1}. {row['Feature']}: {row['Mutual_Info']:.4f}")
    
    def predict(self, text):
        """
        Predict educational level for a given text.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Predictions from all models with confidence scores
        """
        if not self.models:
            raise ValueError("Models not trained. Please train models first.")
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(text)
            X = np.array([[features[feat] for feat in self.feature_names]])
            
            predictions = {}
            
            # Get predictions from all traditional models
            for name, model in self.models.items():
                try:
                    if name == 'Ensemble':
                        y_pred = model.predict(X)[0]
                        y_pred_proba = model.predict_proba(X)[0]
                    elif self.scalers.get(name):
                        X_scaled = self.scalers[name].transform(X)
                        y_pred = model.predict(X_scaled)[0]
                        y_pred_proba = model.predict_proba(X_scaled)[0]
                    else:
                        y_pred = model.predict(X)[0]
                        y_pred_proba = model.predict_proba(X)[0]
                    
                    # Convert to class label
                    class_label = self.label_encoder.inverse_transform([y_pred])[0]
                    confidence = np.max(y_pred_proba)
                    
                    predictions[name] = {
                        'prediction': class_label,
                        'confidence': confidence,
                        'probabilities': {
                            self.label_encoder.inverse_transform([i])[0]: prob 
                            for i, prob in enumerate(y_pred_proba)
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error predicting with {name}: {e}")
            
            # Get transformer predictions (simplified)
            for name, model in self.transformers.items():
                try:
                    # Note: This is a simplified prediction for transformers
                    # In practice, you'd want to properly tokenize and predict
                    predictions[name] = {
                        'prediction': 'university',  # Placeholder
                        'confidence': 0.5,
                        'probabilities': {'school': 0.3, 'university': 0.5, 'phd': 0.2}
                    }
                except Exception as e:
                    logger.error(f"Error predicting with transformer {name}: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {}
    
    def save_model(self, filepath='cognitive_load_model.pkl'):
        """Save the trained model."""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'label_encoder': self.label_encoder,
                'feature_extractor': self.feature_extractor,
                'feature_names': self.feature_names,
                'best_model': self.best_model,
                'performance_metrics': self.performance_metrics
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath='cognitive_load_model.pkl'):
        """Load a trained model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.label_encoder = model_data['label_encoder']
            self.feature_extractor = model_data['feature_extractor']
            self.feature_names = model_data['feature_names']
            self.best_model = model_data['best_model']
            self.performance_metrics = model_data['performance_metrics']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")


def main():
    """Main function to demonstrate the model training and evaluation."""
    try:
        # Initialize classifier
        classifier = AdvancedTextClassifier()
        
        # Train the model
        classifier.train('book_dataset.csv')
        
        # Save the trained model
        classifier.save_model()
        
        # Test prediction
        sample_text = """
        The methodological framework employed in this study utilizes a mixed-methods approach
        to investigate the phenomenological aspects of cognitive load theory. Through systematic
        analysis of qualitative and quantitative data, we examine the epistemological foundations
        underlying the theoretical constructs that inform our understanding of learning processes.
        """
        
        predictions = classifier.predict(sample_text)
        
        print(f"\nSample Prediction Results:")
        for model_name, result in predictions.items():
            print(f"{model_name}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        logger.info("Model training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()