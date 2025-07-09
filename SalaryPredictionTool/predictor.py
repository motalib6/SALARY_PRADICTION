import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictor:
    def __init__(self, models, encoders, scaler, feature_columns=None):
        self.models = models
        self.encoders = encoders
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.best_model_name, self.best_model = self._determine_best_model()
    
    def _determine_best_model(self):
        """
        Determine the best model based on available models
        Priority: Gradient Boosting > Random Forest > Linear Regression
        """
        model_priority = ["Gradient Boosting", "Random Forest", "Linear Regression"]
        best_model_name = None
        best_model = None
        
        for model_name in model_priority:
            if model_name in self.models:
                best_model_name = model_name
                best_model = self.models[model_name]
                break
        
        if best_model is None and self.models:
            # Fallback to first available model
            best_model_name = list(self.models.keys())[0]
            best_model = self.models[best_model_name]
        
        return best_model_name, best_model
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction
        """
        # Convert to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure all required columns are present and in correct order
        if self.feature_columns is None:
            # If feature_columns not provided, use default order
            required_columns = ['age', 'gender', 'education', 'experience', 'job_title', 
                               'location', 'industry', 'company_size', 'remote_work']
        else:
            required_columns = self.feature_columns
        
        # Add missing columns with default values
        default_values = {
            'age': 30,
            'gender': 'Male',
            'education': "Bachelor's",
            'experience': 5,
            'job_title': 'Software Engineer',
            'location': 'New York, NY',
            'industry': 'Technology',
            'company_size': 'Medium (51-200)',
            'remote_work': 'No'
        }
        
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = default_values.get(col, 'Unknown')
        
        # Reorder columns to match training order
        input_df = input_df[required_columns]
        
        # Encode categorical variables
        processed_df = input_df.copy()
        
        for col, encoder in self.encoders.items():
            if col in processed_df.columns:
                try:
                    # Handle unseen categories
                    unique_values = processed_df[col].unique()
                    for value in unique_values:
                        if value not in encoder.classes_:
                            # Use the most frequent class as default
                            processed_df[col] = processed_df[col].replace(value, encoder.classes_[0])
                    
                    processed_df[col] = encoder.transform(processed_df[col].astype(str))
                except Exception as e:
                    print(f"Warning: Error encoding {col}: {e}")
                    # Use default value (first class)
                    processed_df[col] = 0
        
        # Scale the features
        try:
            processed_array = self.scaler.transform(processed_df)
            processed_df = pd.DataFrame(processed_array, columns=processed_df.columns)
        except Exception as e:
            print(f"Warning: Error scaling features: {e}")
        
        return processed_df
    
    def predict(self, input_data, model_name=None):
        """
        Make salary prediction
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        # Preprocess input
        processed_input = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        # Return single value if single prediction
        if len(prediction) == 1:
            return float(prediction[0])
        
        return prediction
    
    def predict_with_confidence(self, input_data, model_name=None):
        """
        Make prediction with confidence interval (for tree-based models)
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        # Preprocess input
        processed_input = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        
        # Calculate confidence interval
        confidence_interval = None
        
        if hasattr(model, 'estimators_'):
            # For ensemble methods, use predictions from individual estimators
            try:
                individual_predictions = []
                for estimator in model.estimators_:
                    pred = estimator.predict(processed_input)[0]
                    individual_predictions.append(pred)
                
                std_dev = np.std(individual_predictions)
                confidence_interval = {
                    'lower': prediction - 1.96 * std_dev,
                    'upper': prediction + 1.96 * std_dev,
                    'std_dev': std_dev
                }
            except Exception as e:
                print(f"Warning: Could not calculate confidence interval: {e}")
        
        return {
            'prediction': float(prediction),
            'confidence_interval': confidence_interval,
            'model_used': model_name
        }
    
    def predict_all_models(self, input_data):
        """
        Make predictions using all available models
        """
        predictions = {}
        
        for model_name in self.models:
            try:
                prediction = self.predict(input_data, model_name)
                predictions[model_name] = prediction
            except Exception as e:
                print(f"Warning: Error predicting with {model_name}: {e}")
                predictions[model_name] = None
        
        return predictions
    
    def get_feature_impact(self, input_data, model_name=None):
        """
        Analyze feature impact on prediction (for tree-based models)
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            return None
        
        # Get feature names
        feature_names = ['age', 'gender', 'education', 'experience', 'job_title', 
                        'location', 'industry', 'company_size', 'remote_work']
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create impact analysis
        impact_analysis = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'value': [input_data.get(feature, 'N/A') for feature in feature_names]
        }).sort_values('importance', ascending=False)
        
        return impact_analysis
    
    def validate_input(self, input_data):
        """
        Validate input data
        """
        errors = []
        
        # Check required fields
        required_fields = ['experience', 'education', 'job_title', 'location', 'industry']
        
        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate data types and ranges
        if 'age' in input_data:
            if not isinstance(input_data['age'], (int, float)) or input_data['age'] < 18 or input_data['age'] > 100:
                errors.append("Age must be a number between 18 and 100")
        
        if 'experience' in input_data:
            if not isinstance(input_data['experience'], (int, float)) or input_data['experience'] < 0:
                errors.append("Experience must be a non-negative number")
        
        # Validate categorical values
        valid_categories = {
            'education': ["High School", "Bachelor's", "Master's", "PhD"],
            'gender': ["Male", "Female", "Other"],
            'remote_work': ["Yes", "No", "Hybrid"],
            'company_size': ["Small (1-50)", "Medium (51-200)", "Large (201-1000)", "Enterprise (1000+)"]
        }
        
        for field, valid_values in valid_categories.items():
            if field in input_data and input_data[field] not in valid_values:
                errors.append(f"Invalid value for {field}: {input_data[field]}")
        
        return errors
    
    def get_model_info(self):
        """
        Get information about available models
        """
        info = {
            'available_models': list(self.models.keys()),
            'best_model': self.best_model_name,
            'encoders': list(self.encoders.keys()),
            'scaler_available': self.scaler is not None
        }
        
        return info

if __name__ == "__main__":
    # Test the predictor
    from data_generator import generate_synthetic_data
    from model_trainer import ModelTrainer
    
    # Generate sample data and train models
    data = generate_synthetic_data(1000)
    trainer = ModelTrainer(data)
    trainer.train_models()
    
    # Initialize predictor
    predictor = SalaryPredictor(trainer.models, trainer.encoders, trainer.scaler)
    
    # Test prediction
    test_input = {
        'age': 30,
        'gender': 'Male',
        'education': "Bachelor's",
        'experience': 5,
        'job_title': 'Software Engineer',
        'location': 'New York, NY',
        'industry': 'Technology',
        'company_size': 'Medium (51-200)',
        'remote_work': 'No'
    }
    
    # Make prediction
    prediction = predictor.predict(test_input)
    print(f"Predicted salary: ${prediction:,.0f}")
    
    # Prediction with confidence
    detailed_prediction = predictor.predict_with_confidence(test_input)
    print(f"Detailed prediction: {detailed_prediction}")
    
    # Get model info
    info = predictor.get_model_info()
    print(f"Model info: {info}")
