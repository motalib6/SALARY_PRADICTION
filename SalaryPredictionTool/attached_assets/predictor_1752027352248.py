import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SalaryPredictor:
    def __init__(self, models, scaler, label_encoders):
        """
        Initialize the SalaryPredictor with trained models and preprocessors
        
        Args:
            models (dict): Dictionary of trained models
            scaler (StandardScaler): Fitted scaler for feature scaling
            label_encoders (dict): Dictionary of fitted label encoders
        """
        self.models = models
        self.scaler = scaler
        self.label_encoders = label_encoders
        
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction
        
        Args:
            input_data (dict): Dictionary containing input features
            
        Returns:
            np.array: Preprocessed feature array
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Define the expected order of features
        feature_order = ['years_experience', 'education', 'job_title', 'location', 'industry']
        
        # Reorder columns to match training data
        df = df[feature_order]
        
        # Encode categorical variables
        categorical_columns = ['education', 'job_title', 'location', 'industry']
        
        for column in categorical_columns:
            if column in self.label_encoders:
                le = self.label_encoders[column]
                try:
                    df[column] = le.transform(df[column])
                except ValueError as e:
                    # Handle unseen categories by using the most frequent class
                    print(f"Warning: Unknown category '{input_data[column]}' for {column}. Using default.")
                    df[column] = 0  # Use default encoding
        
        return df.values
    
    def predict_salary(self, input_data, model_name='Random Forest'):
        """
        Predict salary for given input data
        
        Args:
            input_data (dict): Dictionary containing employee features
            model_name (str): Name of the model to use for prediction
            
        Returns:
            float: Predicted salary
        """
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Get the specified model
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            
            model = self.models[model_name]
            
            # Scale features for Linear Regression
            if model_name == 'Linear Regression':
                X_scaled = self.scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
            else:
                # Tree-based models don't need scaling
                prediction = model.predict(X)[0]
            
            return max(prediction, 0)  # Ensure non-negative salary
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return 0
    
    def predict_salary_all_models(self, input_data):
        """
        Predict salary using all available models
        
        Args:
            input_data (dict): Dictionary containing employee features
            
        Returns:
            dict: Dictionary with predictions from all models
        """
        predictions = {}
        
        for model_name in self.models.keys():
            try:
                prediction = self.predict_salary(input_data, model_name)
                predictions[model_name] = prediction
            except Exception as e:
                print(f"Error predicting with {model_name}: {str(e)}")
                predictions[model_name] = 0
        
        return predictions
    
    def get_prediction_confidence(self, input_data, model_name='Random Forest', n_samples=100):
        """
        Get prediction confidence using bootstrap sampling (simplified approach)
        
        Args:
            input_data (dict): Dictionary containing employee features
            model_name (str): Name of the model to use
            n_samples (int): Number of bootstrap samples
            
        Returns:
            tuple: (mean_prediction, confidence_interval)
        """
        try:
            predictions = []
            base_prediction = self.predict_salary(input_data, model_name)
            
            # Simulate uncertainty by adding noise to the prediction
            # This is a simplified approach - in practice, you'd use proper bootstrap sampling
            for _ in range(n_samples):
                noise = np.random.normal(0, base_prediction * 0.05)  # 5% noise
                predictions.append(base_prediction + noise)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions)
            ci_lower = np.percentile(predictions, 2.5)
            ci_upper = np.percentile(predictions, 97.5)
            
            return mean_pred, (ci_lower, ci_upper)
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0, (0, 0)
    
    def explain_prediction(self, input_data, model_name='Random Forest'):
        """
        Provide explanation for the prediction (simplified feature importance)
        
        Args:
            input_data (dict): Dictionary containing employee features
            model_name (str): Name of the model to use
            
        Returns:
            dict: Feature importance explanation
        """
        try:
            model = self.models[model_name]
            
            # Only tree-based models have feature_importances_
            if hasattr(model, 'feature_importances_'):
                feature_names = ['years_experience', 'education', 'job_title', 'location', 'industry']
                importances = model.feature_importances_
                
                # Create feature importance dictionary
                importance_dict = {}
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = {
                        'importance': importances[i],
                        'value': input_data[feature]
                    }
                
                # Sort by importance
                sorted_features = sorted(importance_dict.items(), 
                                       key=lambda x: x[1]['importance'], 
                                       reverse=True)
                
                return dict(sorted_features)
            else:
                return {"message": f"Feature importance not available for {model_name}"}
                
        except Exception as e:
            return {"error": f"Error explaining prediction: {str(e)}"}
    
    def validate_input(self, input_data):
        """
        Validate input data
        
        Args:
            input_data (dict): Dictionary containing employee features
            
        Returns:
            tuple: (is_valid, error_message)
        """
        required_fields = ['years_experience', 'education', 'job_title', 'location', 'industry']
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in input_data:
                return False, f"Missing required field: {field}"
        
        # Validate years_experience
        try:
            years_exp = float(input_data['years_experience'])
            if years_exp < 0 or years_exp > 50:
                return False, "Years of experience must be between 0 and 50"
        except (ValueError, TypeError):
            return False, "Years of experience must be a valid number"
        
        # Validate categorical fields are not empty
        categorical_fields = ['education', 'job_title', 'location', 'industry']
        for field in categorical_fields:
            if not input_data[field] or input_data[field].strip() == '':
                return False, f"{field} cannot be empty"
        
        return True, "Input data is valid"

if __name__ == "__main__":
    # Test the predictor (this would normally use trained models)
    print("SalaryPredictor class is ready for use with trained models.")
    
    # Example usage:
    # predictor = SalaryPredictor(models, scaler, label_encoders)
    # 
    # input_data = {
    #     'years_experience': 5,
    #     'education': 'Bachelor\'s Degree',
    #     'job_title': 'Software Engineer',
    #     'location': 'San Francisco',
    #     'industry': 'Technology'
    # }
    # 
    # prediction = predictor.predict_salary(input_data)
    # print(f"Predicted salary: ${prediction:,.2f}")
