import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, data):
        self.data = data.copy()
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None # Initialize feature_columns
        
    def preprocess_data(self):
        """
        Preprocess the data for training
        """
        # Prepare features and target
        feature_columns = [col for col in self.data.columns if col != 'salary']
        self.feature_columns = feature_columns # Store feature_columns
        X = self.data[feature_columns].copy()
        y = self.data['salary'].copy()
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def train_models(self, models_to_train=None, test_size=0.2, random_state=42):
        """
        Train multiple models and return performance metrics
        """
        if models_to_train is None:
            models_to_train = ["Linear Regression", "Random Forest", "Gradient Boosting"]
        
        # Preprocess data
        X, y = self.preprocess_data()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Define models
        model_dict = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                random_state=random_state,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2
            )
        }
        
        results = {}
        
        for model_name in models_to_train:
            if model_name in model_dict:
                print(f"Training {model_name}...")
                
                model = model_dict[model_name]
                
                # Record training time
                start_time = time.time()
                
                # Train model
                model.fit(self.X_train, self.y_train)
                
                training_time = time.time() - start_time
                
                # Make predictions
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Calculate metrics
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                
                train_mae = mean_absolute_error(self.y_train, y_pred_train)
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=5, scoring='r2')
                
                # Store model and results
                self.models[model_name] = model
                
                results[model_name] = {
                    'Training R² Score': train_r2,
                    'Test R² Score': test_r2,
                    'R² Score': test_r2,  # Main metric for comparison
                    'Training RMSE': train_rmse,
                    'Test RMSE': test_rmse,
                    'RMSE': test_rmse,  # Main metric for comparison
                    'Training MAE': train_mae,
                    'Test MAE': test_mae,
                    'MAE': test_mae,  # Main metric for comparison
                    'CV R² Mean': cv_scores.mean(),
                    'CV R² Std': cv_scores.std(),
                    'Training Time (s)': training_time,
                    'Overfitting': train_r2 - test_r2
                }
                
                print(f"  ✓ {model_name} - R² Score: {test_r2:.4f}, RMSE: ${test_rmse:,.0f}")
        
        return results
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance for tree-based models
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            feature_names = [col for col in self.data.columns if col != 'salary']
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def get_model_predictions(self, model_name, X_input):
        """
        Get predictions from a specific model
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        return model.predict(X_input)
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models on test data
        """
        if not self.models:
            return None
        
        results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            results[model_name] = {
                'R² Score': r2_score(self.y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'MAE': mean_absolute_error(self.y_test, y_pred)
            }
        
        return results
    
    def get_best_model(self, metric='R² Score'):
        """
        Get the best model based on a specific metric
        """
        if not self.models:
            return None
        
        results = self.evaluate_all_models()
        
        if metric == 'R² Score':
            best_model_name = max(results, key=lambda x: results[x][metric])
        else:  # For RMSE and MAE, lower is better
            best_model_name = min(results, key=lambda x: results[x][metric])
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, filepath):
        """
        Save trained models to file
        """
        import joblib
        
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """
        Load trained models from file
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.encoders = model_data['encoders']
        self.scaler = model_data['scaler']
        
        print(f"Models loaded from {filepath}")

if __name__ == "__main__":
    # Test the trainer
    from data_generator import generate_synthetic_data
    
    # Generate sample data
    data = generate_synthetic_data(1000)
    
    # Initialize trainer
    trainer = ModelTrainer(data)
    
    # Train models
    results = trainer.train_models()
    
    # Print results
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if 'Time' in metric:
                print(f"  {metric}: {value:.2f}s")
            elif 'R²' in metric or 'CV' in metric:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: ${value:,.0f}")
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model()
    print(f"\nBest Model: {best_model_name}")
