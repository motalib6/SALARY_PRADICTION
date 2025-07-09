import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, df):
        """
        Initialize the ModelTrainer with employee data
        
        Args:
            df (pd.DataFrame): Employee dataset
        """
        self.df = df.copy()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.model_performance = {}
        
    def preprocess_data(self):
        """
        Preprocess the data for machine learning
        """
        # Separate features and target
        self.y = self.df['salary'].values
        feature_columns = ['years_experience', 'education', 'job_title', 'location', 'industry']
        self.X = self.df[feature_columns].copy()
        
        # Encode categorical variables
        categorical_columns = ['education', 'job_title', 'location', 'industry']
        
        for column in categorical_columns:
            le = LabelEncoder()
            self.X[column] = le.fit_transform(self.X[column])
            self.label_encoders[column] = le
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_linear_regression(self):
        """
        Train Linear Regression model
        """
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        
        self.models['Linear Regression'] = model
        self.model_performance['Linear Regression'] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R² Score': r2,
            'CV R² Score': cv_mean
        }
        
        return model, {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R² Score': r2,
            'CV R² Score': cv_mean
        }
    
    def train_random_forest(self):
        """
        Train Random Forest model
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)  # RF doesn't need scaled data
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        
        self.models['Random Forest'] = model
        self.model_performance['Random Forest'] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R² Score': r2,
            'CV R² Score': cv_mean
        }
        
        return model, {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R² Score': r2,
            'CV R² Score': cv_mean
        }
    
    def train_gradient_boosting(self):
        """
        Train Gradient Boosting model
        """
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)  # GB doesn't need scaled data
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        
        self.models['Gradient Boosting'] = model
        self.model_performance['Gradient Boosting'] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R² Score': r2,
            'CV R² Score': cv_mean
        }
        
        return model, {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R² Score': r2,
            'CV R² Score': cv_mean
        }
    
    def train_all_models(self):
        """
        Train all models and return results
        """
        # Preprocess data
        self.preprocess_data()
        
        # Train all models
        print("Training Linear Regression...")
        self.train_linear_regression()
        
        print("Training Random Forest...")
        self.train_random_forest()
        
        print("Training Gradient Boosting...")
        self.train_gradient_boosting()
        
        print("All models trained successfully!")
        
        return self.models, self.model_performance, self.scaler, self.label_encoders
    
    def get_best_model(self):
        """
        Get the best performing model based on R² score
        """
        if not self.model_performance:
            raise ValueError("No models have been trained yet. Call train_all_models() first.")
        
        best_model_name = max(self.model_performance.keys(), 
                            key=lambda x: self.model_performance[x]['R² Score'])
        
        return best_model_name, self.models[best_model_name]
    
    def print_model_comparison(self):
        """
        Print a comparison of all trained models
        """
        if not self.model_performance:
            print("No models have been trained yet.")
            return
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        for model_name, metrics in self.model_performance.items():
            print(f"\n{model_name}:")
            print(f"  Mean Absolute Error: ${metrics['MAE']:,.2f}")
            print(f"  Mean Squared Error: ${metrics['MSE']:,.2f}")
            print(f"  Root Mean Squared Error: ${metrics['RMSE']:,.2f}")
            print(f"  R² Score: {metrics['R² Score']:.4f}")
            print(f"  Cross-Validation R² Score: {metrics['CV R² Score']:.4f}")
        
        best_model_name, _ = self.get_best_model()
        print(f"\nBest performing model: {best_model_name}")
        print("="*80)

if __name__ == "__main__":
    # Test the model trainer
    from data_generator import generate_employee_data
    
    # Generate test data
    df = generate_employee_data(1000)
    
    # Initialize and train models
    trainer = ModelTrainer(df)
    models, performance, scaler, label_encoders = trainer.train_all_models()
    
    # Print comparison
    trainer.print_model_comparison()
