# Employee Salary Predictor by SK MOTALIB

## Overview

This is a Streamlit-based web application that predicts employee salaries using machine learning models. The application generates synthetic employee data and provides an interactive interface for salary prediction with support for multiple currencies and regions. The system uses a modular architecture with separate components for data generation, model training, and prediction services.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - chosen for rapid prototyping and easy deployment of ML applications
- **UI Components**: Interactive forms, data visualizations, and real-time predictions
- **Styling**: Custom CSS for enhanced user experience with gradient backgrounds and modern styling
- **Visualization**: Multiple libraries (Matplotlib, Seaborn, Plotly) for comprehensive data visualization
- **Layout**: Wide layout with expandable sidebar for better user experience

### Backend Architecture
- **Language**: Python 3.x
- **ML Framework**: Scikit-learn for model training and prediction
- **Data Processing**: Pandas and NumPy for data manipulation and numerical operations
- **Model Pipeline**: Automated preprocessing with label encoding and standard scaling
- **Model Types**: 
  - Linear Regression (baseline model)
  - Random Forest Regressor (ensemble method)
  - Gradient Boosting Regressor (advanced ensemble with highest priority)

### Application Structure
- **Modular Design**: Separated concerns across multiple Python modules
- **Object-Oriented**: Uses classes for model training and prediction services
- **Error Handling**: Comprehensive warning suppression and error management
- **Extensible**: Easy to add new models or features

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Streamlit web interface and application orchestration
- **Features**: 
  - Interactive UI with custom styling
  - Real-time predictions and visualizations
  - Multi-currency support
  - Data loading from multiple sources (synthetic and Kaggle)
- **Rationale**: Centralized entry point for the application with clean separation of UI and business logic

### 2. Data Generator (`data_generator.py`)
- **Purpose**: Generates synthetic employee data for training and testing
- **Features**: 
  - Realistic employee profiles with 35+ job titles
  - Multiple industries, locations, and company sizes
  - Configurable data generation with seeded randomization
- **Rationale**: Provides consistent, reproducible data for demonstration without privacy concerns

### 3. Model Trainer (`model_trainer.py`)
- **Purpose**: Handles data preprocessing and model training pipeline
- **Features**: 
  - Automated data preprocessing with label encoding
  - Standard scaling for numerical features
  - Multiple ML model training with cross-validation
  - Performance metrics calculation (R², MSE, MAE)
- **Rationale**: Centralizes ML pipeline logic for maintainability and consistent preprocessing

### 4. Predictor (`predictor.py`)
- **Purpose**: Handles real-time salary predictions
- **Features**: 
  - Input preprocessing and validation
  - Model selection with priority-based fallback
  - Handles missing input columns with default values
- **Rationale**: Separates prediction logic from training for better code organization

### 5. Utilities (`utils.py`)
- **Purpose**: Shared utility functions for currency conversion and data loading
- **Features**: 
  - Multi-currency exchange rates (20+ currencies)
  - Currency formatting with proper symbols
  - Kaggle dataset loading capabilities
- **Rationale**: Centralizes common functionality to avoid code duplication

## Data Flow

1. **Data Generation**: Synthetic data is generated with realistic employee profiles
2. **Data Preprocessing**: Categorical variables are encoded, numerical features are scaled
3. **Model Training**: Multiple ML models are trained with cross-validation
4. **Model Selection**: Best performing model is automatically selected (priority: Gradient Boosting > Random Forest > Linear Regression)
5. **Prediction**: User input is processed through the same preprocessing pipeline
6. **Output**: Salary prediction is formatted and displayed with currency conversion

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and tools

### Visualization Libraries
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive visualizations

### Data Sources
- **Kaggle Hub**: Integration for downloading real-world datasets
- **Synthetic Data**: Generated employee profiles for consistent testing

### Model Dependencies
- **LabelEncoder**: Categorical variable encoding
- **StandardScaler**: Feature scaling
- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting

## Deployment Strategy

### Local Development
- **Environment**: Python 3.x with pip dependencies
- **Execution**: `streamlit run app.py`
- **Configuration**: Page config set for wide layout and custom branding

### Production Considerations
- **Scalability**: Modular architecture allows for easy scaling
- **Performance**: Model caching and efficient data processing
- **Error Handling**: Comprehensive warning suppression and graceful error handling
- **Multi-currency**: Global currency support for international users

### File Structure
```
├── app.py                 # Main Streamlit application
├── data_generator.py      # Synthetic data generation
├── model_trainer.py       # ML model training pipeline
├── predictor.py          # Salary prediction service
├── utils.py              # Shared utility functions
└── attached_assets/      # Additional documentation and assets
```

The application is designed to be self-contained with no external database requirements, making it easy to deploy on various platforms including Replit, Streamlit Cloud, or local environments.
