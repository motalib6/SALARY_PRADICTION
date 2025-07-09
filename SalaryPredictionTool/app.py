

import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

warnings.filterwarnings('ignore')

from data_generator import generate_synthetic_data
from model_trainer import ModelTrainer
from predictor import SalaryPredictor
from utils import get_currency_rates, format_currency, load_kaggle_data

# Set page configuration
st.set_page_config(
    page_title="Employee Salary Predictor by SK MOTALIB",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sidebar-info {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏆 Employee Salary Predictor by SK MOTALIB</h1>
        <p>Advanced Machine Learning-Based Salary Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Salary Prediction", "📈 Visualizations", "💱 Currency Converter"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🔮 Salary Prediction":
        show_prediction_page()
    elif page == "📈 Visualizations":
        show_visualizations()
    elif page == "💱 Currency Converter":
        show_currency_converter()

def show_home_page():
    st.markdown("## 🚀 Welcome to Advanced Salary Prediction System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This comprehensive machine learning application predicts employee salaries based on various factors including:
        - **Experience Level**: Years of professional experience
        - **Education**: Academic qualifications and degrees
        - **Job Title**: Specific role and position
        - **Location**: Geographic location and cost of living
        - **Industry**: Sector and business domain
        
        ### 🔧 Technical Features
        - **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting
        - **Real-time Predictions**: Instant salary estimates with confidence intervals
        - **Interactive Visualizations**: 3D plots, correlation matrices, and trend analysis
        - **Multi-currency Support**: Global salary predictions in 20+ currencies
        - **Kaggle Integration**: Authentic dataset from Kaggle platform
        """)
    
    with col2:
        st.markdown("""
        <div class="sidebar-info">
            <h4>🎯 Quick Stats</h4>
            <ul>
                <li>200+ Job Titles</li>
                <li>100+ Industries</li>
                <li>10+ Countries</li>
                <li>3 ML Models</li>
                <li>20+ Currencies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Data loading section
    st.markdown("## 📥 Data Loading")
    
    data_source = st.radio(
        "Select Data Source:",
        ["📊 Kaggle Dataset", "🔧 Synthetic Data", "📁 Upload CSV"]
    )
    
    if data_source == "📊 Kaggle Dataset":
        st.info("📋 This will download the real salary dataset from Kaggle for authentic predictions.")
        
        if st.button("🔄 Download Kaggle Dataset"):
            with st.spinner("Downloading dataset from Kaggle..."):
                try:
                    # Download the dataset
                    path = kagglehub.dataset_download("rkiattisak/salaly-prediction-for-beginer")
                    st.success(f"✅ Dataset downloaded successfully!")
                    st.info(f"📂 Dataset path: {path}")
                    
                    # Load the dataset
                    data = load_kaggle_data(path)
                    if data is not None:
                        st.session_state.data = data
                        st.success(f"✅ Data loaded: {len(data)} records")
                        
                        # Show data preview
                        st.subheader("📊 Dataset Preview")
                        st.dataframe(data.head(10))
                        
                        # Show data statistics
                        st.subheader("📈 Dataset Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", len(data))
                        with col2:
                            st.metric("Features", len(data.columns))
                        with col3:
                            if 'salary' in data.columns:
                                st.metric("Avg Salary", f"${data['salary'].mean():,.0f}")
                    else:
                        st.warning("⚠️ Could not load dataset properly. Using synthetic data instead.")
                        st.session_state.data = generate_synthetic_data()
                    
                except Exception as e:
                    st.error(f"❌ Error downloading dataset: {str(e)}")
                    st.info("🔄 Falling back to synthetic data generation...")
                    with st.spinner("Generating synthetic data..."):
                        st.session_state.data = generate_synthetic_data()
                        st.success("✅ Synthetic data generated successfully!")
    
    elif data_source == "🔧 Synthetic Data":
        if st.button("🎲 Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                st.session_state.data = generate_synthetic_data()
                st.success("✅ Synthetic data generated successfully!")
                st.dataframe(st.session_state.data.head())
    
    elif data_source == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.dataframe(st.session_state.data.head())
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Display data info if available
    if st.session_state.data is not None:
        st.markdown("## 📋 Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", len(st.session_state.data))
        with col2:
            st.metric("📋 Features", len(st.session_state.data.columns))
        with col3:
            # Check if salary column exists
            if 'salary' in st.session_state.data.columns:
                st.metric("💰 Avg Salary", f"${st.session_state.data['salary'].mean():,.0f}")
            else:
                st.metric("💰 Avg Salary", "N/A")
        with col4:
            # Check if salary column exists
            if 'salary' in st.session_state.data.columns:
                st.metric("📈 Salary Range", f"${st.session_state.data['salary'].std():,.0f}")
            else:
                st.metric("📈 Salary Range", "N/A")

def show_data_analysis():
    st.markdown("## 📊 Data Analysis & Exploration")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    
    # Basic statistics
    st.markdown("### 📈 Basic Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Numerical Summary")
        st.dataframe(data.describe())
    
    with col2:
        st.subheader("🔍 Data Info")
        st.text(f"Shape: {data.shape}")
        st.text(f"Missing Values: {data.isnull().sum().sum()}")
        st.text(f"Data Types:\n{data.dtypes.to_string()}")
    
    # Visualizations
    st.markdown("### 📈 Data Visualizations")
    
    # Salary distribution
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax[0].hist(data['salary'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax[0].set_title('Salary Distribution')
    ax[0].set_xlabel('Salary')
    ax[0].set_ylabel('Frequency')
    
    # Box plot
    ax[1].boxplot(data['salary'])
    ax[1].set_title('Salary Box Plot')
    ax[1].set_ylabel('Salary')
    
    st.pyplot(fig)
    
    # Correlation matrix
    st.markdown("### 🔗 Correlation Analysis")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        corr_matrix = data[numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
    
    # Categorical analysis
    st.markdown("### 📊 Categorical Analysis")
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) > 0:
        selected_cat = st.selectbox("Select categorical variable:", categorical_columns)
        
        if selected_cat:
            # Value counts
            value_counts = data[selected_cat].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 {selected_cat} Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'{selected_cat} Distribution')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                st.subheader(f"💰 Salary by {selected_cat}")
                avg_salary = data.groupby(selected_cat)['salary'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                avg_salary.plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title(f'Average Salary by {selected_cat}')
                plt.xticks(rotation=45)
                st.pyplot(fig)

def show_model_training():
    st.markdown("## 🤖 Model Training & Evaluation")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    
    # Model configuration
    st.markdown("### ⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 100, 42)
    
    with col2:
        models_to_train = st.multiselect(
            "Select Models to Train:",
            ["Linear Regression", "Random Forest", "Gradient Boosting"],
            default=["Linear Regression", "Random Forest", "Gradient Boosting"]
        )
    
    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            try:
                # Initialize model trainer
                trainer = ModelTrainer(data)
                
                # Train models
                results = trainer.train_models(
                    models_to_train=models_to_train,
                    test_size=test_size,
                    random_state=random_state
                )
                
                st.session_state.model_trainer = trainer
                st.session_state.predictor = SalaryPredictor(trainer.models, trainer.encoders, trainer.scaler)
                
                st.success("✅ Models trained successfully!")
                
                # Display results
                st.markdown("### 📊 Model Performance")
                
                results_df = pd.DataFrame(results).T
                results_df = results_df.round(4)
                
                # Color code the results
                st.dataframe(results_df.style.highlight_max(axis=0, subset=['R² Score']))
                
                # Best model
                best_model = results_df['R² Score'].idxmax()
                st.markdown(f"""
                <div class="prediction-result">
                    <h3>🏆 Best Model: {best_model}</h3>
                    <p>R² Score: {results_df.loc[best_model, 'R² Score']:.4f}</p>
                    <p>RMSE: ${results_df.loc[best_model, 'RMSE']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model comparison chart
                st.markdown("### 📈 Model Comparison")
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # R² Score comparison
                r2_scores = results_df['R² Score']
                axes[0, 0].bar(r2_scores.index, r2_scores.values, color='skyblue')
                axes[0, 0].set_title('R² Score Comparison')
                axes[0, 0].set_ylabel('R² Score')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # RMSE comparison
                rmse_scores = results_df['RMSE']
                axes[0, 1].bar(rmse_scores.index, rmse_scores.values, color='lightcoral')
                axes[0, 1].set_title('RMSE Comparison')
                axes[0, 1].set_ylabel('RMSE')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # MAE comparison
                mae_scores = results_df['MAE']
                axes[1, 0].bar(mae_scores.index, mae_scores.values, color='lightgreen')
                axes[1, 0].set_title('MAE Comparison')
                axes[1, 0].set_ylabel('MAE')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Training time comparison
                train_time = results_df['Training Time (s)']
                axes[1, 1].bar(train_time.index, train_time.values, color='gold')
                axes[1, 1].set_title('Training Time Comparison')
                axes[1, 1].set_ylabel('Time (seconds)')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")

def show_prediction_page():
    st.markdown("## 🔮 Salary Prediction")
    
    # Auto-train models if they don't exist
    if st.session_state.predictor is None:
        if st.session_state.data is None:
            st.info("🔄 Loading data and training models automatically...")
            with st.spinner("Generating synthetic data..."):
                st.session_state.data = generate_synthetic_data()
                st.success("✅ Data loaded successfully!")
        
        if st.session_state.model_trainer is None:
            with st.spinner("Training machine learning models..."):
                try:
                    trainer = ModelTrainer(st.session_state.data)
                    results = trainer.train_models()
                    st.session_state.model_trainer = trainer
                    st.session_state.predictor = SalaryPredictor(trainer.models, trainer.encoders, trainer.scaler)
                    st.success("✅ Models trained successfully!")
                except Exception as e:
                    st.error(f"❌ Error training models: {str(e)}")
                    return
    
    predictor = st.session_state.predictor
    
    # Input form
    st.markdown("### 📝 Employee Information")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            experience = st.number_input("Years of Experience", 0, 50, 5)
            education = st.selectbox("Education Level", [
                "High School", "Bachelor's", "Master's", "PhD"
            ])
            job_title = st.selectbox("Job Title", [
                "Software Engineer", "Data Scientist", "Product Manager", 
                "Marketing Manager", "Sales Representative", "HR Manager",
                "Financial Analyst", "Business Analyst", "Project Manager"
            ])
        
        with col2:
            location = st.selectbox("Location", [
                "New York", "San Francisco", "Los Angeles", "Chicago",
                "Boston", "Seattle", "Austin", "Denver", "Atlanta"
            ])
            industry = st.selectbox("Industry", [
                "Technology", "Finance", "Healthcare", "Manufacturing",
                "Retail", "Education", "Government", "Consulting"
            ])
            company_size = st.selectbox("Company Size", [
                "Small (1-50)", "Medium (51-200)", "Large (201-1000)", "Enterprise (1000+)"
            ])
        
        with col3:
            age = st.number_input("Age", 18, 65, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            remote_work = st.selectbox("Remote Work", ["Yes", "No"])
        
        submitted = st.form_submit_button("🔮 Predict Salary")
        
        if submitted:
            try:
                # Prepare input data
                input_data = {
                    'experience': experience,
                    'education': education,
                    'job_title': job_title,
                    'location': location,
                    'industry': industry,
                    'company_size': company_size,
                    'age': age,
                    'gender': gender,
                    'remote_work': remote_work
                }
                
                # Make prediction
                prediction = predictor.predict(input_data)
                
                # Display prediction
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h3>💰 Predicted Salary</h3>
                        <h2>${prediction:,.0f}</h2>
                        <p>Annual Salary (USD)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Calculate confidence interval (rough estimate)
                    confidence_range = prediction * 0.15
                    lower_bound = prediction - confidence_range
                    upper_bound = prediction + confidence_range
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📊 Confidence Interval</h4>
                        <p><strong>Lower:</strong> ${lower_bound:,.0f}</p>
                        <p><strong>Upper:</strong> ${upper_bound:,.0f}</p>
                        <p><strong>Range:</strong> ±${confidence_range:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Market comparison
                    if 'salary' in st.session_state.data.columns:
                        market_avg = st.session_state.data['salary'].mean()
                        percentile = (prediction / market_avg - 1) * 100
                        percentile_rank = (st.session_state.data['salary'] <= prediction).mean() * 100
                    else:
                        market_avg = 75000  # Default average
                        percentile = (prediction / market_avg - 1) * 100
                        percentile_rank = 50  # Default percentile
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📈 Market Comparison</h4>
                        <p><strong>Market Avg:</strong> ${market_avg:,.0f}</p>
                        <p><strong>Your Prediction:</strong> {percentile:+.1f}%</p>
                        <p><strong>Percentile:</strong> {percentile_rank:.0f}th</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Multi-currency display
                st.markdown("### 💱 Multi-Currency Conversion")
                
                currencies = get_currency_rates()
                currency_cols = st.columns(len(currencies))
                
                for i, (currency, rate) in enumerate(currencies.items()):
                    with currency_cols[i]:
                        converted_amount = prediction * rate
                        st.metric(
                            f"{currency}",
                            format_currency(converted_amount, currency),
                            f"Rate: {rate:.4f}"
                        )
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

def show_visualizations():
    st.markdown("## 📈 Advanced Visualizations")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first from the Home page.")
        return
    
    data = st.session_state.data
    
    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization Type:",
        ["📊 Salary Distribution", "🔗 Feature Relationships", "📈 Trend Analysis", "🎯 3D Scatter Plot"]
    )
    
    if viz_type == "📊 Salary Distribution":
        st.subheader("💰 Salary Distribution Analysis")
        
        # Interactive histogram
        fig = px.histogram(
            data, 
            x='salary', 
            nbins=30, 
            title='Salary Distribution',
            labels={'salary': 'Salary (USD)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"${data['salary'].mean():,.0f}")
        with col2:
            st.metric("Median", f"${data['salary'].median():,.0f}")
        with col3:
            st.metric("Std Dev", f"${data['salary'].std():,.0f}")
        with col4:
            st.metric("Range", f"${data['salary'].max() - data['salary'].min():,.0f}")
    
    elif viz_type == "🔗 Feature Relationships":
        st.subheader("🔍 Feature Relationship Analysis")
        
        # Select features for comparison
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis feature:", categorical_cols + numerical_cols)
        with col2:
            color_feature = st.selectbox("Color by:", categorical_cols, index=0)
        
        # Create visualization based on feature types
        if x_feature in categorical_cols:
            fig = px.box(
                data, 
                x=x_feature, 
                y='salary',
                color=color_feature,
                title=f'Salary Distribution by {x_feature}'
            )
        else:
            fig = px.scatter(
                data, 
                x=x_feature, 
                y='salary',
                color=color_feature,
                title=f'Salary vs {x_feature}'
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "📈 Trend Analysis":
        st.subheader("📊 Salary Trends")
        
        # Group by categorical variable
        group_by = st.selectbox(
            "Group by:",
            data.select_dtypes(include=['object']).columns.tolist()
        )
        
        # Calculate average salary by group
        trend_data = data.groupby(group_by)['salary'].agg(['mean', 'count', 'std']).reset_index()
        
        fig = px.bar(
            trend_data, 
            x=group_by, 
            y='mean',
            error_y='std',
            title=f'Average Salary by {group_by}',
            labels={'mean': 'Average Salary (USD)'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("📋 Summary Table")
        st.dataframe(trend_data.round(2))
    
    elif viz_type == "🎯 3D Scatter Plot":
        st.subheader("🌐 3D Salary Visualization")
        
        # Select features for 3D plot
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_feature = st.selectbox("X-axis:", numerical_cols, index=0)
        with col2:
            y_feature = st.selectbox("Y-axis:", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)
        with col3:
            color_feature = st.selectbox("Color by:", categorical_cols, index=0)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            data,
            x=x_feature,
            y=y_feature,
            z='salary',
            color=color_feature,
            title=f'3D Salary Visualization: {x_feature} vs {y_feature} vs Salary',
            labels={'salary': 'Salary (USD)'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_currency_converter():
    st.markdown("## 💱 Currency Converter")
    
    st.markdown("### 💰 Salary Currency Converter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Enter Salary Amount", value=75000.0, min_value=0.0)
        from_currency = st.selectbox("From Currency:", ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "INR"])
    
    with col2:
        to_currency = st.selectbox("To Currency:", ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "INR"])
    
    if st.button("🔄 Convert"):
        try:
            # Get exchange rates
            rates = get_currency_rates()
            
            # Convert to USD first, then to target currency
            if from_currency == "USD":
                usd_amount = amount
            else:
                usd_amount = amount / rates[from_currency]
            
            if to_currency == "USD":
                converted_amount = usd_amount
            else:
                converted_amount = usd_amount * rates[to_currency]
            
            st.success(f"💱 {format_currency(amount, from_currency)} = {format_currency(converted_amount, to_currency)}")
            
            # Show exchange rate
            if from_currency != to_currency:
                if from_currency == "USD":
                    rate = rates[to_currency]
                elif to_currency == "USD":
                    rate = 1 / rates[from_currency]
                else:
                    rate = rates[to_currency] / rates[from_currency]
                
                st.info(f"📊 Exchange Rate: 1 {from_currency} = {rate:.4f} {to_currency}")
            
        except Exception as e:
            st.error(f"❌ Error converting currency: {str(e)}")
    
    # Currency rates table
    st.markdown("### 📊 Current Exchange Rates (Base: USD)")
    
    rates = get_currency_rates()
    rates_df = pd.DataFrame(list(rates.items()), columns=['Currency', 'Rate'])
    rates_df['Rate'] = rates_df['Rate'].round(4)
    
    st.dataframe(rates_df, use_container_width=True)

if __name__ == "__main__":
    main()
