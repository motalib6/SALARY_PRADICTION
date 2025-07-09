import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sqlalchemy import text
import warnings
warnings.filterwarnings('ignore')

from data_generator import generate_employee_data
from model_trainer import ModelTrainer
from predictor import SalaryPredictor
from database import DatabaseManager, init_database_with_sample_data, get_database_manager
from global_data import GLOBAL_LOCATIONS, JOB_TITLES_2030, EDUCATION_LEVELS, GLOBAL_INDUSTRIES, PREDICTION_MODELS, CURRENCY_RATES, convert_currency

# Set page configuration
st.set_page_config(
    page_title="Employee Salary Predictor by SK MOTALIB",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A4A4A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the employee data from database"""
    db_manager = get_database_manager()
    if db_manager and db_manager.engine:
        df = db_manager.get_all_employees()
        if not df.empty and len(df) >= 100:  # Ensure sufficient data for ML
            return df
    # Initialize with sample data if database is empty or has insufficient data
    db_manager = init_database_with_sample_data()
    if db_manager:
        return db_manager.get_all_employees()
    # Final fallback to generated data
    return generate_employee_data(5000)

@st.cache_resource
def train_models(df):
    """Train and cache the machine learning models"""
    trainer = ModelTrainer(df)
    return trainer.train_all_models()

def main():
    st.markdown('<h1 class="main-header">💼 Employee Salary Predictor by SK MOTALIB</h1>', unsafe_allow_html=True)
    st.markdown("### Predict employee salaries using advanced machine learning algorithms")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["🏠 Home", "📊 Data Analysis", "🔮 Salary Prediction", "📈 Model Performance", "🎯 3D Visualization", "🗄️ Database Management"])
    
    # Load data
    with st.spinner("Loading employee data..."):
        df = load_data()
    
    # Train models
    with st.spinner("Training machine learning models..."):
        models, model_performance, scaler, label_encoders = train_models(df)
    
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🔮 Salary Prediction":
        show_prediction_page(models, scaler, label_encoders, df)
    elif page == "📈 Model Performance":
        show_model_performance(model_performance, models)
    elif page == "🎯 3D Visualization":
        show_3d_visualization(df, models, scaler, label_encoders)
    elif page == "🗄️ Database Management":
        show_database_management()

def show_home_page(df):
    """Display the home page with overview statistics"""
    st.markdown('<h2 class="sub-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        st.metric("Average Salary", f"${df['salary'].mean():,.0f}")
    with col3:
        st.metric("Salary Range", f"${df['salary'].min():,.0f} - ${df['salary'].max():,.0f}")
    with col4:
        st.metric("Features", len(df.columns) - 1)
    
    st.markdown("---")
    
    # Sample data preview
    st.markdown('<h3 class="sub-header">📋 Sample Data</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    # Quick insights
    st.markdown('<h3 class="sub-header">💡 Quick Insights</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary by education
        fig_edu = px.box(df, x='education', y='salary', title='Salary Distribution by Education Level')
        fig_edu.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_edu, use_container_width=True)
    
    with col2:
        # Salary by experience
        fig_exp = px.scatter(df, x='years_experience', y='salary', 
                           color='job_title', title='Salary vs Experience')
        st.plotly_chart(fig_exp, use_container_width=True)

def show_data_analysis(df):
    """Display comprehensive data analysis"""
    st.markdown('<h2 class="sub-header">📊 Data Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    # Statistical summary
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Correlation matrix
    st.markdown("### 🔗 Feature Correlations")
    
    # Prepare numerical data for correlation
    df_numeric = df.copy()
    categorical_columns = ['job_title', 'education', 'location', 'industry']
    
    # Label encode categorical variables for correlation
    le = LabelEncoder()
    for col in categorical_columns:
        df_numeric[col] = le.fit_transform(df_numeric[col])
    
    correlation_matrix = df_numeric.corr()
    
    fig_corr = px.imshow(correlation_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="Feature Correlation Matrix")
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Salary Distribution")
        fig_hist = px.histogram(df, x='salary', nbins=50, title='Salary Distribution')
        fig_hist.update_layout(xaxis_title="Salary ($)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Salary by Industry")
        fig_industry = px.violin(df, x='industry', y='salary', title='Salary Distribution by Industry')
        fig_industry.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_industry, use_container_width=True)
    
    # Advanced visualizations
    st.markdown("### 🔍 Advanced Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Location Analysis", "Job Title Analysis", "Experience Analysis"])
    
    with tab1:
        avg_salary_location = df.groupby('location')['salary'].mean().sort_values(ascending=False)
        fig_location = px.bar(x=avg_salary_location.index, y=avg_salary_location.values,
                             title='Average Salary by Location')
        fig_location.update_layout(xaxis_title="Location", yaxis_title="Average Salary ($)")
        st.plotly_chart(fig_location, use_container_width=True)
    
    with tab2:
        avg_salary_job = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
        fig_job = px.bar(x=avg_salary_job.values, y=avg_salary_job.index,
                        orientation='h', title='Average Salary by Job Title')
        fig_job.update_layout(xaxis_title="Average Salary ($)", yaxis_title="Job Title")
        st.plotly_chart(fig_job, use_container_width=True)
    
    with tab3:
        # Create experience bins
        df['experience_bin'] = pd.cut(df['years_experience'], 
                                    bins=[0, 2, 5, 10, 15, 20, 25], 
                                    labels=['0-2', '3-5', '6-10', '11-15', '16-20', '21-25'])
        exp_salary = df.groupby('experience_bin')['salary'].mean()
        fig_exp_bin = px.bar(x=exp_salary.index, y=exp_salary.values,
                            title='Average Salary by Experience Range')
        fig_exp_bin.update_layout(xaxis_title="Years of Experience", yaxis_title="Average Salary ($)")
        st.plotly_chart(fig_exp_bin, use_container_width=True)

def show_prediction_page(models, scaler, label_encoders, df):
    """Display the salary prediction interface with comprehensive global inputs"""
    st.markdown('<h2 class="sub-header">🔮 Global Salary Prediction Tool</h2>', unsafe_allow_html=True)
    
    predictor = SalaryPredictor(models, scaler, label_encoders)
    
    # Enhanced Input form with global data
    st.markdown("### 🌍 Enter Employee Details")
    st.markdown("*Complete global coverage with 2030 job projections and worldwide locations*")
    
    # First row - Experience and Education
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Experience & Education**")
        years_experience = st.slider("Years of Experience", 0, 25, 5, 
                                    help="Professional work experience in years")
        education = st.selectbox("Education Level", 
                                options=EDUCATION_LEVELS,
                                help="Highest level of education completed")
    
    with col2:
        st.markdown("**💼 Career & Model**")
        job_title = st.selectbox("Job Title (2030 Projections)", 
                                options=sorted(JOB_TITLES_2030),
                                help="Select from 200+ job titles based on 2030 employment projections")
        model_choice = st.selectbox("Prediction Model", 
                                   options=PREDICTION_MODELS,
                                   help="Choose the machine learning model for prediction")
    
    # Second row - Location Selection (Global)
    st.markdown("**🌍 Global Location Selection**")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        selected_country = st.selectbox("Country", 
                                       options=list(GLOBAL_LOCATIONS.keys()),
                                       help="Select from major countries worldwide")
    
    with col4:
        # Get states/regions for selected country
        if selected_country in GLOBAL_LOCATIONS:
            location_type = list(GLOBAL_LOCATIONS[selected_country].keys())[0]  # states, provinces, countries, etc.
            state_options = list(GLOBAL_LOCATIONS[selected_country][location_type].keys())
            selected_state = st.selectbox(f"{location_type.title()}", 
                                         options=state_options,
                                         help=f"Select {location_type} within {selected_country}")
        else:
            selected_state = st.selectbox("State/Region", options=["N/A"])
    
    with col5:
        # Get cities for selected state
        if selected_country in GLOBAL_LOCATIONS and selected_state:
            location_type = list(GLOBAL_LOCATIONS[selected_country].keys())[0]
            if selected_state in GLOBAL_LOCATIONS[selected_country][location_type]:
                city_options = GLOBAL_LOCATIONS[selected_country][location_type][selected_state]
                selected_city = st.selectbox("City", 
                                           options=city_options,
                                           help=f"Select city within {selected_state}")
                # Create location string
                location = f"{selected_city}, {selected_state}, {selected_country}"
            else:
                selected_city = st.selectbox("City", options=["N/A"])
                location = f"{selected_state}, {selected_country}"
        else:
            selected_city = st.selectbox("City", options=["N/A"])
            location = selected_country
    
    # Third row - Industry and Currency Selection
    col6, col7 = st.columns(2)
    
    with col6:
        st.markdown("**🏭 Industry Selection**")
        industry = st.selectbox("Industry (Global Coverage)", 
                               options=sorted(GLOBAL_INDUSTRIES),
                               help="Select from 100+ industries covering all global sectors")
    
    with col7:
        st.markdown("**💱 Currency Display**")
        selected_currency = st.selectbox("Display Currency", 
                                        options=list(CURRENCY_RATES.keys()),
                                        index=0,
                                        help="Choose currency for salary display")
    
    # Display current selection summary
    with st.expander("📋 Current Selection Summary", expanded=False):
        st.write(f"**Experience:** {years_experience} years")
        st.write(f"**Education:** {education}")
        st.write(f"**Job Title:** {job_title}")
        st.write(f"**Location:** {location}")
        st.write(f"**Industry:** {industry}")
        st.write(f"**Model:** {model_choice}")
        st.write(f"**Currency:** {CURRENCY_RATES[selected_currency]['name']} ({selected_currency})")
    
    # Map global inputs to database format for prediction
    # Use closest match from database for accurate prediction
    db_education = education if education in df['education'].unique() else df['education'].mode()[0]
    db_job_title = job_title if job_title in df['job_title'].unique() else df['job_title'].mode()[0]
    db_location = location if location in df['location'].unique() else df['location'].mode()[0]
    db_industry = industry if industry in df['industry'].unique() else df['industry'].mode()[0]
    
    # Prediction button
    if st.button("🚀 Predict Global Salary", type="primary"):
        with st.spinner("Calculating salary prediction using global data..."):
            # Create input data using mapped database values
            input_data = {
                'years_experience': years_experience,
                'education': db_education,
                'job_title': db_job_title,
                'location': db_location,
                'industry': db_industry
            }
            
            # Get prediction
            prediction = predictor.predict_salary(input_data, model_choice)
            
            # Display results with global context
            st.markdown("---")
            st.markdown("### 🎯 Global Salary Prediction Results")
            
            # Convert salary to selected currency
            converted_amount, currency_code, currency_symbol = convert_currency(prediction, selected_currency)
            
            # Main prediction display with currency conversion
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if selected_currency == "USD":
                    st.metric("Predicted Salary", f"${prediction:,.0f}")
                else:
                    st.metric("Predicted Salary", f"{currency_symbol}{converted_amount:,.0f} {currency_code}")
            
            with col2:
                # Calculate percentile
                percentile = (df['salary'] < prediction).mean() * 100
                st.metric("Global Percentile", f"{percentile:.1f}%")
            
            with col3:
                # Market comparison using mapped data
                similar_profiles = df[
                    (df['education'] == db_education) & 
                    (df['job_title'] == db_job_title) &
                    (abs(df['years_experience'] - years_experience) <= 2)
                ]
                if len(similar_profiles) > 0:
                    market_avg = similar_profiles['salary'].mean()
                    difference = ((prediction - market_avg) / market_avg) * 100
                    st.metric("vs Similar Roles", f"{difference:+.1f}%")
                else:
                    st.metric("vs Similar Roles", "N/A")
            
            with col4:
                # Experience level indicator
                if years_experience <= 2:
                    level = "Entry Level"
                elif years_experience <= 5:
                    level = "Mid Level"
                elif years_experience <= 10:
                    level = "Senior Level"
                else:
                    level = "Executive Level"
                st.metric("Career Level", level)
            
            # Multi-currency display section
            if selected_currency != "USD":
                st.markdown("### 💱 Multi-Currency Display")
                
                # Show major currency conversions
                major_currencies = ["USD", "INR", "EUR", "GBP", "JPY", "CNY"]
                currency_cols = st.columns(len(major_currencies))
                
                for i, curr in enumerate(major_currencies):
                    if curr in CURRENCY_RATES:
                        conv_amt, _, curr_symbol = convert_currency(prediction, curr)
                        with currency_cols[i]:
                            if curr == "JPY" or curr == "KRW":
                                st.metric(f"{curr}", f"{curr_symbol}{conv_amt:,.0f}")
                            else:
                                st.metric(f"{curr}", f"{curr_symbol}{conv_amt:,.0f}")
                
                # Exchange rate information
                st.info(f"Exchange rates as of June 27, 2025: 1 USD = {CURRENCY_RATES[selected_currency]['rate']:.2f} {selected_currency}")
            
            # Global context information
            st.markdown("### 🌍 Global Profile Summary")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.info(f"""
                **Selected Profile:**
                - **Location:** {location}
                - **Job Title:** {job_title}
                - **Industry:** {industry}
                - **Education:** {education}
                - **Experience:** {years_experience} years
                """)
            
            with col_right:
                # Show prediction in selected currency for the summary
                if selected_currency == "USD":
                    salary_display = f"${prediction:,.0f}"
                else:
                    salary_display = f"{currency_symbol}{converted_amount:,.0f} {currency_code}"
                
                st.success(f"""
                **Prediction Details:**
                - **Model Used:** {model_choice}
                - **Predicted Salary:** {salary_display}
                - **Career Level:** {level}
                - **Global Ranking:** Top {100-percentile:.1f}%
                """)
            
            # Show mapping information if different from input
            mapping_info = []
            if db_education != education:
                mapping_info.append(f"Education mapped to: {db_education}")
            if db_job_title != job_title:
                mapping_info.append(f"Job title mapped to: {db_job_title}")
            if db_location != location:
                mapping_info.append(f"Location mapped to: {db_location}")
            if db_industry != industry:
                mapping_info.append(f"Industry mapped to: {db_industry}")
            
            if mapping_info:
                with st.expander("Data Mapping Information", expanded=False):
                    st.warning("**Note:** Some inputs were mapped to closest available categories in our database:")
                    for info in mapping_info:
                        st.write(f"• {info}")
            
            # Confidence interval (simplified)
            st.markdown("### 📊 Prediction Analysis")
            
            # Create a range based on model uncertainty
            uncertainty = 0.15  # 15% uncertainty
            lower_bound = prediction * (1 - uncertainty)
            upper_bound = prediction * (1 + uncertainty)
            
            st.info(f"**Confidence Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")
            
            # Show similar profiles
            if len(similar_profiles) > 0:
                st.markdown("### 👥 Similar Profiles in Dataset")
                display_cols = ['years_experience', 'education', 'job_title', 'location', 'industry', 'salary']
                st.dataframe(similar_profiles[display_cols].head(5), use_container_width=True)

def show_model_performance(model_performance, models):
    """Display model performance metrics and comparisons"""
    st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Performance metrics table
    st.markdown("### 🎯 Model Comparison")
    
    performance_df = pd.DataFrame(model_performance).T
    performance_df = performance_df.round(4)
    
    # Color code the best performing model for each metric
    st.dataframe(performance_df.style.highlight_max(axis=0, subset=['R² Score']).highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE']), 
                use_container_width=True)
    
    # Visualization of model performance
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score comparison
        r2_scores = [model_performance[model]['R² Score'] for model in model_performance.keys()]
        fig_r2 = px.bar(x=list(model_performance.keys()), y=r2_scores,
                       title='R² Score Comparison', 
                       labels={'y': 'R² Score', 'x': 'Model'})
        fig_r2.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # MAE comparison
        mae_scores = [model_performance[model]['MAE'] for model in model_performance.keys()]
        fig_mae = px.bar(x=list(model_performance.keys()), y=mae_scores,
                        title='Mean Absolute Error Comparison',
                        labels={'y': 'MAE ($)', 'x': 'Model'})
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # Feature importance (for tree-based models)
    st.markdown("### 🌟 Feature Importance")
    
    tab1, tab2 = st.tabs(["Random Forest", "Gradient Boosting"])
    
    with tab1:
        if 'Random Forest' in models:
            rf_model = models['Random Forest']
            feature_names = ['years_experience', 'education', 'job_title', 'location', 'industry']
            
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig_rf_imp = px.bar(importance_df, x='Importance', y='Feature',
                                   orientation='h', title='Random Forest Feature Importance')
                st.plotly_chart(fig_rf_imp, use_container_width=True)
            else:
                st.info("Feature importance not available for this model configuration.")
    
    with tab2:
        if 'Gradient Boosting' in models:
            gb_model = models['Gradient Boosting']
            feature_names = ['years_experience', 'education', 'job_title', 'location', 'industry']
            
            if hasattr(gb_model, 'feature_importances_'):
                importances = gb_model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig_gb_imp = px.bar(importance_df, x='Importance', y='Feature',
                                   orientation='h', title='Gradient Boosting Feature Importance')
                st.plotly_chart(fig_gb_imp, use_container_width=True)
            else:
                st.info("Feature importance not available for this model configuration.")
    
    # Model recommendations
    st.markdown("### 💡 Model Recommendations")
    
    best_r2_model = max(model_performance.keys(), key=lambda x: model_performance[x]['R² Score'])
    best_mae_model = min(model_performance.keys(), key=lambda x: model_performance[x]['MAE'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Best Overall Performance:** {best_r2_model}")
        st.write(f"R² Score: {model_performance[best_r2_model]['R² Score']:.4f}")
    
    with col2:
        st.info(f"**Most Accurate Predictions:** {best_mae_model}")
        st.write(f"Mean Absolute Error: ${model_performance[best_mae_model]['MAE']:,.0f}")

def show_3d_visualization(df, models, scaler, label_encoders):
    """Display advanced 3D visualizations and modeling features"""
    st.markdown('<h2 class="sub-header">🎯 3D Visualization & Advanced Modeling</h2>', unsafe_allow_html=True)
    
    # Create tabs for different 3D visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "3D Salary Surface", "Interactive 3D Scatter", "Model Decision Boundaries", 
        "Feature Space Explorer", "Prediction Confidence 3D"
    ])
    
    with tab1:
        st.markdown("### 🌐 3D Salary Surface Modeling")
        st.markdown("Explore how salary varies across multiple dimensions simultaneously")
        
        # Select features for 3D surface
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-Axis Feature", 
                                   ["years_experience", "education", "job_title", "location", "industry"],
                                   key="surface_x")
        with col2:
            y_feature = st.selectbox("Y-Axis Feature", 
                                   ["years_experience", "education", "job_title", "location", "industry"],
                                   index=1, key="surface_y")
        
        if x_feature != y_feature:
            # Create 3D surface plot
            if x_feature == "years_experience":
                x_vals = np.linspace(df[x_feature].min(), df[x_feature].max(), 20)
                x_data = df[x_feature].values
            else:
                unique_vals = df[x_feature].unique()
                x_vals = np.arange(len(unique_vals))
                x_data = df[x_feature].map({val: i for i, val in enumerate(unique_vals)}).values
            
            if y_feature == "years_experience":
                y_vals = np.linspace(df[y_feature].min(), df[y_feature].max(), 20)
                y_data = df[y_feature].values
            else:
                unique_vals = df[y_feature].unique()
                y_vals = np.arange(len(unique_vals))
                y_data = df[y_feature].map({val: i for i, val in enumerate(unique_vals)}).values
            
            # Create meshgrid for surface
            X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
            
            # Calculate average salary for each combination
            Z_mesh = np.zeros_like(X_mesh)
            for i in range(len(x_vals)):
                for j in range(len(y_vals)):
                    # Find points close to this grid point
                    if x_feature == "years_experience":
                        x_mask = np.abs(x_data - x_vals[i]) <= 1
                    else:
                        x_mask = x_data == x_vals[i]
                    
                    if y_feature == "years_experience":
                        y_mask = np.abs(y_data - y_vals[j]) <= 1
                    else:
                        y_mask = y_data == y_vals[j]
                    
                    mask = x_mask & y_mask
                    if np.any(mask):
                        Z_mesh[j, i] = df.loc[mask, 'salary'].mean()
                    else:
                        Z_mesh[j, i] = df['salary'].mean()
            
            # Create 3D surface plot
            fig_surface = go.Figure(data=[go.Surface(
                x=X_mesh, y=Y_mesh, z=Z_mesh,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Salary ($)")
            )])
            
            fig_surface.update_layout(
                title=f'3D Salary Surface: {x_feature.replace("_", " ").title()} vs {y_feature.replace("_", " ").title()}',
                scene=dict(
                    xaxis_title=x_feature.replace("_", " ").title(),
                    yaxis_title=y_feature.replace("_", " ").title(),
                    zaxis_title="Salary ($)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
            
            st.plotly_chart(fig_surface, use_container_width=True)
            
            # Add interpretation
            st.info(f"This 3D surface shows how salary varies with {x_feature} and {y_feature}. "
                   f"Higher peaks indicate higher average salaries for those combinations.")
    
    with tab2:
        st.markdown("### 🎨 Interactive 3D Scatter Plot")
        st.markdown("Explore relationships between three features simultaneously")
        
        # Feature selection for 3D scatter
        col1, col2, col3 = st.columns(3)
        with col1:
            x_scatter = st.selectbox("X-Axis", ["years_experience", "salary"], key="scatter_x")
        with col2:
            y_scatter = st.selectbox("Y-Axis", ["years_experience", "salary"], index=1, key="scatter_y")
        with col3:
            z_scatter = st.selectbox("Z-Axis", ["salary", "years_experience"], key="scatter_z")
        
        color_by = st.selectbox("Color By", ["education", "job_title", "location", "industry"], key="scatter_color")
        
        # Create 3D scatter plot
        fig_scatter = px.scatter_3d(
            df, 
            x=x_scatter, y=y_scatter, z=z_scatter,
            color=color_by,
            title=f'3D Scatter: {x_scatter} vs {y_scatter} vs {z_scatter}',
            hover_data=['education', 'job_title', 'location', 'industry']
        )
        
        fig_scatter.update_layout(
            scene=dict(
                xaxis_title=x_scatter.replace("_", " ").title(),
                yaxis_title=y_scatter.replace("_", " ").title(),
                zaxis_title=z_scatter.replace("_", " ").title(),
            ),
            height=600
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Interactive controls
        st.markdown("#### 🎛️ Interactive Controls")
        if st.button("Rotate View", key="rotate_scatter"):
            st.info("Click and drag the plot to rotate the 3D view!")
    
    with tab3:
        st.markdown("### 🧠 Model Decision Boundaries in 3D")
        st.markdown("Visualize how different ML models make predictions across feature space")
        
        # Model selection
        model_name = st.selectbox("Select Model", list(models.keys()), key="boundary_model")
        if not model_name:
            model_name = list(models.keys())[0]
        
        # Create prediction surface for two features
        feature1 = "years_experience"
        feature2_options = ["education", "job_title", "location", "industry"]
        feature2 = st.selectbox("Second Feature", feature2_options, key="boundary_feature2")
        
        # Get unique values for the categorical feature
        if feature2 in ["education", "job_title", "location", "industry"]:
            unique_vals = df[feature2].unique()[:5]  # Limit to 5 categories for clarity
            
            # Create prediction surfaces for each category
            fig_boundary = go.Figure()
            
            for i, category in enumerate(unique_vals):
                # Create grid
                exp_range = np.linspace(0, 25, 20)
                predictions = []
                
                for exp in exp_range:
                    # Use most common values for other features
                    input_data = {
                        'years_experience': exp,
                        'education': df['education'].mode()[0],
                        'job_title': df['job_title'].mode()[0],
                        'location': df['location'].mode()[0],
                        'industry': df['industry'].mode()[0]
                    }
                    input_data[feature2] = category
                    
                    predictor = SalaryPredictor(models, scaler, label_encoders)
                    pred = predictor.predict_salary(input_data, model_name)
                    predictions.append(pred)
                
                # Add trace for this category
                fig_boundary.add_trace(go.Scatter3d(
                    x=exp_range,
                    y=[i] * len(exp_range),
                    z=predictions,
                    mode='lines+markers',
                    name=f'{category}',
                    line=dict(width=4),
                    marker=dict(size=4)
                ))
            
            fig_boundary.update_layout(
                title=f'{model_name} Decision Boundaries: Experience vs {feature2}',
                scene=dict(
                    xaxis_title='Years Experience',
                    yaxis_title=feature2.replace("_", " ").title(),
                    zaxis_title='Predicted Salary ($)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
            
            st.plotly_chart(fig_boundary, use_container_width=True)
            
            st.info(f"This visualization shows how the {model_name} model predicts salary "
                   f"for different levels of experience across various {feature2} categories.")
    
    with tab4:
        st.markdown("### 🔍 Feature Space Explorer")
        st.markdown("Explore the high-dimensional feature space in 3D projections")
        
        # PCA or feature selection
        analysis_type = st.radio("Analysis Type", ["Principal Components", "Feature Selection"], key="explorer_type")
        
        if analysis_type == "Principal Components":
            
            # Prepare data for PCA
            df_numeric = df.copy()
            categorical_columns = ['job_title', 'education', 'location', 'industry']
            
            # Label encode categorical variables
            le = LabelEncoder()
            for col in categorical_columns:
                df_numeric[col] = le.fit_transform(df_numeric[col])
            
            # Perform PCA
            features = ['years_experience', 'education', 'job_title', 'location', 'industry']
            X_pca = df_numeric[features].values
            
            pca = PCA(n_components=3)
            X_pca_transformed = pca.fit_transform(X_pca)
            
            # Create 3D PCA plot
            fig_pca = px.scatter_3d(
                x=X_pca_transformed[:, 0],
                y=X_pca_transformed[:, 1],
                z=X_pca_transformed[:, 2],
                color=df['salary'],
                title='3D PCA Projection of Feature Space',
                labels={'color': 'Salary ($)'},
                color_continuous_scale='Viridis'
            )
            
            fig_pca.update_layout(
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)',
                ),
                height=600
            )
            
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Show explained variance
            st.markdown("#### 📊 Principal Component Analysis Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PC1 Variance Explained", f"{pca.explained_variance_ratio_[0]:.2%}")
            with col2:
                st.metric("PC2 Variance Explained", f"{pca.explained_variance_ratio_[1]:.2%}")
            with col3:
                st.metric("PC3 Variance Explained", f"{pca.explained_variance_ratio_[2]:.2%}")
            
            st.info(f"Total variance explained by first 3 components: {pca.explained_variance_ratio_[:3].sum():.2%}")
        
        else:
            # Feature selection visualization
            st.markdown("Select three features to explore their relationships:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                feat1 = st.selectbox("Feature 1", ["years_experience", "salary"], key="feat1")
            with col2:
                feat2 = st.selectbox("Feature 2", ["years_experience", "salary"], index=1, key="feat2")
            with col3:
                feat3 = st.selectbox("Feature 3", ["education", "job_title", "location", "industry"], key="feat3")
            
            # Create feature space plot
            fig_feat = px.scatter_3d(
                df,
                x=feat1, y=feat2, z=feat3,
                color='salary',
                title=f'Feature Space: {feat1} vs {feat2} vs {feat3}',
                color_continuous_scale='Plasma'
            )
            
            fig_feat.update_layout(height=600)
            st.plotly_chart(fig_feat, use_container_width=True)
    
    with tab5:
        st.markdown("### 📈 Prediction Confidence in 3D")
        st.markdown("Visualize prediction confidence across different scenarios")
        
        # Interactive prediction with confidence
        st.markdown("#### 🎯 Interactive Prediction Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            exp_range = st.slider("Experience Range", 0, 25, (2, 15), key="conf_exp")
            education_sel = st.selectbox("Education", df['education'].unique(), key="conf_edu")
        
        with col2:
            job_sel = st.selectbox("Job Title", df['job_title'].unique()[:10], key="conf_job")
            location_sel = st.selectbox("Location", df['location'].unique()[:8], key="conf_loc")
        
        # Generate prediction confidence surface
        if st.button("Generate 3D Confidence Map", key="gen_confidence"):
            with st.spinner("Generating 3D confidence visualization..."):
                exp_vals = np.linspace(exp_range[0], exp_range[1], 15)
                industries = df['industry'].unique()[:5]
                
                # Create prediction surface
                predictions = []
                confidence_lower = []
                confidence_upper = []
                exp_grid = []
                industry_grid = []
                
                predictor = SalaryPredictor(models, scaler, label_encoders)
                
                for i, exp in enumerate(exp_vals):
                    for j, industry in enumerate(industries):
                        input_data = {
                            'years_experience': exp,
                            'education': education_sel,
                            'job_title': job_sel,
                            'location': location_sel,
                            'industry': industry
                        }
                        
                        # Get prediction with confidence
                        pred = predictor.predict_salary(input_data, 'Random Forest')
                        mean_pred, (ci_lower, ci_upper) = predictor.get_prediction_confidence(input_data)
                        
                        predictions.append(pred)
                        confidence_lower.append(ci_lower)
                        confidence_upper.append(ci_upper)
                        exp_grid.append(exp)
                        industry_grid.append(j)
                
                # Create 3D confidence plot
                fig_conf = go.Figure()
                
                # Add prediction surface
                fig_conf.add_trace(go.Scatter3d(
                    x=exp_grid,
                    y=industry_grid,
                    z=predictions,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=predictions,
                        colorscale='Viridis',
                        colorbar=dict(title="Predicted Salary ($)")
                    ),
                    name='Predictions'
                ))
                
                # Add confidence intervals as error bars
                for i in range(len(predictions)):
                    fig_conf.add_trace(go.Scatter3d(
                        x=[exp_grid[i], exp_grid[i]],
                        y=[industry_grid[i], industry_grid[i]],
                        z=[confidence_lower[i], confidence_upper[i]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False,
                        opacity=0.6
                    ))
                
                fig_conf.update_layout(
                    title='3D Prediction Confidence Map',
                    scene=dict(
                        xaxis_title='Years Experience',
                        yaxis_title='Industry Index',
                        zaxis_title='Salary ($)',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_conf, use_container_width=True)
                
                # Show confidence statistics
                avg_confidence_range = np.mean(np.array(confidence_upper) - np.array(confidence_lower))
                st.metric("Average Confidence Range", f"${avg_confidence_range:,.0f}")
                
                st.info("Red lines show confidence intervals. Longer lines indicate higher prediction uncertainty.")

def show_database_management():
    """Display database management interface"""
    st.markdown('<h2 class="sub-header">🗄️ Database Management</h2>', unsafe_allow_html=True)
    
    # Initialize database manager
    db_manager = get_database_manager()
    
    if not db_manager or not db_manager.engine:
        st.error("Database connection failed. Please check your PostgreSQL configuration.")
        st.info("Make sure your DATABASE_URL environment variable is properly set.")
        return
    
    # Create tabs for different database operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Database Overview", "View Data", "Add Employee", "Update Records", "Analytics"
    ])
    
    with tab1:
        st.markdown("### 📊 Database Overview")
        
        # Get database statistics
        stats = db_manager.get_salary_statistics()
        table_info = db_manager.get_table_info()
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Employees", f"{stats.get('total_employees', 0):,}")
            with col2:
                st.metric("Average Salary", f"${stats.get('avg_salary', 0):,.0f}")
            with col3:
                st.metric("Min Salary", f"${stats.get('min_salary', 0):,.0f}")
            with col4:
                st.metric("Max Salary", f"${stats.get('max_salary', 0):,.0f}")
            
            st.markdown("---")
            
            # Show table structure
            st.markdown("### 🏗️ Table Structure")
            if table_info.get('columns'):
                structure_data = {'Column': [col[0] for col in table_info['columns']], 
                                'Data Type': [col[1] for col in table_info['columns']]}
                structure_df = pd.DataFrame(structure_data)
                st.dataframe(structure_df, use_container_width=True)
            
            # Refresh data button
            if st.button("🔄 Refresh Database Statistics"):
                st.cache_data.clear()
                st.rerun()
    
    with tab2:
        st.markdown("### 👀 View Employee Data")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            education_filter = st.selectbox("Filter by Education", 
                                          ["All"] + list(db_manager.get_all_employees()['education'].unique()) 
                                          if not db_manager.get_all_employees().empty else ["All"])
            
        with col2:
            job_title_filter = st.selectbox("Filter by Job Title", 
                                          ["All"] + list(db_manager.get_all_employees()['job_title'].unique())
                                          if not db_manager.get_all_employees().empty else ["All"])
        
        # Apply filters
        if education_filter == "All" and job_title_filter == "All":
            filtered_data = db_manager.get_all_employees()
        else:
            filters = {}
            if education_filter != "All":
                filters['education'] = education_filter
            if job_title_filter != "All":
                filters['job_title'] = job_title_filter
            
            filtered_data = db_manager.get_employees_by_criteria(**filters)
        
        if not filtered_data.empty:
            st.markdown(f"**Showing {len(filtered_data)} employees**")
            st.dataframe(filtered_data, use_container_width=True)
            
            # Download data
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"employee_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data found matching the selected filters.")
    
    with tab3:
        st.markdown("### ➕ Add New Employee")
        
        with st.form("add_employee_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
                new_education = st.selectbox("Education Level", 
                                           ['High School', "Bachelor's Degree", "Master's Degree", 'PhD', 'Associate Degree'])
                new_job_title = st.text_input("Job Title", value="Software Engineer")
            
            with col2:
                new_location = st.selectbox("Location", 
                                          ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston',
                                           'Seattle', 'Austin', 'Denver', 'Atlanta', 'Miami', 'Dallas', 'Phoenix'])
                new_industry = st.selectbox("Industry", 
                                          ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail',
                                           'Manufacturing', 'Consulting', 'Media', 'Non-profit', 'Government'])
                new_salary = st.number_input("Salary ($)", min_value=20000, max_value=500000, value=75000)
            
            submit_button = st.form_submit_button("Add Employee")
            
            if submit_button:
                success = db_manager.add_new_employee(
                    new_experience, new_education, new_job_title, 
                    new_location, new_industry, new_salary
                )
                
                if success:
                    st.success("Employee added successfully!")
                    st.cache_data.clear()  # Clear cache to refresh data
                else:
                    st.error("Failed to add employee. Please try again.")
    
    with tab4:
        st.markdown("### ✏️ Update Employee Records")
        
        # Get all employees for selection
        all_employees = db_manager.get_all_employees()
        
        if not all_employees.empty:
            # Employee selection
            employee_options = []
            for _, emp in all_employees.iterrows():
                employee_options.append(f"ID: {emp['id']} - {emp['job_title']} - {emp['education']} - ${emp['salary']:,.0f}")
            
            selected_employee = st.selectbox("Select Employee to Update", employee_options)
            
            if selected_employee:
                emp_id = int(selected_employee.split("ID: ")[1].split(" -")[0])
                employee_data = all_employees[all_employees['id'] == emp_id].iloc[0]
                
                st.markdown(f"**Current Data for Employee ID {emp_id}:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Experience:** {employee_data['years_experience']} years")
                    st.write(f"**Education:** {employee_data['education']}")
                
                with col2:
                    st.write(f"**Job Title:** {employee_data['job_title']}")
                    st.write(f"**Location:** {employee_data['location']}")
                
                with col3:
                    st.write(f"**Industry:** {employee_data['industry']}")
                    st.write(f"**Current Salary:** ${employee_data['salary']:,.0f}")
                
                st.markdown("---")
                
                # Update salary form
                with st.form("update_salary_form"):
                    new_salary = st.number_input("New Salary ($)", 
                                               min_value=20000, 
                                               max_value=500000, 
                                               value=int(employee_data['salary']))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        update_button = st.form_submit_button("Update Salary")
                    
                    with col2:
                        delete_button = st.form_submit_button("Delete Employee", type="secondary")
                    
                    if update_button:
                        success = db_manager.update_employee_salary(emp_id, new_salary)
                        if success:
                            st.success("Salary updated successfully!")
                            st.cache_data.clear()
                        else:
                            st.error("Failed to update salary.")
                    
                    if delete_button:
                        success = db_manager.delete_employee(emp_id)
                        if success:
                            st.success("Employee deleted successfully!")
                            st.cache_data.clear()
                        else:
                            st.error("Failed to delete employee.")
        else:
            st.info("No employees found in the database.")
    
    with tab5:
        st.markdown("### 📈 Database Analytics")
        
        # Category analysis
        analysis_category = st.selectbox("Analyze by Category", 
                                       ["education", "job_title", "location", "industry"])
        
        category_data = db_manager.get_salary_by_category(analysis_category)
        
        if not category_data.empty:
            st.markdown(f"#### Salary Analysis by {analysis_category.replace('_', ' ').title()}")
            
            # Display table
            st.dataframe(category_data.round(2), use_container_width=True)
            
            # Create visualization
            fig_category = px.bar(
                category_data, 
                x=analysis_category, 
                y='avg_salary',
                title=f'Average Salary by {analysis_category.replace("_", " ").title()}',
                labels={'avg_salary': 'Average Salary ($)', analysis_category: analysis_category.replace('_', ' ').title()}
            )
            fig_category.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Salary distribution
            st.markdown("#### Salary Range Analysis")
            fig_range = px.bar(
                category_data,
                x=analysis_category,
                y=['min_salary', 'avg_salary', 'max_salary'],
                title=f'Salary Range by {analysis_category.replace("_", " ").title()}',
                barmode='group'
            )
            fig_range.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_range, use_container_width=True)
        else:
            st.info("No data available for analysis.")
        
        # Database maintenance
        st.markdown("---")
        st.markdown("#### 🔧 Database Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Regenerate Sample Data"):
                with st.spinner("Regenerating sample data..."):
                    sample_data = generate_employee_data(5000)
                    success = db_manager.insert_employee_data(sample_data)
                    if success:
                        st.success("Sample data regenerated successfully!")
                        st.cache_data.clear()
                    else:
                        st.error("Failed to regenerate sample data.")
        
        with col2:
            if st.button("🧹 Clear All Data"):
                if st.button("⚠️ Confirm Clear All Data"):
                    with db_manager.engine.connect() as conn:
                        conn.execute(text("DELETE FROM employees"))
                        conn.commit()
                    st.success("All data cleared from database!")
                    st.cache_data.clear()

if __name__ == "__main__":
    main()
