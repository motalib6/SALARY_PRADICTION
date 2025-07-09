import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def get_currency_rates():
    """
    Get exchange rates for multi-currency support
    """
    # Current exchange rates (as of 2025)
    rates = {
        'USD': 1.0000,      # Base currency
        'EUR': 0.9200,      # Euro
        'GBP': 0.7900,      # British Pound
        'JPY': 149.50,      # Japanese Yen
        'CAD': 1.3500,      # Canadian Dollar
        'AUD': 1.5200,      # Australian Dollar
        'CHF': 0.9100,      # Swiss Franc
        'CNY': 7.2500,      # Chinese Yuan
        'INR': 85.56,       # Indian Rupee
        'BRL': 5.8500,      # Brazilian Real
        'KRW': 1340.00,     # Korean Won
        'MXN': 20.45,       # Mexican Peso
        'SGD': 1.3400,      # Singapore Dollar
        'HKD': 7.8000,      # Hong Kong Dollar
        'NOK': 10.80,       # Norwegian Krone
        'SEK': 11.20,       # Swedish Krona
        'DKK': 6.8500,      # Danish Krone
        'PLN': 4.0500,      # Polish Zloty
        'RUB': 92.00,       # Russian Ruble
        'TRY': 29.50,       # Turkish Lira
    }
    
    return rates

def format_currency(amount, currency_code):
    """
    Format currency amount with proper symbols and formatting
    """
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CAD': 'C$',
        'AUD': 'A$',
        'CHF': 'CHF',
        'CNY': '¥',
        'INR': '₹',
        'BRL': 'R$',
        'KRW': '₩',
        'MXN': '$',
        'SGD': 'S$',
        'HKD': 'HK$',
        'NOK': 'kr',
        'SEK': 'kr',
        'DKK': 'kr',
        'PLN': 'zł',
        'RUB': '₽',
        'TRY': '₺'
    }
    
    symbol = currency_symbols.get(currency_code, currency_code)
    
    # Format with appropriate decimal places
    if currency_code in ['JPY', 'KRW']:
        # No decimal places for these currencies
        return f"{symbol}{amount:,.0f}"
    else:
        return f"{symbol}{amount:,.2f}"

def load_kaggle_data(dataset_path):
    """
    Load salary dataset from Kaggle with improved column mapping
    """
    try:
        # Look for CSV files in the dataset directory
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
        
        if not csv_files:
            print("No CSV files found in the dataset directory")
            return None
        
        # Try to load the first CSV file
        data_file = csv_files[0]
        print(f"Loading data from: {data_file}")
        
        # Load the data
        df = pd.read_csv(data_file)
        
        # Column mapping for different dataset formats
        column_mapping = {
            'Salary': 'salary',
            'Annual Salary': 'salary',
            'yearly_salary': 'salary',
            'wage': 'salary',
            'income': 'salary',
            'compensation': 'salary',
            'Age': 'age',
            'Gender': 'gender',
            'Sex': 'gender',
            'Education': 'education',
            'Education Level': 'education',
            'degree': 'education',
            'Experience': 'experience',
            'Years of Experience': 'experience',
            'work_experience': 'experience',
            'Job Title': 'job_title',
            'Position': 'job_title',
            'role': 'job_title',
            'Location': 'location',
            'City': 'location',
            'State': 'location',
            'Industry': 'industry',
            'Sector': 'industry',
            'Company Size': 'company_size',
            'company_size': 'company_size',
            'Remote Work': 'remote_work',
            'remote': 'remote_work',
            'work_from_home': 'remote_work'
        }
        
        # Apply column mapping before making lowercase
        df = df.rename(columns=column_mapping)
        
        # Make column names lowercase for consistency
        df.columns = df.columns.str.lower()
        
        # Find salary column (case-insensitive)
        salary_columns = ['salary', 'annual_salary', 'yearly_salary', 'wage', 'income', 'compensation']
        salary_col = None
        
        for col in df.columns:
            if col.lower() in salary_columns:
                salary_col = col
                break
        
        if salary_col is None:
            print("No salary column found in the dataset")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Rename salary column to standard name
        if salary_col != 'salary':
            df = df.rename(columns={salary_col: 'salary'})
        
        # Basic data validation and cleaning
        # Remove rows with missing salary values
        df = df.dropna(subset=['salary'])
        
        # Convert salary to numeric if it's not already
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        
        # Remove rows with invalid salary values
        df = df.dropna(subset=['salary'])
        df = df[df['salary'] > 0]
        
        # Clean and standardize other columns
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.title()
        
        if 'education' in df.columns:
            df['education'] = df['education'].astype(str).str.title()
        
        if 'industry' in df.columns:
            df['industry'] = df['industry'].astype(str).str.title()
        
        if 'location' in df.columns:
            df['location'] = df['location'].astype(str).str.title()
        
        if 'job_title' in df.columns:
            df['job_title'] = df['job_title'].astype(str).str.title()
        
        # Add missing columns with default values if needed
        required_columns = ['age', 'gender', 'education', 'experience', 'job_title', 
                           'location', 'industry', 'company_size', 'remote_work']
        
        for col in required_columns:
            if col not in df.columns:
                # Add default values for missing columns
                default_values = {
                    'age': 30,
                    'gender': 'Unknown',
                    'education': "Bachelor's",
                    'experience': 5,
                    'job_title': 'Unknown',
                    'location': 'Unknown',
                    'industry': 'Unknown',
                    'company_size': 'Medium (51-200)',
                    'remote_work': 'No'
                }
                df[col] = default_values[col]
        
        print(f"Loaded {len(df)} records with valid salary data")
        print(f"Dataset columns: {df.columns.tolist()}")
        print(f"Salary range: ${df['salary'].min():,.0f} - ${df['salary'].max():,.0f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading Kaggle data: {e}")
        return None

def validate_data_quality(df):
    """
    Validate data quality and provide statistics
    """
    quality_report = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_records': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'salary_statistics': {}
    }
    
    if 'salary' in df.columns:
        quality_report['salary_statistics'] = {
            'min': df['salary'].min(),
            'max': df['salary'].max(),
            'mean': df['salary'].mean(),
            'median': df['salary'].median(),
            'std': df['salary'].std(),
            'outliers': len(df[df['salary'] > df['salary'].quantile(0.95)])
        }
    
    return quality_report

def clean_data(df):
    """
    Clean and preprocess the data
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # Fill missing categorical values with mode
            mode_value = cleaned_df[col].mode()
            if len(mode_value) > 0:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
        else:
            # Fill missing numerical values with median
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # Handle outliers in salary (remove extreme outliers)
    if 'salary' in cleaned_df.columns:
        Q1 = cleaned_df['salary'].quantile(0.25)
        Q3 = cleaned_df['salary'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        cleaned_df = cleaned_df[
            (cleaned_df['salary'] >= lower_bound) & 
            (cleaned_df['salary'] <= upper_bound)
        ]
    
    return cleaned_df

def export_data(df, filename, format='csv'):
    """
    Export data to various formats
    """
    try:
        if format.lower() == 'csv':
            df.to_csv(filename, index=False)
        elif format.lower() == 'excel':
            df.to_excel(filename, index=False)
        elif format.lower() == 'json':
            df.to_json(filename, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data exported successfully to {filename}")
        return True
        
    except Exception as e:
        print(f"Error exporting data: {e}")
        return False

def calculate_salary_statistics(df, group_by_column=None):
    """
    Calculate comprehensive salary statistics
    """
    if 'salary' not in df.columns:
        return None
    
    stats = {}
    
    if group_by_column and group_by_column in df.columns:
        # Group statistics
        grouped_stats = df.groupby(group_by_column)['salary'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        stats['grouped_statistics'] = grouped_stats.to_dict()
        
        # Additional grouped metrics
        stats['salary_distribution_by_group'] = (
            df.groupby(group_by_column)['salary']
            .apply(lambda x: x.describe())
            .to_dict()
        )
    
    # Overall statistics
    stats['overall_statistics'] = {
        'total_records': len(df),
        'mean_salary': df['salary'].mean(),
        'median_salary': df['salary'].median(),
        'std_salary': df['salary'].std(),
        'min_salary': df['salary'].min(),
        'max_salary': df['salary'].max(),
        'salary_range': df['salary'].max() - df['salary'].min(),
        'percentiles': {
            '25th': df['salary'].quantile(0.25),
            '50th': df['salary'].quantile(0.50),
            '75th': df['salary'].quantile(0.75),
            '90th': df['salary'].quantile(0.90),
            '95th': df['salary'].quantile(0.95),
            '99th': df['salary'].quantile(0.99)
        }
    }
    
    return stats

def generate_data_report(df):
    """
    Generate a comprehensive data report
    """
    report = {
        'dataset_info': {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        },
        'data_quality': validate_data_quality(df),
        'salary_statistics': calculate_salary_statistics(df)
    }
    
    # Add categorical column analysis
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        report['categorical_analysis'] = {}
        for col in categorical_columns:
            report['categorical_analysis'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(10).to_dict(),
                'missing_values': df[col].isnull().sum()
            }
    
    return report

def convert_salary_currency(amount, from_currency, to_currency):
    """
    Convert salary from one currency to another
    """
    rates = get_currency_rates()
    
    if from_currency not in rates or to_currency not in rates:
        raise ValueError("Unsupported currency")
    
    # Convert to USD first, then to target currency
    usd_amount = amount / rates[from_currency]
    converted_amount = usd_amount * rates[to_currency]
    
    return converted_amount

def get_salary_percentile(salary, df):
    """
    Get the percentile rank of a salary within the dataset
    """
    if 'salary' not in df.columns:
        return None
    
    percentile = (df['salary'] <= salary).mean() * 100
    return round(percentile, 1)

def benchmark_salary(input_data, df):
    """
    Benchmark a salary against similar profiles in the dataset
    """
    if 'salary' not in df.columns:
        return None
    
    # Filter similar profiles
    similar_profiles = df.copy()
    
    # Filter by categorical variables if available
    categorical_filters = ['education', 'job_title', 'industry', 'location']
    
    for col in categorical_filters:
        if col in input_data and col in similar_profiles.columns:
            similar_profiles = similar_profiles[
                similar_profiles[col] == input_data[col]
            ]
    
    # Filter by experience range if available
    if 'experience' in input_data and 'experience' in similar_profiles.columns:
        exp_range = 2  # ±2 years
        similar_profiles = similar_profiles[
            (similar_profiles['experience'] >= input_data['experience'] - exp_range) &
            (similar_profiles['experience'] <= input_data['experience'] + exp_range)
        ]
    
    if len(similar_profiles) == 0:
        return None
    
    benchmark_stats = {
        'sample_size': len(similar_profiles),
        'mean_salary': similar_profiles['salary'].mean(),
        'median_salary': similar_profiles['salary'].median(),
        'std_salary': similar_profiles['salary'].std(),
        'min_salary': similar_profiles['salary'].min(),
        'max_salary': similar_profiles['salary'].max(),
        'percentiles': {
            '25th': similar_profiles['salary'].quantile(0.25),
            '75th': similar_profiles['salary'].quantile(0.75),
            '90th': similar_profiles['salary'].quantile(0.90)
        }
    }
    
    return benchmark_stats

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test currency conversion
    rates = get_currency_rates()
    print(f"Currency rates: {rates}")
    
    # Test formatting
    formatted = format_currency(75000, 'USD')
    print(f"Formatted currency: {formatted}")
    
    # Test currency conversion
    converted = convert_salary_currency(75000, 'USD', 'EUR')
    print(f"Converted salary: {converted}")
