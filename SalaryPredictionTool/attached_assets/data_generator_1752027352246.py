import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_employee_data(n_samples=5000):
    """
    Generate synthetic employee data for demonstration purposes.
    In a real-world scenario, this would be replaced with actual employee data.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Define possible values for categorical variables
    job_titles = [
        'Software Engineer', 'Data Scientist', 'Product Manager', 'Marketing Manager',
        'Sales Representative', 'HR Manager', 'Financial Analyst', 'Operations Manager',
        'UX Designer', 'DevOps Engineer', 'Business Analyst', 'Project Manager',
        'Quality Assurance', 'Customer Success Manager', 'Content Writer', 'Accountant'
    ]
    
    education_levels = [
        'High School', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD', 'Associate Degree'
    ]
    
    locations = [
        'New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston',
        'Seattle', 'Austin', 'Denver', 'Atlanta', 'Miami', 'Dallas', 'Phoenix'
    ]
    
    industries = [
        'Technology', 'Finance', 'Healthcare', 'Education', 'Retail',
        'Manufacturing', 'Consulting', 'Media', 'Non-profit', 'Government'
    ]
    
    # Define salary ranges and multipliers for different factors
    base_salary_ranges = {
        'Software Engineer': (70000, 180000),
        'Data Scientist': (80000, 200000),
        'Product Manager': (90000, 220000),
        'Marketing Manager': (60000, 150000),
        'Sales Representative': (40000, 120000),
        'HR Manager': (55000, 130000),
        'Financial Analyst': (50000, 120000),
        'Operations Manager': (65000, 160000),
        'UX Designer': (55000, 140000),
        'DevOps Engineer': (75000, 190000),
        'Business Analyst': (55000, 130000),
        'Project Manager': (70000, 170000),
        'Quality Assurance': (45000, 110000),
        'Customer Success Manager': (50000, 120000),
        'Content Writer': (35000, 80000),
        'Accountant': (40000, 100000)
    }
    
    education_multipliers = {
        'High School': 0.8,
        'Associate Degree': 0.9,
        'Bachelor\'s Degree': 1.0,
        'Master\'s Degree': 1.2,
        'PhD': 1.4
    }
    
    location_multipliers = {
        'San Francisco': 1.4,
        'New York': 1.3,
        'Seattle': 1.2,
        'Boston': 1.15,
        'Los Angeles': 1.1,
        'Chicago': 1.0,
        'Austin': 0.95,
        'Denver': 0.9,
        'Atlanta': 0.85,
        'Dallas': 0.85,
        'Miami': 0.9,
        'Phoenix': 0.8
    }
    
    industry_multipliers = {
        'Technology': 1.2,
        'Finance': 1.15,
        'Consulting': 1.1,
        'Healthcare': 1.05,
        'Media': 1.0,
        'Manufacturing': 0.95,
        'Education': 0.85,
        'Retail': 0.8,
        'Non-profit': 0.75,
        'Government': 0.9
    }
    
    # Generate data
    data = []
    
    for _ in range(n_samples):
        # Select random values
        job_title = random.choice(job_titles)
        education = random.choice(education_levels)
        location = random.choice(locations)
        industry = random.choice(industries)
        
        # Generate years of experience (weighted towards lower values)
        years_experience = int(np.random.exponential(5))
        years_experience = min(years_experience, 25)  # Cap at 25 years
        
        # Calculate base salary
        base_min, base_max = base_salary_ranges[job_title]
        base_salary = np.random.uniform(base_min, base_max)
        
        # Apply multipliers
        salary = base_salary
        salary *= education_multipliers[education]
        salary *= location_multipliers[location]
        salary *= industry_multipliers[industry]
        
        # Experience bonus (3-5% per year)
        experience_multiplier = 1 + (years_experience * np.random.uniform(0.03, 0.05))
        salary *= experience_multiplier
        
        # Add some random noise
        salary *= np.random.uniform(0.9, 1.1)
        
        # Round to nearest thousand
        salary = round(salary / 1000) * 1000
        
        data.append({
            'years_experience': years_experience,
            'education': education,
            'job_title': job_title,
            'location': location,
            'industry': industry,
            'salary': salary
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Remove extreme outliers
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]
    
    return df.reset_index(drop=True)

if __name__ == "__main__":
    # Test the data generator
    df = generate_employee_data(1000)
    print(f"Generated {len(df)} employee records")
    print("\nDataset info:")
    print(df.info())
    print("\nSample data:")
    print(df.head())
    print(f"\nSalary statistics:")
    print(df['salary'].describe())
