import os
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st

DATABASE_URL = os.getenv('DATABASE_URL')

class DatabaseManager:
    """Simplified database manager for PostgreSQL operations"""
    
    def __init__(self):
        """Initialize database connection"""
        self.engine = None
        
        if not DATABASE_URL:
            print("DATABASE_URL not found in environment variables")
            return
            
        try:
            self.engine = create_engine(DATABASE_URL)
            self.create_employees_table()
            print("Database connection established successfully")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.engine = None
    
    def create_employees_table(self):
        """Create employees table if it doesn't exist"""
        if not self.engine:
            return False
            
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS employees (
                id SERIAL PRIMARY KEY,
                years_experience INTEGER NOT NULL,
                education VARCHAR(100) NOT NULL,
                job_title VARCHAR(150) NOT NULL,
                location VARCHAR(100) NOT NULL,
                industry VARCHAR(100) NOT NULL,
                salary DECIMAL(10,2) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            return True
            
        except Exception as e:
            print(f"Error creating table: {e}")
            return False
    
    def insert_employee_data(self, df):
        """Insert employee data from DataFrame into database"""
        if not self.engine:
            print("Database connection not available")
            return False
            
        try:
            # Clear existing data
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM employees"))
                conn.commit()
            
            # Insert new data using pandas to_sql
            df.to_sql('employees', self.engine, if_exists='append', index=False, method='multi')
            print(f"Successfully inserted {len(df)} employee records")
            return True
            
        except Exception as e:
            print(f"Error inserting data: {e}")
            return False
    
    def get_all_employees(self):
        """Retrieve all employee data as DataFrame"""
        if not self.engine:
            return pd.DataFrame()
            
        try:
            query = "SELECT * FROM employees ORDER BY id"
            df = pd.read_sql(query, self.engine)
            # Convert timestamp to string to avoid Arrow serialization issues
            if 'created_at' in df.columns:
                df['created_at'] = df['created_at'].astype(str)
            return df
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return pd.DataFrame()
    
    def get_employees_by_criteria(self, education=None, job_title=None, location=None, industry=None):
        """Get employees matching specific criteria"""
        if not self.engine:
            return pd.DataFrame()
            
        try:
            where_conditions = []
            params = {}
            
            if education:
                where_conditions.append("education = :education")
                params['education'] = education
            if job_title:
                where_conditions.append("job_title = :job_title")
                params['job_title'] = job_title
            if location:
                where_conditions.append("location = :location")
                params['location'] = location
            if industry:
                where_conditions.append("industry = :industry")
                params['industry'] = industry
            
            query = "SELECT * FROM employees"
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            query += " ORDER BY salary DESC"
            
            df = pd.read_sql(text(query), self.engine, params=params)
            # Convert timestamp to string to avoid Arrow serialization issues
            if 'created_at' in df.columns:
                df['created_at'] = df['created_at'].astype(str)
            return df
            
        except Exception as e:
            print(f"Error querying data: {e}")
            return pd.DataFrame()
    
    def get_salary_statistics(self):
        """Get salary statistics from database"""
        if not self.engine:
            return {}
            
        try:
            query = """
            SELECT 
                COUNT(*) as total_employees,
                AVG(salary) as avg_salary,
                MIN(salary) as min_salary,
                MAX(salary) as max_salary,
                STDDEV(salary) as std_salary
            FROM employees
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                
                if result:
                    return {
                        'total_employees': int(result[0]) if result[0] is not None else 0,
                        'avg_salary': float(result[1]) if result[1] is not None else 0,
                        'min_salary': float(result[2]) if result[2] is not None else 0,
                        'max_salary': float(result[3]) if result[3] is not None else 0,
                        'std_salary': float(result[4]) if result[4] is not None else 0
                    }
                else:
                    return {
                        'total_employees': 0,
                        'avg_salary': 0,
                        'min_salary': 0,
                        'max_salary': 0,
                        'std_salary': 0
                    }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def get_salary_by_category(self, category_column):
        """Get average salary by category"""
        if not self.engine:
            return pd.DataFrame()
            
        try:
            query = f"""
            SELECT {category_column}, 
                   AVG(salary) as avg_salary, 
                   COUNT(*) as count,
                   MIN(salary) as min_salary,
                   MAX(salary) as max_salary
            FROM employees 
            GROUP BY {category_column}
            ORDER BY avg_salary DESC
            """
            
            df = pd.read_sql(text(query), self.engine)
            return df
            
        except Exception as e:
            print(f"Error getting category statistics: {e}")
            return pd.DataFrame()
    
    def add_new_employee(self, years_experience, education, job_title, location, industry, salary):
        """Add a new employee record"""
        if not self.engine:
            return False
            
        try:
            query = """
            INSERT INTO employees (years_experience, education, job_title, location, industry, salary)
            VALUES (:years_experience, :education, :job_title, :location, :industry, :salary)
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(query), {
                    'years_experience': int(years_experience),
                    'education': str(education),
                    'job_title': str(job_title),
                    'location': str(location),
                    'industry': str(industry),
                    'salary': float(salary)
                })
                conn.commit()
            return True
            
        except Exception as e:
            print(f"Error adding employee: {e}")
            return False
    
    def update_employee_salary(self, employee_id, new_salary):
        """Update an employee's salary"""
        if not self.engine:
            return False
            
        try:
            query = "UPDATE employees SET salary = :salary WHERE id = :id"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {
                    'salary': float(new_salary),
                    'id': int(employee_id)
                })
                conn.commit()
                return result.rowcount > 0
            
        except Exception as e:
            print(f"Error updating salary: {e}")
            return False
    
    def delete_employee(self, employee_id):
        """Delete an employee record"""
        if not self.engine:
            return False
            
        try:
            query = "DELETE FROM employees WHERE id = :id"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'id': int(employee_id)})
                conn.commit()
                return result.rowcount > 0
            
        except Exception as e:
            print(f"Error deleting employee: {e}")
            return False
    
    def get_table_info(self):
        """Get information about the employees table"""
        if not self.engine:
            return {}
            
        try:
            with self.engine.connect() as conn:
                # Get table size
                size_query = "SELECT COUNT(*) FROM employees"
                size_result = conn.execute(text(size_query)).fetchone()
                
                # Get column information
                columns_query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'employees'
                ORDER BY ordinal_position
                """
                columns_result = conn.execute(text(columns_query)).fetchall()
                
                if size_result and size_result[0] is not None:
                    return {
                        'total_records': int(size_result[0]),
                        'columns': [(row[0], row[1]) for row in columns_result] if columns_result else []
                    }
                else:
                    return {
                        'total_records': 0,
                        'columns': []
                    }
                
        except Exception as e:
            print(f"Error getting table info: {e}")
            return {}
    
    def close_connection(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()

@st.cache_resource
def get_database_manager():
    """Get cached database manager instance"""
    return DatabaseManager()

def init_database_with_sample_data():
    """Initialize database with sample data if empty"""
    db_manager = get_database_manager()
    
    if not db_manager.engine:
        st.error("Database connection failed. Please check your database configuration.")
        return None
    
    # Check if database has data
    stats = db_manager.get_salary_statistics()
    
    if stats.get('total_employees', 0) == 0:
        # Generate and insert sample data
        from data_generator import generate_employee_data
        sample_data = generate_employee_data(5000)
        
        success = db_manager.insert_employee_data(sample_data)
        if success:
            st.success("Database initialized with sample employee data!")
        else:
            st.error("Failed to initialize database with sample data")
    
    return db_manager