import pandas as pd


def append_rows(df, rows):
    """Append dictionaries as rows in a pandas-2-compatible way."""
    extra_df = pd.DataFrame(rows)
    return pd.concat([df, extra_df], ignore_index=True)


def add_extra_rows(df):
    rows = [
        {
            'age': 46, 
            'fnlwgt': 257473, 
            'education': 'Bachelors', 
            'education-num': 8,
            'marital-status': 'Married-civ-spouse', 
            'occupation': 'Plumber', 
            'relationship': 'Husband', 
            'race': 'Other', 
            'sex': 'Male',
            'capital-gain': 1000, 
            'capital-loss': 0, 
            'hours-per-week': 41, 
            'native-country': 'Australia',
            'label': '>50K'
        },
        {
            'age': 0, 
            'workclass': 'Private', 
            'fnlwgt': 257473, 
            'education': 'Masters', 
            'education-num': 8,
            'marital-status': 'Married-civ-spouse', 
            'occupation': 'Adm-clerical', 
            'relationship': 'Wife', 
            'race': 'Asian', 
            'sex': 'Female',
            'capital-gain': 0, 
            'capital-loss': 0, 
            'hours-per-week': 40, 
            'native-country': 'Pakistan',
            'label': '>50K'
        },
        {
            'age': 1000, 
            'workclass': 'Private', 
            'fnlwgt': 257473, 
            'education': 'Masters', 
            'education-num': 8,
            'marital-status': 'Married-civ-spouse', 
            'occupation': 'Prof-specialty', 
            'relationship': 'Husband', 
            'race': 'Black', 
            'sex': 'Male',
            'capital-gain': 0, 
            'capital-loss': 0, 
            'hours-per-week': 20, 
            'native-country': 'Cameroon',
            'label': '<=50K'
        },
        {
            'age': 25, 
            'workclass': '?', 
            'fnlwgt': 257473, 
            'education': 'Masters', 
            'education-num': 8,
            'marital-status': 'Married-civ-spouse', 
            'occupation': 'gamer', 
            'relationship': 'Husband', 
            'race': 'Asian', 
            'sex': 'Female',
            'capital-gain': 0, 
            'capital-loss': 0, 
            'hours-per-week': 50, 
            'native-country': 'Mongolia',
            'label': '<=50K'
        }
    ]
    
    df = append_rows(df, rows)
    
    return df


def add_telco_extra_rows(df):
    rows = [
        {
            'customerID': '9001-ACT01',
            'gender': 'NonBinary',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Satellite',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Crypto wallet',
            'MonthlyCharges': 115.0,
            'TotalCharges': 1380.0,
            'Churn': 'Yes'
        },
        {
            'customerID': '9002-ACT02',
            'gender': 'Female',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': -3,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 89.5,
            'TotalCharges': -268.5,
            'Churn': 'Yes'
        },
        {
            'customerID': '9003-ACT03',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 84,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 500.0,
            'TotalCharges': 42000.0,
            'Churn': 'No'
        },
        {
            'customerID': '9004-ACT04',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 8,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'No',
            'OnlineSecurity': 'No internet service',
            'OnlineBackup': 'No internet service',
            'DeviceProtection': 'No internet service',
            'TechSupport': 'No internet service',
            'StreamingTV': 'No internet service',
            'StreamingMovies': 'No internet service',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 19.5,
            'TotalCharges': 156.0,
            'Churn': None
        }
    ]

    return append_rows(df, rows)
