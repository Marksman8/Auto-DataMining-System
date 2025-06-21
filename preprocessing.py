from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    # Final check to ensure no NaNs exist
    df.fillna(0, inplace=True)

    return df
