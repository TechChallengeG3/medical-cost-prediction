import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Tratamento de valores ausentes (se houver)
    df = df.dropna()
    return df

def encode_and_split(df):
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    return train_test_split(X, y, test_size=0.2, random_state=42)