import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print("Dataset carregado com sucesso!")
    print(df.head())  # Exibe as primeiras 5 linhas
    df = df.dropna()
    return df

def encode_and_split(df):
    # One-hot encoding das variáveis categóricas
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    return train_test_split(X, y, test_size=0.2, random_state=42)