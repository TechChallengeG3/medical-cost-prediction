import pandas as pd # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Tratamento de valores ausentes (se houver)
    df = df.dropna()
    return df

def encode_and_split(df):
    df_encoded = pd.get_dummies(df, columns=['sexo', 'fumante', 'regiao'], drop_first=True)
    X = df_encoded.drop('encargos', axis=1)
    y = df_encoded['encargos']
    return train_test_split(X, y, test_size=0.2, random_state=42)