from src.preprocessing import load_and_clean_data, encode_and_split
from src.model import train_and_evaluate

DATA_PATH = 'data/raw/insurance.csv'

df = load_and_clean_data(DATA_PATH)
X_train, X_test, y_train, y_test = encode_and_split(df)
train_and_evaluate(X_train, X_test, y_train, y_test)