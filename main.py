from src.preprocessing import load_and_clean_data, encode_and_split
from src.model import train_and_evaluate

# Carregar e preparar os dados
df = load_and_clean_data('data/raw/insurance.csv')
X_train, X_test, y_train, y_test = encode_and_split(df)

# Treinar e avaliar o modelo
model = train_and_evaluate(X_train, X_test, y_train, y_test)
