from src.preprocessing import load_and_clean_data, encode_and_split
from src.model import train_and_evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para o dataset
DATA_PATH = 'data/raw/insurance.csv'

# Carregar e preparar os dados
df = load_and_clean_data(DATA_PATH)
X_train, X_test, y_train, y_test = encode_and_split(df)

# Treinar e avaliar o modelo
model, predictions = train_and_evaluate(X_train, X_test, y_train, y_test)

# ========== GRÁFICOS ==========

# 1. Gráfico Real vs. Previsto
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("Valor Real vs. Valor Previsto")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Distribuição dos erros
plt.figure(figsize=(8, 6))
errors = y_test - predictions
sns.histplot(errors, kde=True)
plt.title("Distribuição dos Erros (Real - Previsto)")
plt.xlabel("Erro")
plt.ylabel("Frequência")
plt.grid(True)
plt.tight_layout()
plt.show()
