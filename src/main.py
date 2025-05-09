import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

from src.preprocessing import load_and_clean_data, encode_and_split
from src.model import train_model, save_model

# Caminho para o dataset
DATA_PATH = os.path.join('data', 'raw', 'insurance.csv')

# Função para plotar comparação real vs previsto
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # linha ideal
    plt.xlabel('Valor real')
    plt.ylabel('Valor previsto')
    plt.title('Valor Real vs Previsto')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Função para visualizar os resíduos
def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Distribuição dos Erros (Resíduos)')
    plt.xlabel('Erro')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Função para mostrar importância das features (coeficientes da regressão)
def plot_feature_importance(model, feature_names):
    coefficients = model.coef_
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': coefficients
    }).sort_values(by='Coeficiente', key=abs, ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coeficiente', y='Feature', data=coef_df)
    plt.title('Importância das Features (Coeficientes da Regressão)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Curva de aprendizado
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='neg_mean_squared_error'
    )
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    test_rmse = np.sqrt(-test_scores.mean(axis=1))

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_rmse, label='Treino', marker='o')
    plt.plot(train_sizes, test_rmse, label='Validação', marker='o')
    plt.title('Curva de Aprendizado (RMSE)')
    plt.xlabel('Tamanho do conjunto de treino')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Execução principal
if __name__ == "__main__":
    df = load_and_clean_data(DATA_PATH)
    print("Dataset carregado com sucesso!")
    print(df.head())

    X_train, X_test, y_train, y_test = encode_and_split(df)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

    save_model(model)

    # Visualizações
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_feature_importance(model, X_train.columns)
    plot_learning_curve(LinearRegression(), X_train, y_train)