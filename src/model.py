from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='model.joblib'):
    joblib.dump(model, filename)