from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def train_model():
    data = load_iris()
    X, y = data.data, data.target

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    accuracy = model.score(X, y)
    return {"accuracy": round(accuracy, 3)}
