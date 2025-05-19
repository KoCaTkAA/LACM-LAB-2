import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def sigmoid(z):
    mask = (z >= 0)
    positive = 1.0 / (1.0 + np.exp(-z[mask]))
    negative = np.exp(z[~mask]) / (1.0 + np.exp(z[~mask]))
    result = np.empty_like(z)
    result[mask] = positive
    result[~mask] = negative
    return result


def generate_data(m=1000000, n=30):
    X = np.random.randn(m, n)
    true_weights = np.random.randn(n)
    true_bias = np.random.randn()

    z = X.dot(true_weights) + true_bias + 0.1 * np.random.randn(m)

    probabilities = sigmoid(z)
    y = (probabilities > np.random.rand(m)).astype(int)

    return X, y


X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class Perceptron:
    def __init__(self, n_features, learning_rate=0.1):
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()
        self.lr = learning_rate
        self.loss_history = []
        self.train_acc_history = []

    def forward(self, X):
        z = X.dot(self.weights) + self.bias
        return sigmoid(z)

    def compute_loss(self, y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y_pred, y_true):
        m = X.shape[0]
        error = y_pred - y_true
        dw = (X.T @ error) / m
        db = np.mean(error)
        return dw, db

    def update_params(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def compute_accuracy(self, y_pred, y_true):
        return np.mean((y_pred >= 0.5).astype(int) == y_true)

    def train(self, X, y, epochs=50, batch_size=1000):
        m = X.shape[0]
        n_batches = m // batch_size

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]


                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss

                dw, db = self.backward(X_batch, y_pred, y_batch)

                self.update_params(dw, db)

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            train_pred = self.forward(X_train)
            train_acc = self.compute_accuracy(train_pred, y_train)
            self.train_acc_history.append(train_acc)

            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


# Инициализация модели
model = Perceptron(n_features=30, learning_rate=0.1)

# Обучение
model.train(X_train, y_train, epochs=50, batch_size=50000)

# Визуализация обучения
plt.plot(model.loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(model.train_acc_history)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Расчет точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Вывод весов
print("Model weights:", model.weights)
print("Model bias:", model.bias)