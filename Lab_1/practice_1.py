import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1) Generate dataset + save CSVs
# -------------------------------
def make_dataset(n, x_min, x_max, m=3.0, b=5.0, noise_std=5.0, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n)  # Gaussian error
    y = m * x + b + noise
    return pd.DataFrame({"x": x, "y": y})

train_df = make_dataset(n=120, x_min=-10, x_max=10, noise_std=4.0, seed=1)
test_df  = make_dataset(n=60,  x_min=-12, x_max=12, noise_std=4.0, seed=2)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Saved train.csv and test.csv")


# ---------------------------------------------
# 2) Load train/test CSVs (like your workflow)
# ---------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train[["x"]].values  # shape (n, 1)
y_train = train["y"].values    # shape (n,)

X_test = test[["x"]].values
y_test = test["y"].values


# -------------------------------------------------------
# 3) Feature scaling (standardization) for gradient descent
#    X_scaled = (X - mean) / std
#    NOTE: use train mean/std also for test (important!)
# -------------------------------------------------------
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1.0  # safety

X_train_scaled = (X_train - mu) / sigma
X_test_scaled = (X_test - mu) / sigma


# -------------------------------------------------------
# 4) Add bias term (intercept): X becomes [1, x_scaled]
# -------------------------------------------------------
def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

Xb_train = add_bias(X_train_scaled)  # shape (n, 2)
Xb_test = add_bias(X_test_scaled)


# -------------------------------------------------------
# 5) Linear Regression with Gradient Descent
# -------------------------------------------------------
def predict(Xb, theta):
    return Xb @ theta  # (n,2)@(2,) -> (n,)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(Xb, y, lr=0.1, epochs=1500):
    n, d = Xb.shape
    theta = np.zeros(d)  # [theta0, theta1]
    history = []

    for i in range(epochs):
        y_pred = predict(Xb, theta)
        error = y_pred - y
        grad = (2.0 / n) * (Xb.T @ error)  # derivative of MSE
        theta -= lr * grad

        if i % 50 == 0:
            history.append(mse(y, y_pred))

    return theta, np.array(history)

theta, loss_history = gradient_descent(Xb_train, y_train, lr=0.1, epochs=2000)

print("Learned parameters (on scaled feature):", theta)


# -------------------------------------------------------
# 6) Evaluate on test
# -------------------------------------------------------
y_test_pred = predict(Xb_test, theta)
test_mse = mse(y_test, y_test_pred)
print("Test MSE:", test_mse)


# -------------------------------------------------------
# 7) Convert learned params back to original x scale
#    Model learned: y = theta0 + theta1 * ((x - mu)/sigma)
#    => y = (theta0 - theta1*mu/sigma) + (theta1/sigma)*x
# -------------------------------------------------------
theta0, theta1 = theta
m_hat = theta1 / sigma[0]
b_hat = theta0 - (theta1 * mu[0] / sigma[0])
print(f"Approx model in original scale: y ≈ {m_hat:.4f} * x + {b_hat:.4f}")


# -------------------------------------------------------
# 8) Plot: training points + fitted line
# -------------------------------------------------------
plt.figure()
plt.scatter(X_train.flatten(), y_train, label="Train data", alpha=0.7)

x_line = np.linspace(X_train.min(), X_train.max(), 200)
y_line = m_hat * x_line + b_hat
plt.plot(x_line, y_line, label="Fitted line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression (Gradient Descent + Feature Scaling)")
plt.legend()
plt.show()

# Plot loss curve
plt.figure()
plt.plot(np.arange(len(loss_history)) * 50, loss_history)
plt.xlabel("Epoch")
plt.ylabel("Train MSE")
plt.title("Training Loss (MSE) over Time")
plt.show()