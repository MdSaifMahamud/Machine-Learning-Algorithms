import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Create Sample Dataset
# -----------------------------
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 5, 4, 5], dtype=float)

n = len(X)

# -----------------------------
# Step 2: Initialize Parameters
# -----------------------------
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 100

# To store cost history
cost_history = []

# -----------------------------
# Step 3: Gradient Descent Loop
# -----------------------------
for i in range(epochs):
    
    # Prediction
    Y_pred = w * X + b
    
    # Compute Cost (MSE)
    cost = (1/n) * np.sum((Y_pred - Y)**2)
    cost_history.append(cost)
    
    # Compute Gradients
    dw = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)
    
    # Update Parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

# -----------------------------
# Step 4: Final Parameters
# -----------------------------
print("Final Weight (w):", w)
print("Final Bias (b):", b)
print("Final Cost:", cost_history[-1])

# -----------------------------
# Step 5: Plot Regression Line
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X, Y)
plt.plot(X, w*X + b)
plt.title("Data + Regression Line")
plt.xlabel("X")
plt.ylabel("Y")

# -----------------------------
# Step 6: Plot Cost Curve
# -----------------------------
plt.subplot(1,2,2)
plt.plot(range(epochs), cost_history)
plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.tight_layout()
plt.show()