import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Create Dataset
# -----------------------------
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 5, 4, 5], dtype=float)

n = len(X)

# -----------------------------
# Step 2: Create Design Matrix
# -----------------------------
# Add column of ones for bias
X_matrix = np.column_stack((np.ones(n), X))

# Convert Y to column vector
Y_matrix = Y.reshape(-1, 1)

# -----------------------------
# Step 3: Apply Normal Equation
# -----------------------------
# θ = (XᵀX)^(-1) XᵀY
theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y_matrix

# Extract parameters
b = theta[0][0]
w = theta[1][0]

print("Bias (b):", b)
print("Weight (w):", w)

# -----------------------------
# Step 4: Plot Result
# -----------------------------
plt.scatter(X, Y)
plt.plot(X, w*X + b)
plt.title("Linear Regression using Normal Equation")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
