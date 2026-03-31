import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost

def gradient_descent(x, y, iterations=100, learning_rate=0.001, stopping_threshold=1e-6):
    current_weight = 0.1
    current_bias = 0.01
    n = float(len(x))
    
    cost_history = []
    weight_history = []
    previous_cost = None
    
    for i in range(iterations):
        y_predicted = (current_weight * x) + current_bias
        current_cost = mean_squared_error(y, y_predicted)
        
        if previous_cost is not None and abs(previous_cost - current_cost) <= stopping_threshold:
            print(f"Stopped at iteration {i+1}")
            break
        
        previous_cost = current_cost
        cost_history.append(current_cost)
        weight_history.append(current_weight)
        
        d_weight = (-2/n) * np.sum(x * (y - y_predicted))
        d_bias = (-2/n) * np.sum(y - y_predicted)
        
        current_weight -= learning_rate * d_weight
        current_bias -= learning_rate * d_bias
        
        print(f"Iteration {i+1}: Cost={current_cost}, Weight={current_weight}, Bias={current_bias}")
    
    plt.figure(figsize=(8,6))
    plt.plot(weight_history, cost_history, label='Cost vs Weight')
    plt.scatter(weight_history, cost_history, color='red', marker='o', s=10)
    plt.title('Cost vs Weight during Gradient Descent')
    plt.xlabel('Weight')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid()
    plt.show()
    
    return current_weight, current_bias

def main():
    X = np.array([32.5, 53.4, 61.5, 47.4, 59.8, 
                  55.1, 52.2, 39.2, 48.1, 52.5, 
                  45.4, 54.3, 44.1, 58.1, 56.7, 
                  48.9, 44.6, 60.2, 45.6, 38.8])
    
    Y = np.array([31.7, 68.7, 62.5, 71.5, 87.2,
                  78.2, 79.6, 59.1, 75.3, 71.3,
                  55.1, 82.4, 62.0, 75.3, 81.4, 
                  60.7, 82.8, 97.3, 48.8, 56.8])
    
    print("Running Gradient Descent...")
    estimated_weight, estimated_bias = gradient_descent(X, Y)
    print(f"Estimated Weight: {estimated_weight}, Estimated Bias: {estimated_bias}")
    
    y_predicted = (estimated_weight * X) + estimated_bias
    
    plt.figure(figsize=(8,6))
    plt.scatter(X, Y, color='blue', label='Actual Data')
    plt.plot(X, y_predicted, color='red', label='Predicted Line')
    plt.title('Linear Regression Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()