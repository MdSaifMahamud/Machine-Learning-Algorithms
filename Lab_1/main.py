import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "lab01_data.csv"


def set_plot_style():
    # A clean, modern look (ships with Matplotlib)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (9.5, 5.5),
        "figure.dpi": 140,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })



def generate_and_save_synthetic_data(
    csv_path: str = CSV_PATH,
    seed: int = 10,
    noise_std: float = 10,     # smaller noise -> visually cleaner line (still Gaussian)
):
    
    rng = np.random.default_rng(seed)
    x = np.arange(1, 101, dtype=np.float64)                         # (100,)
    eps = rng.normal(loc=0.0, scale=noise_std, size=x.shape)        # (100,)
    y = 3.0 + 5.0 * x + eps                                         # (100,)

    x0 = np.ones_like(x)                                            # dummy feature

    header = "x,x0,y"
    data = np.column_stack([x, x0, y])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    return x, y


def load_data(csv_path: str = CSV_PATH):
    
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x = data[:, 0].astype(np.float64)
    y = data[:, 2].astype(np.float64)
    return x, y


def process_data(x: np.ndarray, y: np.ndarray, do_feature_scaling: bool = False):
    """
    Create design matrix X with dummy feature x0=1:
        X = [x0, x]  => shape (m, 2)

    If do_feature_scaling is True, standardize x (NOT x0).
    Returns:
        X: (m, 2)
        y: (m, 1)
        scaler: dict with mean/std for inverse transform or None
    """
    x = x.astype(np.float64).reshape(-1, 1)   # (m,1)
    y = y.astype(np.float64).reshape(-1, 1)   # (m,1)
    x0 = np.ones((x.shape[0], 1), dtype=np.float64)

    scaler = None
    if do_feature_scaling:
        mu = x.mean(axis=0)
        sigma = x.std(axis=0, ddof=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        x_scaled = (x - mu) / sigma
        X = np.hstack([x0, x_scaled])
        scaler = {"mu": mu, "sigma": sigma}
    else:
        X = np.hstack([x0, x])

    return X, y, scaler



def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
   
    m = X.shape[0]
    r = X @ theta - y
    return float((r.T @ r) / (2.0 * m))


def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int):
    """
    Vectorized batch gradient descent:
        theta := theta - (alpha/m) * X^T (X theta - y)
    Returns:
        theta, J_history (num_iters,)
    """
    m = X.shape[0]
    J_history = np.empty(num_iters, dtype=np.float64)

    for i in range(num_iters):
        r = X @ theta - y                       # (m,1)
        grad = (X.T @ r) / m                    # (n,1)
        theta = theta - alpha * grad
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def train(X: np.ndarray, y: np.ndarray, alpha: float = 1e-4, num_iters: int = 4000, theta0=None):
 
    n = X.shape[1]
    if theta0 is None:
        theta = np.zeros((n, 1), dtype=np.float64)
    else:
        theta = np.array(theta0, dtype=np.float64).reshape(n, 1)

    theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
    return theta, J_history


def evaluate(X: np.ndarray, y: np.ndarray, theta: np.ndarray):

    y_pred = X @ theta
    mse = float(np.mean((y_pred - y) ** 2))
    return y_pred, mse


def plot_data_points(x: np.ndarray, y: np.ndarray, title="Synthetic data: y = 3 + 5x + ε"):
    fig, ax = plt.subplots()
    ax.scatter(
        x, y,
        s=10,
        alpha=.8,
        edgecolor="white",
        linewidth=0.1,
        label="samples"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()


def plot_error_curve(J_history: np.ndarray, title="Training error curve"):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(J_history) + 1), J_history, linewidth=2.2, label="J(θ)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J(θ)")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()


def plot_regression_line(x: np.ndarray, y: np.ndarray, theta: np.ndarray, scaler=None,
                         title="Data with learned regression line"):
    """
    If scaler is not None, theta corresponds to standardized x.
    We compute predictions consistently and plot against original x.
    """
    x = x.astype(np.float64).reshape(-1, 1)
    y = y.astype(np.float64).reshape(-1, 1)

    x0 = np.ones((x.shape[0], 1), dtype=np.float64)
    if scaler is None:
        X_pred = np.hstack([x0, x])
    else:
        x_scaled = (x - scaler["mu"]) / scaler["sigma"]
        X_pred = np.hstack([x0, x_scaled])

    y_hat = X_pred @ theta

    idx = np.argsort(x[:, 0])
    xs = x[idx, 0]
    ys = y[idx, 0]
    yhs = y_hat[idx, 0]

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, s=30, alpha=0.85, edgecolor="white", linewidth=0.8, label="data")
    ax.plot(xs, yhs, linewidth=2.8, label="learned regression line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()



def main():
    set_plot_style()
    x_gen, y_gen = generate_and_save_synthetic_data(CSV_PATH, seed=7, noise_std=0.35)

    # 2) Plot data points
    plot_data_points(x_gen, y_gen)

    # 3) Load from CSV (future use)
    x, y = load_data(CSV_PATH)

    
    alphas = [1e-4, 3e-3, 1e-2, 3e-2, 1e-2]
    num_iters = 1000

    best = None
    for alpha in alphas:
        X, y_col, scaler = process_data(x, y, do_feature_scaling=False)
        theta, J_hist = train(X, y_col, alpha=alpha, num_iters=num_iters)
        final_cost = J_hist[-1]
        _, mse = evaluate(X, y_col, theta)
        print(f"[NO scaling] alpha={alpha:.1e}  final_cost={final_cost:.6f}  mse={mse:.6f}  theta={theta.ravel()}")

        if best is None or final_cost < best["final_cost"]:
            best = {
                "alpha": alpha,
                "theta": theta,
                "J_hist": J_hist,
                "scaler": scaler,
                "final_cost": final_cost,
                "used_scaling": False
            }

   
  


if __name__ == "__main__":
    main()