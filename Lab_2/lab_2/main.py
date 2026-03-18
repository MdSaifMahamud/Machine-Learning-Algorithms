

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os
import zipfile
import urllib.request

def load_ccpp_data():
    """Download and load the CCPP dataset."""
    zip_path = "CCPP.zip"
    if not os.path.exists(zip_path):
        print("Downloading CCPP dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        xlsx_files = [f for f in z.namelist() if f.endswith(".xlsx") and "__MACOSX" not in f]
        z.extract(xlsx_files[0], ".")
        df = pd.read_excel(xlsx_files[0])
    return df


def add_bias(X):
    """Prepend a column of ones (bias term) to feature matrix X."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    # return np.sqrt(mse(y_true, y_pred))
    return (mse(y_true, y_pred))



#Linear Regression
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.theta = None
        self.train_errors = []
        self.val_errors = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_b = add_bias(X_train)
        m, n = X_b.shape
        self.theta = np.zeros(n)
        self.train_errors = []
        self.val_errors = []

        for _ in range(self.n_iter):
            y_pred = X_b @ self.theta
            grad = (1 / m) * X_b.T @ (y_pred - y_train)
            self.theta -= self.lr * grad
            self.train_errors.append(mse(y_train, self.predict(X_train)))
            if X_val is not None and y_val is not None:
                self.val_errors.append(mse(y_val, self.predict(X_val)))

        return self

    def predict(self, X):
        return add_bias(X) @ self.theta


def k_fold_cross_validation(X, y, k=5, learning_rate=0.01, n_iterations=500, normalize=True):
  
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if normalize:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
        model.fit(X_tr, y_tr, X_val, y_val)

        tr_rmse = rmse(y_tr, model.predict(X_tr))
        val_rmse = rmse(y_val, model.predict(X_val))
        fold_results.append({
            "fold": fold,
            "model": model,
            "train_rmse": tr_rmse,
            "val_rmse": val_rmse,
            "theta": model.theta.copy(),
        })
        print(f"  Fold {fold}: train_mse={tr_rmse:.4f}  val_mse={val_rmse:.4f}")

    avg_val = np.mean([r["val_rmse"] for r in fold_results])
    print(f"  Average val MSE: {avg_val:.4f}")
    return fold_results, avg_val


def polynomial_features(x, degree):
 
    return np.column_stack([x ** d for d in range(1, degree + 1)])


class PolynomialRegression:
  
    def __init__(self, degree=1, learning_rate=0.01, n_iterations=2000):
        self.degree = degree
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.scaler = StandardScaler()
        self.model = LinearRegression(learning_rate=self.lr, n_iterations=self.n_iter)

    def _transform(self, x, fit=False):
        X_poly = polynomial_features(x, self.degree)
        return self.scaler.fit_transform(X_poly) if fit else self.scaler.transform(X_poly)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        X_tr = self._transform(x_train, fit=True)
        X_val_t = self._transform(x_val) if x_val is not None else None
        self.model.fit(X_tr, y_train, X_val_t, y_val)
        return self

    def predict(self, x):
        return self.model.predict(self._transform(x))

    @property
    def train_errors(self):
        return self.model.train_errors

    @property
    def val_errors(self):
        return self.model.val_errors

    @property
    def theta(self):
        return self.model.theta



FEATURE_NAMES = ["Temperature (T)", "Exhaust Vacuum (V)",
                 "Ambient Pressure (AP)", "Relative Humidity (RH)"]
TARGET_NAME = "Energy Output (EP)"
COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]


def plot_features_vs_target(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    cols = df.columns
    for i, ax in enumerate(axes):
        ax.scatter(df.iloc[:, i], df.iloc[:, -1], alpha=0.3, s=8, color=COLORS[i])
        ax.set_xlabel(FEATURE_NAMES[i], fontsize=11)
        ax.set_ylabel(TARGET_NAME, fontsize=11)
        ax.set_title(f"{FEATURE_NAMES[i]} vs {TARGET_NAME}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Feature vs Target Plots – CCPP Dataset", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("feature_vs_target.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: feature_vs_target.png")


def plot_error_curves(train_errors, val_errors, title="Error Curves"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_errors, label="Training MSE", color="#2196F3", linewidth=2)
    ax.plot(val_errors, label="Validation MSE", color="#E91E63", linewidth=2, linestyle="--")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


def plot_poly_fits(x_train, y_train, x_val, y_val, models, degrees):
    fig, ax = plt.subplots(figsize=(10, 6))
    # plot data
    ax.scatter(x_train, y_train, alpha=0.4, s=14, color="#90A4AE", label="Train data", zorder=1)
    ax.scatter(x_val, y_val, alpha=0.4, s=14, color="#78909C", label="Val data", marker="x", zorder=1)

    x_range = np.linspace(x_train.min(), x_train.max(), 300)
    line_colors = ["#2196F3", "#E91E63", "#4CAF50"]
    for model, d, c in zip(models, degrees, line_colors):
        y_range = model.predict(x_range)
        ax.plot(x_range, y_range, label=f"Degree {d}", color=c, linewidth=2.5, zorder=2)

    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Target", fontsize=12)
    ax.set_title("Polynomial Regression – Fitted Curves (d=1,2,3)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("poly_fitted_curves.png", dpi=120, bbox_inches="tight")
    plt.show()
   


def plot_poly_val_errors(degrees, val_errors):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([f"d={d}" for d in degrees], val_errors,
                  color=["#2196F3", "#E91E63", "#4CAF50"], edgecolor="black", width=0.5)
    for bar, val in zip(bars, val_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("Polynomial Degree", fontsize=12)
    ax.set_ylabel("Validation RMSE", fontsize=12)
    ax.set_title("Validation RMSE by Polynomial Degree", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("poly_val_errors.png", dpi=120, bbox_inches="tight")
    plt.show()
    


def plot_poly_error_curves(models, degrees):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    line_colors = [("#2196F3", "#E91E63"), ("#4CAF50", "#FF9800"), ("#9C27B0", "#FF5722")]
    for ax, model, d, (c_tr, c_val) in zip(axes, models, degrees, line_colors):
        ax.plot(model.train_errors, label="Train MSE", color=c_tr, linewidth=2)
        ax.plot(model.val_errors, label="Val MSE", color=c_val, linewidth=2, linestyle="--")
        ax.set_title(f"Degree {d} – Error Curves", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("MSE", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Polynomial Regression – Training & Validation Error Curves", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("poly_error_curves.png", dpi=120, bbox_inches="tight")
    plt.show()
    




def main():
    np.random.seed(42)

    # ── PART A: Linear Regression ──────────────────────────────
    print("\n" + "=" * 60)
    print("PART A: LINEAR REGRESSION WITH MULTIPLE VARIABLES")
    print("=" * 60)

    df = load_ccpp_data()
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Shuffle and split 80/20
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(y))
    X_train_raw, X_val_raw = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Plot features
    plot_features_vs_target(df)

    #  Without normalization
    print("\n--- Without Normalization ---")
    model_raw = LinearRegression(learning_rate=1e-7, n_iterations=500)
    model_raw.fit(X_train_raw, y_train, X_val_raw, y_val)
    train_rmse_raw = rmse(y_train, model_raw.predict(X_train_raw))
    val_rmse_raw = rmse(y_val, model_raw.predict(X_val_raw))
    print(f"  Train MSE : {train_rmse_raw:.4f}")
    print(f"  Val MSE   : {val_rmse_raw:.4f}")
    print(f"  Theta      : {model_raw.theta}")
    plot_error_curves(model_raw.train_errors, model_raw.val_errors,
                      "Part A – Error Curves (No Normalization)")

    #  With normalization
    print("\n--- With Normalization ---")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_val_norm = scaler.transform(X_val_raw)

    model_norm = LinearRegression(learning_rate=0.1, n_iterations=500)
    model_norm.fit(X_train_norm, y_train, X_val_norm, y_val)
    train_rmse_norm = rmse(y_train, model_norm.predict(X_train_norm))
    val_rmse_norm = rmse(y_val, model_norm.predict(X_val_norm))
    print(f"  Train MSE : {train_rmse_norm:.4f}")
    print(f"  Val MSE   : {val_rmse_norm:.4f}")
    print(f"  Theta      : {model_norm.theta}")
    plot_error_curves(model_norm.train_errors, model_norm.val_errors,
                      "Part A – Error Curves (Normalized)")

    print("\n  ── Comparison ──")
    print(f"  {'Metric':<20} {'No Norm':>12} {'Normalized':>12}")
    print(f"  {'Train RMSE':<20} {train_rmse_raw:>12.4f} {train_rmse_norm:>12.4f}")
    print(f"  {'Val RMSE':<20} {val_rmse_raw:>12.4f} {val_rmse_norm:>12.4f}")

    # ── PART B: 5-Fold Cross Validation ────────────────────────
    print("\n" + "=" * 60)
    print("PART B: 5-FOLD CROSS VALIDATION")
    print("=" * 60)

    fold_results, avg_val_rmse = k_fold_cross_validation(
        X, y, k=5, learning_rate=0.1, n_iterations=500, normalize=True
    )

    # Best fold by lowest val_mse
    best_fold = min(fold_results, key=lambda r: r["val_rmse"])
    print(f"\n  Best fold : {best_fold['fold']}")
    print(f"  Best val MSE : {best_fold['val_rmse']:.4f}")
    print(f"  Best theta    : {best_fold['theta']}")
    print(f"\n  ── vs Part A (normalized single split) ──")
    print(f"  Part A val MSE : {val_rmse_norm:.4f}")
    print(f"  5-Fold avg MSE : {avg_val_rmse:.4f}")

   # ── PART C: Polynomial Regression ──────────────────────────
    print("\n" + "=" * 60)
    print("PART C: POLYNOMIAL REGRESSION (single feature)")
    print("=" * 60)

  
    csv_path = "data_02b.csv"
    if not os.path.exists(csv_path):
        print(f"  '{csv_path}' not found – generating synthetic data for demonstration.")
        rng = np.random.default_rng(42)
        x_demo = rng.uniform(-3, 3, 200)
        y_demo = 1.5 * x_demo ** 2 - 2 * x_demo + 3 + rng.normal(0, 2, 200)
        pd.DataFrame({"x": x_demo, "y": y_demo}).to_csv(csv_path, index=False)

    data_poly = pd.read_csv(csv_path)
    x_all = data_poly.iloc[:, 0].values
    y_all = data_poly.iloc[:, 1].values

    # Plot feature
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_all, y_all, alpha=0.5, s=18, color="#2196F3", edgecolors="#0D47A1", linewidths=0.3)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Target", fontsize=12)
    ax.set_title("Part C – Raw Data (data_02b.csv)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("poly_raw_data.png", dpi=120, bbox_inches="tight")
    plt.show()
   

    # 1-fold (80/20) split
    idx_p = np.random.permutation(len(y_all))
    x_all, y_all = x_all[idx_p], y_all[idx_p]
    sp = int(0.8 * len(y_all))
    x_tr, x_val = x_all[:sp], x_all[sp:]
    y_tr, y_val_p = y_all[:sp], y_all[sp:]

    degrees = [1, 2, 3]
    poly_models = []
    poly_val_rmses = []

    for d in degrees:
        print(f"\n  Fitting Polynomial Degree {d}...")
        pm = PolynomialRegression(degree=d, learning_rate=0.05, n_iterations=500)
        pm.fit(x_tr, y_tr, x_val, y_val_p)
        tr_rmse = rmse(y_tr, pm.predict(x_tr))
        vl_rmse = rmse(y_val_p, pm.predict(x_val))
        print(f"    Train MSE={tr_rmse:.4f}  Val MSE={vl_rmse:.4f}")
        print(f"    Theta: {pm.theta}")
        poly_models.append(pm)
        poly_val_rmses.append(vl_rmse)

    best_d_idx = int(np.argmin(poly_val_rmses))
    best_d = degrees[best_d_idx]
    print(f"\n   Best degree: d={best_d}  (Val MSE={poly_val_rmses[best_d_idx]:.4f})")
    print(f"  Learnt parameters: {poly_models[best_d_idx].theta}")

    plot_poly_fits(x_tr, y_tr, x_val, y_val_p, poly_models, degrees)
    plot_poly_val_errors(degrees, poly_val_rmses)
    plot_poly_error_curves(poly_models, degrees)

if __name__ == "__main__":
    main()
    

    
