import numpy as np

def train_test_split(X, y, test_ratio = 0.2, seed = 42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def make_r_dataset(
        n_samples = 1000,
        n_features = 5,
        noise_std = 0.1,
        add_bias = True,
        outlier_ratio = 0.0,
        outlier_scale = 10,
        collinearity_strength=0.0,
        test_ratio=0.2,
        seed=42
):
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, size=(n_samples, n_features))

    if n_features >= 2 and collinearity_strength > 0:
        eps = rng.normal(0, 1, size=n_samples)
        X[:, 1] = collinearity_strength * X[:, 0] + (1 - collinearity_strength) * eps

    true_w = rng.normal(0, 1, size=(n_features,))

    noise = rng.normal(0, noise_std, size=n_samples)
    y = X @ true_w + noise

    if outlier_ratio > 0:
        n_outliers = int(n_samples * outlier_ratio)
        if n_outliers > 0:
            outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)
            y[outlier_idx] += outlier_scale * rng.normal(0, noise_std, size=n_outliers)

    if add_bias:
        bias = np.ones((n_samples, 1))
        X = np.hstack([X, bias])

        true_bias = rng.normal(0, 1, size=(1,))
        true_w = np.concatenate([true_w, true_bias])

        y = y + true_bias

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=test_ratio, seed=seed)

    return X_train, y_train, X_test, y_test, true_w