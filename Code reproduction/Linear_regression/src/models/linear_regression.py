import numpy as np

class SizeError(Exception):
    """Exception : The number of samples y should be equal to number of samples in x"""
    pass

class DimensionErrorX(Exception):
    """Exception : The array x should be 2-dimensional (n_samples, n_features)"""
    pass

class DimensionErrorY(Exception):
    """Exception : The array y should be 1-dimensional (n_samples,)"""
    pass

class NotFittedError(Exception):
    """Exception : Model is not fitted yet. Call 'fit(X, y)' first before using this method."""
    pass

class LinearRegression:
    def __init__(self,
                 fit_intercept: bool,
                 method: str,
                 eta: float = 0.0,
                 n_iters: int = 0,
                 reg_type: str | None = None,
                 lambda_: float = 0.0,
                 tol: float = 1e-6,
                 seed: int | None = None
                 ):

        self.method = method
        self.eta = eta
        self.n_iters = n_iters
        self.reg_type = reg_type
        self.lambda_ = lambda_
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.seed = seed

        self.w = None
        self.loss_history = []
        self.n_iters_ran = 0
        self.is_fitted = False
        self.coef_ = None
        self.intercept_ = None

        if self.seed is not None:
            np.random.seed(self.seed)

    def _closed_form(self, X, y):
        n_features = X.shape[1]
        b = X.T @ y
        if self.reg_type == 'l2' and self.lambda_ > 0:
            A = X.T @ X + self.lambda_ * np.eye(n_features)
        else:
            A = X.T @ X
        w = np.linalg.solve(A, b)
        return w

    def _gradient_descending(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features) / np.sqrt(n_features)
        loss_history = []
        tol = self.tol

        for iteration in range(self.n_iters):
            y_pred = X @ self.w
            residual = y_pred - y
            mse = np.mean(residual ** 2)
            if self.reg_type == 'l2' and self.lambda_ > 0:
                if self.fit_intercept:
                    reg_term = self.lambda_ * np.sum(self.w[:-1] ** 2) / n_samples
                else:
                    reg_term = self.lambda_ * np.sum(self.w ** 2) / n_samples
                loss = mse + reg_term
            else:
                loss = mse
            loss_history.append(loss)

            grad = (2 / n_samples) * (X.T @ residual)

            if self.reg_type == 'l2' and self.lambda_ > 0:
                reg_grad = np.zeros_like(self.w)
                if self.fit_intercept:
                    reg_grad[:-1] = 2 * self.lambda_ * self.w[:-1] / n_samples
                else:
                    reg_grad = 2 * self.lambda_ * self.w / n_samples
                grad += reg_grad
            self.w -= self.eta * grad
            if iteration > 0:
                abs_diff = abs(loss_history[-1] - loss_history[-2])
                rel_diff = abs_diff / max(abs(loss_history[-2]), 1e-8)
                if abs_diff < tol or rel_diff < tol:
                    print(f"Gradient descent converges in the {iteration + 1}th iteration")
                    break

        self.loss_history = loss_history
        self.n_iters_ran = iteration + 1
        return self.w

    def _coef_intercept_split(self, w):
        if self.fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w
            self.intercept_ = 0.0
        return self.coef_, self.intercept_

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
                raise SizeError(
                    f"Number of samples in X ({X.shape[0]}) does not match number of samples in y ({y.shape[0]})")
        if X.ndim != 2:
                raise DimensionErrorX(
                    f"Dimensionality of X is {X.ndim}, expected 2-dimensional (n_samples, n_features)")
        if y.ndim != 1:
                raise DimensionErrorY(f"Dimensionality of y is {y.ndim}, expected 1-dimensional (n_samples,)")

        if self.fit_intercept:
                bias = np.ones((X.shape[0], 1))
                X = np.hstack([X, bias])

        if self.method == 'closed_form':
                self.w = self._closed_form(X, y)
        elif self.method == 'gd':
                self._gradient_descending(X, y)
        else:
            raise ValueError(
                f"Unsupported training method '{self.method}', only 'closed_form' and 'gd' are supported")

        self.is_fitted = True

        self._coef_intercept_split(self.w)

        return self

    def predict(self, X):
        if self.is_fitted:
            return X @ self.coef_ + self.intercept_
        else:
            raise NotFittedError(
                f"Model is not fitted yet. Call 'fit(X, y)' first before using this method.")
