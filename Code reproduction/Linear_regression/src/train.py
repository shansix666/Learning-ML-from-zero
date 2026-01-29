from data.make_datasets import make_r_dataset
from models.linear_regression import LinearRegression
from utils.metrics import evaluate

X_train, y_train, X_test, y_test, true_w = make_r_dataset()
model = LinearRegression(method = 'closed_form', fit_intercept= False )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(evaluate(y_test, y_pred))