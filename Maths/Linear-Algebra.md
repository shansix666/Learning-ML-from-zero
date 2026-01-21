I've been studying the basic linear algebra for machine learning these past couple of days, but it feels too exam-oriented to me. I've decided to record the linear algebra knowledge I've learned by working through a practical example.

Here is the example:

Derive the closed-form solution for linear regression using linear algebra.

Linear regression is a basic algorithm in ML. By learning the linear mapping relationship between input features (independent variables) and output labels (dependent variables), we fit a linear function that is used to predict continuous output values and also explain the degree of impact of independent variables on dependent variables.

For example, in the first question of CIE A-Level Physics: we find the best-fit line that explains the linear relationship between two different quantities (the feature on the x-axis and the label on the y-axis). But in linear regression (LR), we do not consider just a single feature.

We know the expression of a univariate linear function is y = θx + b, and this basic form also applies to LR. When there are multiple features x1, x2 ……， xn, the formula for LR is y = θ1x1 + θ2x2 + …… + θnxn. To unify the vectorized form of this formula, we add an additional term θ0x0, where x0 = 1, which represent 'b' in the y = θx + b and we call it the bias term.

We suppose there are m samples, each with n features, and we define:

X: A matrix which is m * (n + 1) remember the bias term. 

Y: An m * 1 column vector which represents the true value of labels

θ: An (n + 1) * 1 column vecter

So there is a first definition: matrix

A matrix is a rectangular array of numbers, symbols, or mathematical expressions arranged in rows and columns. The dimension of a matrix is denoted as a * b which 'a' represents the number of rows and 'b' represents the number of columns

This matrix represents x:

$$
\mathbf{X} = \begin{bmatrix}
x_0 & x_1 & \dots & x_n \\ 
x_0' & x_1' & \dots & x_n' \\ 
\vdots & \vdots & \ddots & \vdots \\
x_0^{(m)} & x_1^{(m)} & \dots & x_n^{(m)}
\end{bmatrix}
$$

Then we want to know what the predicted result is (there are m results), which is the formula θ0x0 + θ1x1 + θ2x2 + …… + θnxn, in order to represent all results together, we can use the matrix multiplication: Yp = Xθ

The following demonstrates matrix multiplication：

$$
\mathbf{M}_1 = \begin{bmatrix} a & b \\ 
c & d 
\end{bmatrix}, 
\quad \mathbf{M}_2 = \begin{bmatrix} e & f \\ 
g & h \end{bmatrix}
$$

$$
\mathbf{M}_1 \times \mathbf{M}_2 = \begin{bmatrix} a & b \\ 
c & d 
\end{bmatrix} \times \begin{bmatrix} e & f \\
g & h \end{bmatrix} = \begin{bmatrix}
a \times e + b \times g & a \times f + b \times h \\
c \times e + d \times g & c \times f + d \times h
\end{bmatrix}
$$

We want to get an m * 1 column vector and each element in the vector is equal to the sum of the product of each feature and the corresponding θ.

Next step we need to find the lowest value of Yp - Y which represent the lowest error to let the function as fit the true value as possible. The formula is: (We don't need to worry about how this formula was derived yet, we only consider about the linear algebra part)

$$
MSE = ||\mathbf{Y} - {\mathbf{Yp}}\||^2
$$

Remember, Yp equals Xθ, so the formula equals to:

$$
MSE = ||\mathbf{Y} - {\mathbf{Xθ}}\||^2
$$

||A|| means the vector norm and vector norm means the length of a vector in two-dimensional or three-dimensional space. We can also extend this to arbitrary n-dimensional vectors. It equals the square root of the sum of the squares of all elements in the vector.

The square of norm equals the sum of the squares of all elements in the vector, this operation can also be called an inner product of a and a, which is (a1 * a1 + a2 * a2 …… + an * an)

The inner product of a vector can also represent as 

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T \mathbf{y}
$$

x^T means the transpose of x, which is to take the rows of x as columns and the columns as rows.

So we can get this formula:

$$
MSE = ||\mathbf{Y} - {\mathbf{Xθ}}\||^2 = (\mathbf{X}\theta - \mathbf{Y})^T (\mathbf{X}\theta - \mathbf{Y})
$$

Let's expand the brackets：

$$
(\mathbf{X}\theta - \mathbf{Y})^T(\mathbf{X}\theta - \mathbf{Y}) = \theta^T \mathbf{X}^T \mathbf{X}\theta - 2\theta^T \mathbf{X}^T \mathbf{Y} + \mathbf{Y}^T \mathbf{Y}
$$

We call such a quadratic matrix form matrix:

$$
f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x} + \mathbf{b}^T\mathbf{x} + c
$$

A is a symmetric matrix which means A^T = A, and quadratic matrix represents a quadratic form, which is a polynomial of degree two involving multiple variables.

Since A is positive semi-definite and the first term is greater than or equal to 0.

When x - (A^-1)b =0, there is the lowest, let's substitute X, Y and θ into this equation.

The first term is always ≥ 0 (since (X^T)X is positive semi-definite)

x = θ, A = (X^T)X, b = (X^T)Y

So, the closed-form solution is:

$$
\theta = (\mathbf{X}^T\mathbf{X})^{-1} (\mathbf{X}^T\mathbf{Y})
$$

Let's verify whether our formula is correct by comparing whether the results calculated from our formula are consistent with those obtained from sklearn (a python machine learning library)

```python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Now we randomly generate a one-dimensional array x and its corresponding result y (where y=5x+6):

x = 2*np.random.rand(100, 1)
y = 5*x + 6 + np.random.rand(100, 1)

# First, we generate a plot using the θ obtained from our formula. The blue dots represent the values of y corresponding to the array x, and the line is the result we fitted (don't forget to add bias term).

x_b = np.c_[np.ones(100),x]
theta = np.linalg.inv(x_b.T.dot(x_b)).dot((x_b.T).dot(y))
y_b = x_b.dot(theta)

plt.plot(x, y, "b.")
plt.plot(x,y_b, '--', label="result")
plt.legend()
plt.show()

# Then, we generate a plot using the result from sklearn

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x, y)
theta_0 = LR.intercept_
theta_1 = LR.coef_
y_sk = LR.predict(x)

plt.plot(x, y, "b.")
plt.plot(x,y_sk, '--', label="result")
plt.legend()
plt.show()

```

Let's verify the result:

The first one is the plot generated using our formula, and the second one is the plot generated from the results obtained by sklearn. We can see that they are basically the same, and through this verification, we can confirm that our formula is correct.

<img src="../images/LA result from the formula.png" alt="LA result from the formula" width="400"> <img src="../images/LA result from sklearn.png" alt="LA result from skearn" width="400">
