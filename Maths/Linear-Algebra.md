I've been studying the basic linear algebra for machine learning these past couple of days, but it feels too exam-oriented to me. I've decided to record the linear algebra knowledge I've learned by working through a practical example.

Here is the example:

Derive the closed-form solution for linear regression using linear algebra.

Linear regression is a basic algorithm in ML. By learning the linear mapping relationship between input features (independent variables) and output labels (dependent variables), we fit a linear function that is used to predict continuous output values and also explain the degree of impact of independent variables on dependent variables.

For example, in the first question of CIE A-Level Physics: we find the best-fit line that explains the linear relationship between two different quantities (the feature on the x-axis and the label on the y-axis). But in linear regression (LR), we do not consider just a single feature.

We know the expression of a univariate linear function is y = θx + b, and this basic form also applies to LR. When there are multiple features x1, x2 ……， xn, the formula for LR is y = θ1x1 + θ2x2 + …… + θnxn. To unify the vectorized form of this formula, we add an additional term θ0x0, where x0 = 1, which represent 'b' in the y = θx + b and we call it the bias term.

We suppose there are m samples, each with n features, and we define:

X: A matrix which is m * (n + 1) remember the bias term. 
Y: An m * 1 column vector
θ: An (n + 1) * 1 column vecter

So there is a first definition: matrix

A matrix is a rectangular array of numbers, symbols, or mathematical expressions arranged in rows and columns. The dimension of a matrix is denoted as a * b which 'a' represents the number of rows and 'b' represents the number of columns

This matrix represents x:

$$
\mathbf{X} = \begin{bmatrix}
x_0 & x_1 & \dots & x_n \\ 
x_0' & x_1' & \dots & x_n' \\ 
\vdots & \vdots & \ddots & \vdots \\
x_0^{(m-1)} & x_1^{(m-1)} & \dots & x_n^{(m-1)}
\end{bmatrix}
$$
