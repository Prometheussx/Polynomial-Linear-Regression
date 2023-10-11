# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:51:11 2023
@author: erdem
"""
# Polynomial regression is used for data that stabilizes at the end and can be defined as an exponential function.
import pandas as pd
import matplotlib.pyplot as plt 
df = pd.read_csv("polynomialregression.csv", sep=";")

x = df.car_price.values.reshape(-1, 1) # Independent variable 
# The 'values' method converts the Series type to an array of int64, and 'reshape' transforms the size from (15,) to (15, 1).

y = df.car_max_speed.values.reshape(-1, 1) # Dependent variable

plt.scatter(x, y)
plt.ylabel("Car's Max Speed")
plt.xlabel("Car's Price")
plt.show()

# Linear Regression & Multiple Linear Regression
# Linear: y = b0 + b1 * x
# Multiple: y = b0 + b1 * x1 + b2 * x2 + ... + bn * xn

# Linear Regression
# from sklearn.linear_model import LinearRegression

# lr = LinearRegression()

# lr.fit(x, y)
# y_head = lr.predict(x)

# plt.plot(x, y_head, color="red")
# plt.show()

# Polynomial Linear Regression (y = b0 + b1 * x1 + b2 * x2^2 + ... + bn * xn^n)
# As the degree increases, the polynomial becomes more complex and the error rate decreases.
# There is always a hidden x^0 in b0, so it is always 1.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polynomial_regression = PolynomialFeatures(degree=2)  # The degree specifies the polynomial degree, how many 'x' there are, it adds that many.

x_polynomial = polynomial_regression.fit_transform(x) # Use fit_transform with x^2 and make the new 'x' x^2.
# We trained based on our real 'y' values and 'x' values, and with this trained model, we provide 'x' values to predict new values.
# We will create a line, since it's the same as the linear structure, we just applied linear regression with 'x' raised.
linear_regression = LinearRegression()
linear_regression.fit(x_polynomial, y)

y_head = linear_regression.predict(x_polynomial)

plt.plot(x, y_head, color="green", label="poly")
plt.legend()
plt.show()
