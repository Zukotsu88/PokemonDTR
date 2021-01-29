import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
poke_data = pd.read_csv('pokemon.csv', index_col='Name')
X = poke_data.iloc[:, 6:7].values  # Pokemon Attack
Y = poke_data.iloc[:, 8].values  # Pokemon Sp. Attack

# Fit the regression model
regr_1 = DecisionTreeRegressor(max_depth=3, random_state=0)
regr_2 = DecisionTreeRegressor(max_depth=6, random_state=0)
regr_1.fit(X, Y)
regr_2.fit(X, Y)

# Make predictions
X_test = np.arange(0.0, 150, 5)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Use Matplotlib to plot the resulting prediction
plt.figure()
plt.scatter(X, Y, edgecolor="black", c="lightblue", label="Pokemon")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=3", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=6", linewidth=2)
plt.xlabel("Attack")
plt.ylabel("Sp. Attack")
plt.title("Decision Tree Regression: Pokemon Attack vs. Sp. Attack")
plt.legend()
plt.show()
