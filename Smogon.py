import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Import both datasets then merge based on common names
pokemon_data = pd.read_csv('Pokemon.csv')
smogon_data = pd.read_csv('smogondata.csv')
poke_smogon = pd.merge(left=pokemon_data, right=smogon_data, left_on='Name', right_on='Name')

# Create the set of X and Y variables
X = poke_smogon.iloc[:, 6:7].values  # Pokemon Attack
Y = poke_smogon.iloc[:, 14].values  # Pokemon Usage Percentage

# Fit the regression model
regr_1 = DecisionTreeRegressor(max_depth=4, random_state=0)
regr_2 = DecisionTreeRegressor(max_depth=8, random_state=0)
regr_1.fit(X, Y)
regr_2.fit(X, Y)

# Make predictions
X_test = np.arange(0, 250, 5)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Use Matplotlib to plot the resulting prediction
plt.figure()
plt.scatter(X, Y, edgecolor="black", c="lightblue", label="Pokemon")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=4", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=8", linewidth=2)
plt.xlabel("Attack")
plt.ylabel("Usage %")
plt.title("Decision Tree Regression: Pokemon Attack vs. Usage %")
plt.legend()
plt.show()













