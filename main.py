# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load data
df = pd.read_csv("./Data/student_scores.csv")

# Plot
# plt.xlabel("Hours")
# plt.ylabel("Scores")
# plt.scatter(df["Hours"], df["Scores"])
# plt.show()

# creat model
reg = linear_model.LinearRegression()

# train model
reg.fit(df[["Hours"]], df["Scores"])

#predict
print(reg.predict([[7.8]]))

# Plot Linear regresion line
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.scatter(df["Hours"], df["Scores"])
plt.plot(df["Hours"], reg.predict(df[["Hours"]]), color = "red")
plt.show()