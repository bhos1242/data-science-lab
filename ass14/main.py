import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load mtcars dataset
mtcars = sns.load_dataset("mpg").dropna()

# Rename columns to match mtcars style
mtcars = mtcars.rename(columns={"horsepower": "hp", "weight": "wt", "acceleration": "qsec"})

# -----------------------------
# Boxplot for mpg
# -----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(y=mtcars["mpg"], color="skyblue")
plt.title("Boxplot of Miles Per Gallon (mpg)")
plt.ylabel("mpg")
plt.show()

# -----------------------------
# Boxplot for mpg grouped by number of cylinders
# -----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x=mtcars["cylinders"], y=mtcars["mpg"], palette="Set2")
plt.title("MPG by Number of Cylinders")
plt.xlabel("Number of Cylinders")
plt.ylabel("MPG")
plt.show()
