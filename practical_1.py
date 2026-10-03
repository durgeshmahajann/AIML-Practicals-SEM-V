# AIML Practical 1
# Identify the Association Between Dependent and Independent Variables

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 45, 50, 55, 60, 65, 72, 80, 88]
}

df = pd.DataFrame(data)

# Display the dataset
print("Dataset:")
print(df)

# Calculate correlation
correlation = df["Study_Hours"].corr(df["Marks"])

print("\nCorrelation between Study Hours and Marks:", round(correlation, 2))

# Interpret the correlation
if correlation > 0:
    print("There is a positive association between the variables.")
elif correlation < 0:
    print("There is a negative association between the variables.")
else:
    print("There is no linear association between the variables.")

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Study_Hours", y="Marks", data=df)

plt.title("Association Between Study Hours and Marks")
plt.xlabel("Independent Variable: Study Hours")
plt.ylabel("Dependent Variable: Marks")
plt.grid(True)
plt.show()