import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("libraries import successfully!")

#loading Dataset
df = pd.read_csv("Student_data.csv")

#show first 5 rows
print("\nfirst 5 rows:")
print(df.head())

#show number of rows & columns
print("\nShape of dataset:")
print(df.shape)

#show column information
print("\nDataset information:")
print(df.info())

#cheak missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(df.describe()) 

#filling missing numarical values using median
df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].median())
df["Previous_Score"] = df["Previous_Score"].fillna(df["Previous_Score"].median())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())
df["Family_Income"] = df["Family_Income"].fillna(df["Family_Income"].median())

# Fill missing City with the most common city
df["City"] = df["City"].fillna(df["City"].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

#convert catagorical columns into numarical columns
df = pd.get_dummies(df, columns=["Gender", "City"], dtype=int)

#convert passed into 0 and 1 
df["Passed"] = df["Passed"].map({"No": 0, "Yes": 1})

print("\nDataset after encoding:")
print(df.head())

from sklearn.preprocessing import StandardScaler

# Numerical columns to standardize
numeric_columns = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Family_Income"
]

# Create scaler
#scaler = StandardScaler()

# Standardize the numerical columns
#df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
#print("\nDataset after standardization:")
#print(df.head())

# Remove outliers using IQR method

Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)

IQR = Q3 - Q1

# Keep only normal values
df = df[
    ~((df[numeric_columns] < (Q1 - 1.5 * IQR)) |
      (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
]

print("\nDataset after removing outliers:")
print(df.shape)

# Standardize numerical columns

scaler = StandardScaler()

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nDataset after standardization:")
print(df.head())

# Visualize outliers using boxplot

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df[
        ["Age", "Study_Hours", "Attendance",
         "Previous_Score", "Family_Income"]
    ]
)

plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)

plt.show()

print("\nRows with possible extreme values:")
print(df[df["Study_Hours"] > 20])
print(df[df["Family_Income"] > 100000])


