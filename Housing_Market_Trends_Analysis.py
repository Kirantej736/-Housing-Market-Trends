# ============================================================
# HOUSING MARKET TRENDS ANALYSIS
# Data Science Internship Project
# Dataset: housing.csv
# ============================================================

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

print("Libraries imported successfully!")


# ============================================================
# 2. LOAD DATASET
# ============================================================

FILE_PATH = "dataset/housing.csv"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {FILE_PATH}\n"
        "Place housing.csv inside the dataset folder."
    )

df = pd.read_csv(FILE_PATH)

print("\nDataset loaded successfully!")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 3. DISPLAY FUNCTION
# ============================================================

try:
    from IPython.display import display
except ImportError:
    def display(data):
        print(data)


# ============================================================
# 4. UNDERSTAND THE DATASET
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)
display(df.head())

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
display(df.describe(include="all").T)


# ============================================================
# 5. CHECK DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)
print("Number of duplicate records:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate records removed.")
else:
    print("No duplicate records found.")


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_table = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage": (df.isnull().sum() / len(df) * 100).round(2)
})

display(missing_table)

if df.isnull().sum().sum() == 0:
    print("No missing values found.")


# ============================================================
# 7. DATA TYPE CONVERSION
# ============================================================

numeric_columns = [
    "Price",
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Sale_Year",
    "Year_Built",
    "Parking"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 8. DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

print("Invalid Price values :", (df["Price"] <= 0).sum())
print("Invalid Area values  :", (df["Area"] <= 0).sum())
print("Invalid Bedrooms     :", (df["Bedrooms"] <= 0).sum())
print("Invalid Bathrooms    :", (df["Bathrooms"] <= 0).sum())
print("Invalid Parking      :", (df["Parking"] < 0).sum())


# Remove invalid records if any
df = df[
    (df["Price"] > 0) &
    (df["Area"] > 0) &
    (df["Bedrooms"] > 0) &
    (df["Bathrooms"] > 0) &
    (df["Parking"] >= 0)
].copy()


# ============================================================
# 9. CREATE NEW FEATURE: PRICE PER SQFT
# ============================================================

df["Price_Per_Sqft"] = df["Price"] / df["Area"]

print("\nPrice_Per_Sqft column created successfully.")


# ============================================================
# 10. CREATE NEW FEATURE: PROPERTY AGE
# ============================================================

df["Property_Age"] = df["Sale_Year"] - df["Year_Built"]

print("Property_Age column created successfully.")


# ============================================================
# 11. CLEANED DATASET
# ============================================================

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

display(df.head())


# ============================================================
# 12. BASIC PRICE STATISTICS
# ============================================================

average_price = df["Price"].mean()
median_price = df["Price"].median()
highest_price = df["Price"].max()
lowest_price = df["Price"].min()

print("\n" + "=" * 60)
print("HOUSING PRICE STATISTICS")
print("=" * 60)

print(f"Average House Price : ₹{average_price:,.2f}")
print(f"Median House Price  : ₹{median_price:,.2f}")
print(f"Highest House Price : ₹{highest_price:,.2f}")
print(f"Lowest House Price  : ₹{lowest_price:,.2f}")


# ============================================================
# 13. PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Price",
    bins=15,
    kde=True
)

plt.title("Distribution of House Prices")
plt.xlabel("House Price (₹)")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.show()


# ============================================================
# 14. PRICE BOX PLOT
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["Price"]
)

plt.title("House Price Distribution and Outliers")
plt.xlabel("House Price (₹)")
plt.tight_layout()
plt.show()


# ============================================================
# 15. PRICE PER SQUARE FOOT
# ============================================================

print("\n" + "=" * 60)
print("PRICE PER SQUARE FOOT")
print("=" * 60)

print(
    f"Average Price/Sq.Ft : ₹{df['Price_Per_Sqft'].mean():,.2f}"
)

print(
    f"Median Price/Sq.Ft  : ₹{df['Price_Per_Sqft'].median():,.2f}"
)


# ============================================================
# 16. AREA VS PRICE
# ============================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="Area",
    y="Price",
    hue="Property_Type",
    style="Property_Type",
    s=80
)

plt.title("Property Area vs House Price")
plt.xlabel("Area (sq.ft)")
plt.ylabel("House Price (₹)")
plt.tight_layout()
plt.show()


# ============================================================
# 17. AREA-PRICE CORRELATION
# ============================================================

area_price_correlation = df[
    ["Area", "Price"]
].corr().iloc[0, 1]

print(
    "\nCorrelation between Area and Price:",
    round(area_price_correlation, 3)
)


# ============================================================
# 18. BEDROOM ANALYSIS
# ============================================================

bedroom_analysis = (
    df.groupby("Bedrooms")["Price"]
    .agg(
        Average_Price="mean",
        Median_Price="median",
        Property_Count="count"
    )
    .sort_index()
)

print("\n" + "=" * 60)
print("PRICE BY NUMBER OF BEDROOMS")
print("=" * 60)

display(bedroom_analysis)


# ============================================================
# 19. AVERAGE PRICE BY BEDROOMS
# ============================================================

bedroom_price = (
    df.groupby("Bedrooms")["Price"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=bedroom_price.index.astype(str),
    y=bedroom_price.values
)

plt.title("Average House Price by Number of Bedrooms")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average House Price (₹)")
plt.tight_layout()
plt.show()


# ============================================================
# 20. BATHROOM ANALYSIS
# ============================================================

bathroom_analysis = (
    df.groupby("Bathrooms")["Price"]
    .agg(
        Average_Price="mean",
        Median_Price="median",
        Property_Count="count"
    )
    .sort_index()
)

print("\n" + "=" * 60)
print("PRICE BY NUMBER OF BATHROOMS")
print("=" * 60)

display(bathroom_analysis)


# ============================================================
# 21. AVERAGE PRICE BY BATHROOMS
# ============================================================

bathroom_price = (
    df.groupby("Bathrooms")["Price"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=bathroom_price.index.astype(str),
    y=bathroom_price.values
)

plt.title("Average House Price by Number of Bathrooms")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Average House Price (₹)")
plt.tight_layout()
plt.show()


# ============================================================
# 22. LOCATION ANALYSIS
# ============================================================

location_analysis = (
    df.groupby("Location")["Price"]
    .agg(
        Average_Price="mean",
        Median_Price="median",
        Property_Count="count"
    )
    .sort_values(
        "Average_Price",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("PRICE BY LOCATION")
print("=" * 60)

display(location_analysis)


# ============================================================
# 23. LOCATION PRICE VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=location_analysis.reset_index(),
    x="Average_Price",
    y="Location"
)

plt.title("Average House Price by Location")
plt.xlabel("Average House Price (₹)")
plt.ylabel("Location")
plt.tight_layout()
plt.show()


# ============================================================
# 24. PROPERTY TYPE ANALYSIS
# ============================================================

property_type_analysis = (
    df.groupby("Property_Type")["Price"]
    .agg(
        Average_Price="mean",
        Median_Price="median",
        Property_Count="count"
    )
    .sort_values(
        "Average_Price",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("PRICE BY PROPERTY TYPE")
print("=" * 60)

display(property_type_analysis)


# ============================================================
# 25. PROPERTY TYPE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=property_type_analysis.reset_index(),
    x="Property_Type",
    y="Average_Price"
)

plt.title("Average House Price by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Average House Price (₹)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ============================================================
# 26. SALE YEAR ANALYSIS
# ============================================================

year_analysis = (
    df.groupby("Sale_Year")["Price"]
    .agg(
        Average_Price="mean",
        Median_Price="median",
        Property_Count="count"
    )
    .sort_index()
)

print("\n" + "=" * 60)
print("HOUSE PRICE BY SALE YEAR")
print("=" * 60)

display(year_analysis)


# ============================================================
# 27. HOUSE PRICE TREND
# ============================================================

yearly_price = (
    df.groupby("Sale_Year")["Price"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(10, 6))

sns.lineplot(
    x=yearly_price.index,
    y=yearly_price.values,
    marker="o"
)

plt.title("Average House Price Trend Over Years")
plt.xlabel("Sale Year")
plt.ylabel("Average House Price (₹)")
plt.tight_layout()
plt.show()


# ============================================================
# 28. PROPERTY AGE ANALYSIS
# ============================================================

age_price_correlation = df[
    ["Property_Age", "Price"]
].corr().iloc[0, 1]

print("\n" + "=" * 60)
print("PROPERTY AGE ANALYSIS")
print("=" * 60)

print(
    "Correlation between Property Age and Price:",
    round(age_price_correlation, 3)
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Property_Age",
    y="Price",
    s=80
)

plt.title("Property Age vs House Price")
plt.xlabel("Property Age (Years)")
plt.ylabel("House Price (₹)")
plt.tight_layout()
plt.show()


# ============================================================
# 29. PARKING ANALYSIS
# ============================================================

parking_analysis = (
    df.groupby("Parking")["Price"]
    .agg(
        Average_Price="mean",
        Median_Price="median",
        Property_Count="count"
    )
    .sort_index()
)

print("\n" + "=" * 60)
print("PRICE BY PARKING SPACES")
print("=" * 60)

display(parking_analysis)


# ============================================================
# 30. CORRELATION MATRIX
# ============================================================

numeric_columns_for_corr = [
    "Price",
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Sale_Year",
    "Year_Built",
    "Parking",
    "Price_Per_Sqft",
    "Property_Age"
]

correlation_matrix = df[
    numeric_columns_for_corr
].corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

display(correlation_matrix.round(2))


# ============================================================
# 31. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Housing Features")
plt.tight_layout()
plt.show()


# ============================================================
# 32. OUTLIER DETECTION USING IQR
# ============================================================

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

price_outliers = df[
    (df["Price"] < lower_bound) |
    (df["Price"] > upper_bound)
]

print("\n" + "=" * 60)
print("PRICE OUTLIER ANALYSIS")
print("=" * 60)

print(f"Q1                 : ₹{Q1:,.2f}")
print(f"Q3                 : ₹{Q3:,.2f}")
print(f"IQR                : ₹{IQR:,.2f}")
print(f"Lower Bound        : ₹{lower_bound:,.2f}")
print(f"Upper Bound        : ₹{upper_bound:,.2f}")
print(f"Number of Outliers : {len(price_outliers)}")


# ============================================================
# 33. TOP 10 MOST EXPENSIVE PROPERTIES
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 MOST EXPENSIVE PROPERTIES")
print("=" * 60)

display(
    df.sort_values(
        "Price",
        ascending=False
    ).head(10)
)


# ============================================================
# 34. TOP 10 LOWEST-PRICED PROPERTIES
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 LOWEST-PRICED PROPERTIES")
print("=" * 60)

display(
    df.sort_values(
        "Price",
        ascending=True
    ).head(10)
)


# ============================================================
# 35. SAVE CLEANED DATASET
# ============================================================

OUTPUT_PATH = "dataset/cleaned_housing.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("CLEANED DATASET SAVED")
print("=" * 60)

print("Saved to:", OUTPUT_PATH)


# ============================================================
# 36. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("HOUSING MARKET TRENDS ANALYSIS COMPLETED")
print("=" * 60)

print("Final number of properties:", len(df))
print("Final number of columns    :", len(df.columns))

print(f"Average house price       : ₹{average_price:,.2f}")
print(f"Median house price        : ₹{median_price:,.2f}")
print(f"Area-price correlation    : {area_price_correlation:.3f}")
print(f"Price outliers detected   : {len(price_outliers)}")

print("\nAnalysis completed successfully!")
