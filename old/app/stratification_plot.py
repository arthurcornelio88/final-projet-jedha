import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'data/total_data.csv' with the path to your CSV file
data_file = 'data/total_data.csv'
df_raw = pd.read_csv(data_file)

# Define a function to clean the 'price' column
def clean_price(price_str):
    try:
        # Remove '$' and commas, then convert to float
        cleaned_price = round(float(price_str.replace('$', '').replace(',', '')), 2)
        return cleaned_price
    except ValueError:
        # Handle non-numeric values by returning NaN
        return np.nan

# Apply the cleaning function to the 'price' column
df_raw['price_cleaned'] = df_raw['price'].astype(str).apply(clean_price)

# Apply log transformation to handle high prices
df_raw['price_cleaned'] = np.log1p(df_raw['price_cleaned'])

# Plot the histogram of the log-transformed prices
plt.figure(figsize=(10, 6))
plt.hist(df_raw['price_cleaned'].dropna(), bins=50, edgecolor='k', alpha=0.7)
plt.title("Distribution of Log-Transformed Prices")
plt.xlabel("Log(Price + 1)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot as an image file
output_file = "log_price_distribution.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Histogram saved as {output_file}")
