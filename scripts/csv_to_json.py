import pandas as pd

file_path = "data/total_data.csv"

# Define the range of rows to read (inclusive)
start_row = 10
end_row = 20

# Read the header and the specific row
header = pd.read_csv(file_path, nrows=0).columns.tolist()  # Read only the header

# Use a function for skiprows to skip rows outside the desired range
df = pd.read_csv(file_path, skiprows=lambda x: x < start_row or x > end_row)

# Assign the header to the DataFrame (if not automatically included)
df.columns = header

# Convert to JSON
json_output = df.to_json(orient="records", indent=4)  # Convert to JSON format
print(json_output)

# prepare filepath name
json_filepath = f"data/json/{start_row}-{end_row}_rows.json"

# Optionally save to a file
with open(json_filepath, "w") as f:
    f.write(json_output)
