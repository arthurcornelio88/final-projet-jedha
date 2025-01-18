import pandas as pd
import numpy as np

# Load data
def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

def load_and_sample_data(file_path, sample_fraction=0.3, random_state=42):
    """
    Load and sample raw data.
    Args:
        file_path (str): Path to the dataset.
        sample_fraction (float): Fraction of data to retain.
        random_state (int): Seed for reproducibility.
    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    df_raw = load_data(file_path)  # Your load_data function
    df_sampled = df_raw.sample(frac=sample_fraction, random_state=random_state)
    return df_sampled

# Define the cleaning function
def clean_price(price_str):
    try:
        # Remove '$' and commas, then convert to float
        cleaned_price = round(float(price_str.replace('$', '').replace(',', '')), 2)
        return cleaned_price
    except ValueError:
        # Handle non-numeric values by returning NaN
        return pd.NA
  
# Numerical, categorical and ordinal columns function selection
def col_selection(X):

    """
    Performs feature selection on the input DataFrame `X`.

    1. Selects numerical columns by excluding 'object' type columns.
    2. Identifies numerical columns to be dropped based on a predefined list `num_describe_drop`.
    3. Defines lists for nominal and ordinal categorical columns.

    Returns:
        - numerical_columns: List of selected numerical column names.
        - ordinal_columns: List of selected ordinal categorical column names.
        - nominal_columns: List of selected nominal categorical column names.
    """

    ### Numerical
    X_num = X.select_dtypes(exclude="object")

    num_describe_drop = ['Unnamed: 0', 'scrape_id', 'availability_30','jurisdiction_names',
    'availability_60', 'availability_90','availability_365','thumbnail_url', 'minimum_minimum_nights',
    'calculated_host_listings_count', 'minimum_maximum_nights', 'maximum_minimum_nights',
    'maximum_maximum_nights', 'minimum_nights_avg_ntm','maximum_nights_avg_ntm',
    'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms']

    numerical_columns = X_num.columns.difference(num_describe_drop)

    ### Categorical

    nominal_columns = ['host_is_superhost','host_identity_verified', 'is_location_exact',
    'property_type', 'room_type','bed_type', 'instant_bookable', 'require_guest_profile_picture',
    'host_has_profile_pic']

    ordinal_columns = ['host_response_time', 'cancellation_policy']

    return numerical_columns, ordinal_columns, nominal_columns

# function that creates beds_per_bedroom column
def beds_per_bedroom(X):
    """
    Creates a new numerical feature 'beds_per_bedroom' by dividing the number of 'beds' by the number of 'bedrooms'.
    Handles potential division by zero by replacing 0 'bedrooms' with NaN, which will be addressed later in the pipeline (likely through imputation).
    """

    X.loc[:, 'beds_per_bedroom'] = X['beds'] / X['bedrooms'].replace(0, np.nan)

    return X

# Function for grouping categories in nominal categories
def group_infrequent_categories(X, threshold=0.005, original_columns=None):
    """
    Groups infrequent categories in the 'property_type' column into an 'Other' category.
    Infrequent categories are those with a relative frequency below the specified threshold.
    """
    # Convert X back to a DataFrame if it's a NumPy array (for easier column handling)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=original_columns)

    # Calculate value counts and relative frequencies for 'property_type'
    value_counts = X['property_type'].value_counts()
    relative_freqs = value_counts / len(X)

    # Identify significant categories based on the threshold
    significant_categories = relative_freqs[relative_freqs >= threshold].index

    # Replace infrequent categories with 'Other'
    X['property_type'] = X['property_type'].apply(lambda x: x if x in significant_categories else 'Other')

    # Return the modified 'property_type' column as a NumPy array, reshaped for compatibility
    return X[['property_type']].values.reshape(-1, 1)

def num_outlier(X):
    """
    Filters numerical outliers based on expected latitude and longitude ranges for the state of Rio de Janeiro.
    Outliers are replaced with NaN for subsequent imputation.
    """
    # Define valid bounds for the state of Rio de Janeiro (with buffer)
    rio_lat_bounds = (-24.1, -21.7)
    rio_lon_bounds = (-44.3, -40.5)

    # Identify outlier rows where latitude or longitude falls outside the bounds
    outlier_mask = (
        (X['latitude'] < rio_lat_bounds[0]) | (X['latitude'] > rio_lat_bounds[1]) |
        (X['longitude'] < rio_lon_bounds[0]) | (X['longitude'] > rio_lon_bounds[1])
    )

    # Replace outlier values with NaN
    X.loc[outlier_mask, ['latitude', 'longitude']] = np.nan

    # Debugging print to check how many outliers were found
    print(f"Number of outliers replaced with NaN: {outlier_mask.sum()}")

    return X

# Nominal outliers
def nom_outlier(X):
    """
    Handles nominal outliers in specific columns based on predefined outlier conditions.
    Replaces outlier values with NaNs for subsequent imputation.
    """

    # Define outlier conditions for different nominal columns
    outlier_conditions = (
        (X['host_is_superhost'] == '3') |
        (X['is_location_exact'] == '2019-03-13') |
        (X['room_type'] == '2019-03-06') |
        (X['bed_type'] == '10.0') |
        (X['host_has_profile_pic'] == '$501.00')
    )

    # List of columns where outliers are detected
    outlier_columns = ['host_is_superhost', 'is_location_exact', 'room_type', 'bed_type', 'host_has_profile_pic']

    # Replace outlier values with NaNs in the specified columns
    X.loc[outlier_conditions, outlier_columns] = np.nan

    return X
