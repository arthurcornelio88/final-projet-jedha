from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, FunctionTransformer
from app.data_preprocessing import (
    clean_price,
    col_selection,
    beds_per_bedroom,
    group_infrequent_categories,
    num_outlier,
    nom_outlier
)
from sklearn.model_selection import train_test_split
from app.static_values import RESPONSE_TIME_ORDER, CANCELLATION_POLICY_ORDER
import numpy as np
import pandas as pd

def process_and_pipeline(df_raw, mlflow=None, strat=False):
    """
    Process raw data and execute the pipeline.

    Args:
        df_raw (pd.DataFrame): Raw dataset.
        test_size (float): Proportion of test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train_prepared (np.ndarray): Processed training data.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features (raw).
        y_test (pd.Series): Test labels.
    """
    # Apply the function to create the new column
    df_raw['price_cleaned'] = df_raw['price'].astype(str).apply(clean_price)

    # Preparing target (price) for stratifying dataset
    df_raw["price_cat"] = pd.cut(df_raw["price_cleaned"],
        bins=[0., 150, 300, 700, 1500, np.inf], # fine-tune bin for bell-shape like distribution
        labels=[1, 2, 3, 4, 5])

    # Drop rows with NaN in 'price_cat'
    df_cleaned = df_raw.dropna(subset=['price_cat']).reset_index(drop=True)

    if strat:
        # Stratify split
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(df_cleaned, df_cleaned["price_cat"]):
            strat_train_set = df_cleaned.loc[train_index]
            strat_test_set = df_cleaned.loc[test_index]
    else:
        # Option 2 for spliting the dataset, without stratification
        strat_train_set, strat_test_set = train_test_split(
            df_cleaned, test_size=0.2, random_state=42
        )

    # droping "price_cat" to make dataset returns to its original state
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("price_cat", axis=1, inplace=True)

    # Separate features and labels
    X_train = strat_train_set.drop("price_cleaned", axis=1)
    y_train = strat_train_set["price_cleaned"].copy()
    X_test = strat_test_set.drop("price_cleaned", axis=1)
    y_test = strat_test_set["price_cleaned"].copy()

    # Column selection
    numerical_columns, ordinal_columns, nominal_columns = col_selection(X_train)

    # Preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('feature_creation', FunctionTransformer(beds_per_bedroom)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', MinMaxScaler())
            ]), numerical_columns),
            ('categorical_ordinal', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(categories=[RESPONSE_TIME_ORDER, CANCELLATION_POLICY_ORDER],
                                                   handle_unknown='use_encoded_value', unknown_value=-1))
            ]), ordinal_columns),
            ('categorical_nominal', Pipeline([
                ('grouper', FunctionTransformer(group_infrequent_categories, kw_args={'original_columns': nominal_columns})),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), nominal_columns)
        ])

    pipeline = Pipeline([
        ('num_outlier', FunctionTransformer(num_outlier)),
        ('nominal_outlier', FunctionTransformer(nom_outlier)),
        ('preprocessor', preprocessor)
    ])

    print(f"Pipeline created!")

    # Fit and transform the pipeline on the training set
    X_train_prepared = pipeline.fit_transform(X_train)

    # Transform the test set with parameters learnt during training phase
    X_test_transformed = pipeline.transform(X_test)

    print(f"Training set fitted and transformed!")

    return X_train_prepared, y_train, X_test_transformed, y_test, pipeline
