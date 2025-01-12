import numpy as np

# Define the desired order for the ordinal columns, including np.nan to handle missing values
RESPONSE_TIME_ORDER = ['within an hour', 'within a few hours', 'within a day', 'a few days or more', np.nan]
CANCELLATION_POLICY_ORDER = ['flexible', 'moderate', 'strict', 'strict_14_with_grace_period', 'super_strict_30', 'super_strict_60', np.nan]
TEST_INPUT = {
    "feature1": 5.0,
    "feature2": 10.0,
    "feature3": 15.0
}
