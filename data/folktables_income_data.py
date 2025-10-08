import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome


def load_folktables_data(sample_size=50000):
    """
    Load and preprocess the Folktables ACSIncome dataset.

    This function loads data for California from the 2018 1-Year ACS survey
    and preprocesses it to extract age and income.

    Parameters
    ----------
    sample_size : int, optional
        Number of samples to draw for faster computation. Defaults to 50000.

    Returns
    -------
    age : np.ndarray
        Array of ages.
    income : np.ndarray
        Array of binary income labels (0 for <$50,000, 1 for >=$50,000).
    """
    print("Loading Folktables dataset...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)

    # Get employment features
    features, label, group = ACSIncome.df_to_numpy(acs_data)

    # Convert to DataFrame for easier manipulation
    feature_names = ACSIncome.features
    df = pd.DataFrame(features, columns=feature_names)
    df['income'] = label

    # Get age and income data
    age_data = df['AGEP'].values  # Age
    income_data = df['income'].values  # Income (0={<$50,000}, 1={>=$50,000})

    # Filter out missing values
    valid_mask = ~(np.isnan(age_data) | np.isnan(income_data))
    age_clean = age_data[valid_mask]
    income_clean = income_data[valid_mask]

    print(f"Data size: {len(age_clean)}")
    print(f"Age range: {age_clean.min()} - {age_clean.max()}")
    print(f"Income over $50,000: {income_clean.mean():.3f}")

    # Sample a subset for faster computation (optional)
    if len(age_clean) > sample_size:
        indices = np.random.choice(len(age_clean), sample_size, replace=False)
        age_sample = age_clean[indices]
        income_sample = income_clean[indices]
        print(f"Using sample of {sample_size} data points for faster computation")
    else:
        age_sample = age_clean
        income_sample = income_clean

    print(f"Final dataset size: {len(age_sample)}")

    return age_sample, income_sample