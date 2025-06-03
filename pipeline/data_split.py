import sys
import pandas as pd
from sklearn.model_selection import train_test_split


MAP_OPEN_CREDIT      = {'opc': 1, 'nopc': 0}
MAP_NEG_AMMO         = {'neg_amm': 1, 'not_neg': 0}
MAP_INTEREST_ONLY    = {'int_only': 1, 'not_int': 0}
MAP_LUMP_SUM_PAYMENT = {'lpsm': 1, 'not_lpsm': 0}
MAP_AGE              = {'<25': 0, '25-34': 1, '35-44': 2, '45-54': 3, '55-64': 4, '65-74': 5, '>74': 6 }
MAP_REGION           = {'south':0, 'North': 1, 'central': 2, 'North-East': 3}
MAP_BS_OR_COMM       = {'nob/c': 0, 'b/c': 1}
MAP_OCC_TYPE         = {'pr': 0, 'sr': 1, 'ir': 2}
MAP_SECURED_BY       = {'home': 0, 'land': 1}
LOW_QUANTILE = 0.10
HIGH_QUANTILE = 0.90
IRQ_COEFF = 3


def data_preprocessing(ds):
    # Remove unnecessary columns
    important_df =  ds[['loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges',
                        'term', 'property_value', 'LTV', 'Credit_Score', 'income', 'dtir1',
                        'open_credit', 'Neg_ammortization', 'interest_only', 'lump_sum_payment',
                        'age', 'Region', 'business_or_commercial', 'occupancy_type', 'Secured_by',
                        'Status']]
    # Categorical features
    important_df.loc[:, 'open_credit']            = important_df['open_credit'].map(MAP_OPEN_CREDIT)
    important_df.loc[:, 'Neg_ammortization']      = important_df['Neg_ammortization'].map(MAP_NEG_AMMO)
    important_df.loc[:, 'interest_only']          = important_df['interest_only'].map(MAP_INTEREST_ONLY)
    important_df.loc[:, 'lump_sum_payment']       = important_df['lump_sum_payment'].map(MAP_LUMP_SUM_PAYMENT)
    important_df.loc[:, 'age']                    = important_df['age'].map(MAP_AGE)
    important_df.loc[:, 'Region']                 = important_df['Region'].map(MAP_REGION)
    important_df.loc[:, 'business_or_commercial'] = important_df['business_or_commercial'].map(MAP_BS_OR_COMM)
    important_df.loc[:, 'occupancy_type']         = important_df['occupancy_type'].map(MAP_OCC_TYPE)
    important_df.loc[:, 'Secured_by']             = important_df['Secured_by'].map(MAP_SECURED_BY)

    important_df = important_df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)

    important_df.drop(columns=['LTV', 'dtir1', 'Interest_rate_spread'], inplace=True)

    # Handling Nans
    missing = list()
    for x in important_df.columns:
        if important_df[x].isnull().sum() != 0:
            missing.append(x)

    for col in missing:
        if col == 'Neg_ammortization' or col == 'age':
            important_df[col].fillna(important_df[col].mode()[0], inplace=True)
        else:
            important_df[col].fillna(important_df[col].median(), inplace=True)

    columns_to_convert = ['Neg_ammortization', 'age']
    important_df[columns_to_convert] = important_df[columns_to_convert].astype(int)

    # Replacing anomalies
    def replace_anomalies_with_minmax_values(df, column):
        sorted_values = df[column].sort_values()
        Q1 = sorted_values.quantile(LOW_QUANTILE)
        Q3 = sorted_values.quantile(HIGH_QUANTILE)
        IQR = Q3 - Q1
        lower_bound = Q1 - IRQ_COEFF * IQR
        upper_bound = Q3 + IRQ_COEFF * IQR

        min_real_value = sorted_values[sorted_values >= lower_bound].min()
        max_real_value = sorted_values[sorted_values <= upper_bound].max()

        df.loc[df[column] < lower_bound, column] = min_real_value
        df.loc[df[column] > upper_bound, column] = max_real_value

    numerical_columns = ['loan_amount', 'rate_of_interest', 'Upfront_charges', 'term', 'property_value', 'Credit_Score', 'income']
    for col in numerical_columns:
        replace_anomalies_with_minmax_values(important_df, col)

    return important_df


def split_data(input_filename, train_filename, val_filename):
    df = data_preprocessing(pd.read_csv(input_filename))

    X = df.drop(columns=['Status'])
    y = df['Status']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df['Status'] = y_train

    val_df = X_val.copy()
    val_df['Status'] = y_val

    train_df.to_csv(train_filename, index=False)
    val_df.to_csv(val_filename, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_split.py <input_file> <train_file> <val_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    train_file = sys.argv[2]
    val_file = sys.argv[3]
    split_data(input_file, train_file, val_file)
