import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_scale_data(train_df):
    X_train = pd.get_dummies(train_df.drop(columns=['Survived']))
    y_train = train_df['Survived']
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=43,)
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train, X_train_scaled, y_train, X_val_scaled, y_val, scaler
    # run the test data on the model