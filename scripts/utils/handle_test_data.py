# Load and preprocess test data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import preprocess_data
import warnings
warnings.filterwarnings('ignore')

def handle_test_data(X_train, scaler, model):
    test_df = pd.read_csv('../data/test.csv')
    test_df = preprocess_data.preprocess(test_df)
    test_X = pd.get_dummies(test_df)
    test_X = test_X.reindex(columns=X_train.columns, fill_value=0)
    test_X_scaled = scaler.transform(test_X)

    # Predict
    predictions = (model.predict(test_X_scaled) > 0.5).astype(int).reshape(-1)
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
    submission.to_csv('../data/submission.csv', index=False)
    print("Submission saved to ../data/submission.csv")