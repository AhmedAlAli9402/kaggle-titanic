import pandas as pd
from utils import preprocess_data, plot_graphs, handle_test_data, create_model, split_scale_data
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load and preprocess training data
    # plot_graphs.plot_survived_vs_pclass(train_df)
    train_df = pd.read_csv('../data/train.csv')
    train_df = preprocess_data.preprocess(train_df)
    X_train, X_train_scaled, y_train, X_val_scaled, y_val, scaler = split_scale_data.split_scale_data(train_df)
    model = create_model.create_model(X_train_scaled, y_train, X_val_scaled, y_val)
    handle_test_data.handle_test_data(X_train, scaler, model)
if __name__ == "__main__":
    main()
