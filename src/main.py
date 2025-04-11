import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

# Feature engineering
def feature_engineering(df):
    df['Name'].fillna('', inplace=True)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
    df['Title'].replace(['Mme'], 'Mrs', inplace=True)
    df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir', 'Don', 'Dona', 'Jonkheer', 'Lady', 'Countess'], 'Rare', inplace=True)
    df['Woman_Child'] = ((df['Sex'] == 'female') | (df['Age'] < 12)).astype(int)
    df['Mother'] = ((df['Sex'] == 'female') & (df['Parch'] > 0) & (df['Age'] > 18)).astype(int)
    df['IsRich'] = ((df['Pclass'] == 1) & (df['Fare'] > df['Fare'].median())).astype(int)

    title_map = {
    'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 0
    }
    df['TitleRank'] = df['Title'].map(title_map)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['HasFamily'] = ((df['SibSp'] + df['Parch']) > 0).astype(int)

    df['Is1stClass'] = (df['Pclass'] == 1).astype(int)
    df['Is3rdClass'] = (df['Pclass'] == 3).astype(int)

    df['Female_1stClass'] = ((df['Sex'] == 'female') & (df['Pclass'] == 1)).astype(int)
    df['Child_1stClass'] = ((df['Age'] < 12) & (df['Pclass'] == 1)).astype(int)

    df['FareBand'] = pd.qcut(df['Fare'], 4, labels=False)
    df['AgeBand'] = pd.cut(df['Age'], 5, labels=False)

    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('U')
    df['CabinCount'] = df['Cabin'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'Deck'], drop_first=True)
    return df


# Preprocessing
def preprocess(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    return feature_engineering(df)

def main():
    # Load and preprocess training data
    train_df = pd.read_csv('../data/train.csv')
    train_df = preprocess(train_df)

    X_train = pd.get_dummies(train_df.drop(columns=['Survived']))
    y_train = train_df['Survived']
    print("Training data shape:", X_train.shape)
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42,)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    # Build more accurate Keras model
    model = Sequential([
        Dense(36, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(36, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(18, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(9, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    model.summary()
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=600,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    # Evaluate the model
    loss, accuracy = model.evaluate(X_val_scaled, y_val)
    y_pred = (model.predict(X_val_scaled) > 0.5).astype(int).reshape(-1)

    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    # Load and preprocess test data
    test_df = pd.read_csv('../data/test.csv')
    test_df = preprocess(test_df)
    test_X = pd.get_dummies(test_df)
    test_X = test_X.reindex(columns=X_train.columns, fill_value=0)
    test_X_scaled = scaler.transform(test_X)

    # Predict
    predictions = (model.predict(test_X_scaled) > 0.5).astype(int).reshape(-1)
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
    submission.to_csv('../data/submission.csv', index=False)
    print("Submission saved to ../data/submission.csv")

if __name__ == "__main__":
    main()
