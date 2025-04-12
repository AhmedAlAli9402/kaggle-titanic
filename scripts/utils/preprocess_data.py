import pandas as pd

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
    df.loc[(df['Age'].isnull()) & (df['Parch'] > 0) & (df['Parch']< 3), 'Age'] = 10
    df.loc[(df['Age'].isnull()) & (df['Parch'] == 0), 'Age'] = 30
    df['Age'].fillna(df['Age'].median(), inplace=True)
    return feature_engineering(df)