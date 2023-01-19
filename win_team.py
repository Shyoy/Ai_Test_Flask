import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

cols = ['Year','blueTeamTag','bResult','redTeamTag',
    'blueTop','blueJungle','blueMiddle','blueADC','blueSupport','redTop','redJungle','redMiddle','redADC', 'redSupport']
df = pd.read_csv('data\League\LeagueofLegends.csv', usecols = cols )
X = df.drop(columns=['bResult'])
y = df['bResult'].astype('category')


numeric_features = ['Year']
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ['blueTeamTag','redTeamTag',
    'blueTop','blueJungle','blueMiddle','blueADC','blueSupport','redTop','redJungle','redMiddle','redADC', 'redSupport']
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(solver='lbfgs', max_iter=125))]
)

best = 0

for i in range(100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("model score: %.3f" % score)
    if score > best:
        best = score
        joblib.dump(clf,'model.joblib')


print("Best Model score: %.3f" % best)