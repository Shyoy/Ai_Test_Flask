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

np.random.seed(2)

cols = ['Survived','Sex','Age','SibSp','Parch',
    'Fare','Cabin','Embarked']
df = pd.read_csv('Titanic\data\\train.csv', usecols = cols )
X_train = df.drop(columns=['Survived'])
y_train = df['Survived'].astype('category')


numeric_features = ['Age','SibSp','Parch',
    'Fare']
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ['Sex','Cabin','Embarked']
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(solver='lbfgs', max_iter=100))]
)


cols = ['PassengerId', 'Sex','Age','SibSp','Parch',
    'Fare','Cabin','Embarked']
df_test1 = pd.read_csv('Titanic\data\\test.csv', usecols = cols )

cols = ['PassengerId','Survived']
df_test2 = pd.read_csv('Titanic\data\\gender_submission.csv', usecols = cols )

test_df = pd.merge(df_test2, df_test1,validate="one_to_one")
test_df = test_df.drop(columns=['PassengerId'])


X_test = test_df.drop(columns=['Survived'])
y_test = test_df['Survived'].astype('category')



# for i in range(200):

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("model score: %.3f" % score)
# if score > best:
#     best = score
# joblib.dump(clf,'model.joblib')
joblib.dump(clf, 'Titanic\model.joblib')


# print("Best Model score: %.3f" % best)