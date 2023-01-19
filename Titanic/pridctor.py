import joblib
import pandas as pd




clf = joblib.load('Titanic\model.joblib')





data = [['female'], [23],
        [None], [None],
        [None], [None], [None]]
header =['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
data = dict(zip(header, data))
df_for_predict = pd.DataFrame.from_dict(data)
# print(df_for_predict.info())


statistics = []
for i in range(1):
    prediction = clf.predict(df_for_predict)

    if prediction[0]:
        statistics += ['Survived']
        print("Survived")
    else:
        statistics += ['Died']
        print("Died")

survival_rate = statistics.count('Survived') / len(statistics)

print(f'Survival rate:{survival_rate}%')
# if prediction[0]:
#         statistics['Survived']
#         print("Survived")
#     else:
#         print("Died")