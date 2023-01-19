import joblib
# from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df_predict = pd. (['TCL', 2018, 'Season', 'YC', 0, 'SUP', 'Elwind', 'Mojito',
        'CoCo', 'Madness', 'Zzus', 'fabFabulous', 'Stomaged', 'GBM',
        'Zeitnot', 'SnowFlower'])

print(df_predict)

# model = joblib.load('our_pridction.joblib')

# predictions= model.predict(df)
# print(predictions)
# df  =pd.read_csv( 'data\wc_players_main.csv')
# df.info()

# df = df.drop(columns=['position','team','player','share_team_deaths','neutral_objectives_stolen','counter_pickrate'])
# for c ,t in zip(df.columns,df.dtypes):
#     if str(t) == "object":
#         df[c] = df[c].str.rstrip('%').astype(float)/100.0

# # prepare 2 groups (features, output)
# X=df[['kills', 'assists', 'deaths', 'cs_diff_10','damage_per_minute','kda','kp']] # sample features ,[Age,Gender]
# Y=df['winrate'].to_frame(name=None)


# acc = model.score(X,Y)

# print("BestScore:", acc)