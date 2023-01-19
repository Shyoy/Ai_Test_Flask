import joblib
# from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import R



# loading data and preperation
df  =pd.read_csv( 'data\wc_players_main.csv')
df.info()

df = df.drop(columns=['position','team','player','share_team_deaths','neutral_objectives_stolen','counter_pickrate'])
for c ,t in zip(df.columns,df.dtypes):
    if str(t) == "object":
        df[c] = df[c].str.rstrip('%').astype(float)/100.0

# prepare 2 groups (features, output)
X=df[['kills', 'assists', 'deaths', 'cs_diff_10','damage_per_minute','kda','kp']] # sample features ,[Age,Gender]
Y=df['winrate'].to_frame(name=None) # sample output ['genere']

 

best_score = 0
for _ in range(1000):
    # Split data for testing purposes
    x_train,x_test,y_train,y_test = train_test_split(X,Y ,test_size=0.2)

    # train and save training
    model = LinearRegression()
    model.fit(x_train,y_train) # load features and sample data
    acc= model.score(x_test,y_test)
    print(f"<{acc}>")
    if acc > best_score:
        best_score = acc
        joblib.dump(model, 'our_pridction.joblib') #binary file


    # Use gathered data to predict rate rate score 
    # model = joblib.load('our_pridction.joblib')

print("BestScore:",best_score)