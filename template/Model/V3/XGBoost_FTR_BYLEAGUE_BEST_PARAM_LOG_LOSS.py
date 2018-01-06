
# coding: utf-8

# ## Import libraries

# In[1]:

# Importing the library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from IPython.display import display # Manage multiple output per cell
import datetime
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss


# ## Configuration

# In[2]:

odd_H = 'INFO_BbAvH'
odd_A = 'INFO_BbAvA'
odd_D = 'INFO_BbAvD'
target = 'INFO_FTR'
start_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
season_list = [2014, 2015, 2016]
league_list = ['D1', 'E0', 'E1', 'E2', 'F1', 'I1', 'SP1', 'SC0']
historical_training_year_list = [1, 3, 5, 9]


# In[3]:

all_features = ["A_MEANS_FIVE_AC","A_MEANS_FIVE_AF","A_MEANS_FIVE_AR","A_MEANS_FIVE_AS","A_MEANS_FIVE_AST","A_MEANS_FIVE_AY","A_MEANS_FIVE_FTAG","A_MEANS_FIVE_FTHG","A_MEANS_FIVE_FTR_A","A_MEANS_FIVE_FTR_D","A_MEANS_FIVE_FTR_H","A_MEANS_FIVE_HC","A_MEANS_FIVE_HF","A_MEANS_FIVE_HR","A_MEANS_FIVE_HS","A_MEANS_FIVE_HST","A_MEANS_FIVE_HTAG","A_MEANS_FIVE_HTHG","A_MEANS_FIVE_HTR_A","A_MEANS_FIVE_HTR_D","A_MEANS_FIVE_HTR_H","A_MEANS_FIVE_HY","H_MEANS_FIVE_AC","H_MEANS_FIVE_AF","H_MEANS_FIVE_AR","H_MEANS_FIVE_AS","H_MEANS_FIVE_AST","H_MEANS_FIVE_AY","H_MEANS_FIVE_FTAG","H_MEANS_FIVE_FTHG","H_MEANS_FIVE_FTR_A","H_MEANS_FIVE_FTR_D","H_MEANS_FIVE_FTR_H","H_MEANS_FIVE_HC","H_MEANS_FIVE_HF","H_MEANS_FIVE_HR","H_MEANS_FIVE_HS","H_MEANS_FIVE_HST","H_MEANS_FIVE_HTAG","H_MEANS_FIVE_HTHG","H_MEANS_FIVE_HTR_A","H_MEANS_FIVE_HTR_D","H_MEANS_FIVE_HTR_H","H_MEANS_FIVE_HY","A_MEANS_THREE_AC","A_MEANS_THREE_AF","A_MEANS_THREE_AR","A_MEANS_THREE_AS","A_MEANS_THREE_AST","A_MEANS_THREE_AY","A_MEANS_THREE_FTAG","A_MEANS_THREE_FTHG","A_MEANS_THREE_FTR_A","A_MEANS_THREE_FTR_D","A_MEANS_THREE_FTR_H","A_MEANS_THREE_HC","A_MEANS_THREE_HF","A_MEANS_THREE_HR","A_MEANS_THREE_HS","A_MEANS_THREE_HST","A_MEANS_THREE_HTAG","A_MEANS_THREE_HTHG","A_MEANS_THREE_HTR_A","A_MEANS_THREE_HTR_D","A_MEANS_THREE_HTR_H","A_MEANS_THREE_HY","H_MEANS_THREE_AC","H_MEANS_THREE_AF","H_MEANS_THREE_AR","H_MEANS_THREE_AS","H_MEANS_THREE_AST","H_MEANS_THREE_AY","H_MEANS_THREE_FTAG","H_MEANS_THREE_FTHG","H_MEANS_THREE_FTR_A","H_MEANS_THREE_FTR_D","H_MEANS_THREE_FTR_H","H_MEANS_THREE_HC","H_MEANS_THREE_HF","H_MEANS_THREE_HR","H_MEANS_THREE_HS","H_MEANS_THREE_HST","H_MEANS_THREE_HTAG","H_MEANS_THREE_HTHG","H_MEANS_THREE_HTR_A","H_MEANS_THREE_HTR_D","H_MEANS_THREE_HTR_H","H_MEANS_THREE_HY","A_STD_FIVE_AC","A_STD_FIVE_AF","A_STD_FIVE_AR","A_STD_FIVE_AS","A_STD_FIVE_AST","A_STD_FIVE_AY","A_STD_FIVE_FTAG","A_STD_FIVE_FTHG","A_STD_FIVE_FTR_A","A_STD_FIVE_FTR_D","A_STD_FIVE_FTR_H","A_STD_FIVE_HC","A_STD_FIVE_HF","A_STD_FIVE_HR","A_STD_FIVE_HS","A_STD_FIVE_HST","A_STD_FIVE_HTAG","A_STD_FIVE_HTHG","A_STD_FIVE_HTR_A","A_STD_FIVE_HTR_D","A_STD_FIVE_HTR_H","A_STD_FIVE_HY","H_STD_FIVE_AC","H_STD_FIVE_AF","H_STD_FIVE_AR","H_STD_FIVE_AS","H_STD_FIVE_AST","H_STD_FIVE_AY","H_STD_FIVE_FTAG","H_STD_FIVE_FTHG","H_STD_FIVE_FTR_A","H_STD_FIVE_FTR_D","H_STD_FIVE_FTR_H","H_STD_FIVE_HC","H_STD_FIVE_HF","H_STD_FIVE_HR","H_STD_FIVE_HS","H_STD_FIVE_HST","H_STD_FIVE_HTAG","H_STD_FIVE_HTHG","H_STD_FIVE_HTR_A","H_STD_FIVE_HTR_D","H_STD_FIVE_HTR_H","H_STD_FIVE_HY","A_STD_THREE_AC","A_STD_THREE_AF","A_STD_THREE_AR","A_STD_THREE_AS","A_STD_THREE_AST","A_STD_THREE_AY","A_STD_THREE_FTAG","A_STD_THREE_FTHG","A_STD_THREE_FTR_A","A_STD_THREE_FTR_D","A_STD_THREE_FTR_H","A_STD_THREE_HC","A_STD_THREE_HF","A_STD_THREE_HR","A_STD_THREE_HS","A_STD_THREE_HST","A_STD_THREE_HTAG","A_STD_THREE_HTHG","A_STD_THREE_HTR_A","A_STD_THREE_HTR_D","A_STD_THREE_HTR_H","A_STD_THREE_HY","H_STD_THREE_AC","H_STD_THREE_AF","H_STD_THREE_AR","H_STD_THREE_AS","H_STD_THREE_AST","H_STD_THREE_AY","H_STD_THREE_FTAG","H_STD_THREE_FTHG","H_STD_THREE_FTR_A","H_STD_THREE_FTR_D","H_STD_THREE_FTR_H","H_STD_THREE_HC","H_STD_THREE_HF","H_STD_THREE_HR","H_STD_THREE_HS","H_STD_THREE_HST","H_STD_THREE_HTAG","H_STD_THREE_HTHG","H_STD_THREE_HTR_A","H_STD_THREE_HTR_D","H_STD_THREE_HTR_H","H_STD_THREE_HY"]
best_features_60 = ['A_MEANS_FIVE_AC', 'A_MEANS_FIVE_AS', 'A_MEANS_FIVE_AST',
       'A_MEANS_FIVE_AY', 'A_MEANS_FIVE_FTAG', 'A_MEANS_FIVE_FTHG',
       'A_MEANS_FIVE_FTR_A', 'A_MEANS_FIVE_FTR_H', 'A_MEANS_FIVE_HC',
       'A_MEANS_FIVE_HF', 'A_MEANS_FIVE_HS', 'A_MEANS_FIVE_HST',
       'A_MEANS_FIVE_HTHG', 'A_MEANS_FIVE_HTR_A', 'A_MEANS_FIVE_HY',
       'H_MEANS_FIVE_AC', 'H_MEANS_FIVE_AS', 'H_MEANS_FIVE_AST',
       'H_MEANS_FIVE_AY', 'H_MEANS_FIVE_FTAG', 'H_MEANS_FIVE_FTHG',
       'H_MEANS_FIVE_FTR_A', 'H_MEANS_FIVE_FTR_H', 'H_MEANS_FIVE_HC',
       'H_MEANS_FIVE_HF', 'H_MEANS_FIVE_HS', 'H_MEANS_FIVE_HST',
       'H_MEANS_FIVE_HTAG', 'H_MEANS_FIVE_HTR_A', 'H_MEANS_FIVE_HTR_H',
       'A_MEANS_THREE_AC', 'A_MEANS_THREE_AS', 'A_MEANS_THREE_FTHG',
       'A_MEANS_THREE_FTR_A', 'A_MEANS_THREE_HC', 'A_MEANS_THREE_HF',
       'A_MEANS_THREE_HS', 'A_MEANS_THREE_HST', 'H_MEANS_THREE_AC',
       'H_MEANS_THREE_AS', 'H_MEANS_THREE_AST', 'H_MEANS_THREE_FTHG',
       'H_MEANS_THREE_HC', 'H_MEANS_THREE_HST', 'H_MEANS_THREE_HTR_H',
       'A_STD_FIVE_AF', 'A_STD_FIVE_AS', 'A_STD_FIVE_AST', 'A_STD_FIVE_HC',
       'A_STD_FIVE_HF', 'A_STD_FIVE_HS', 'H_STD_FIVE_AF', 'H_STD_FIVE_AS',
       'H_STD_FIVE_AST', 'H_STD_FIVE_HC', 'H_STD_FIVE_HF',
       'H_STD_FIVE_HST', 'H_STD_FIVE_HTHG', 'H_STD_THREE_AS',
       'H_STD_THREE_HST']
features_list = [
    ['all_features', all_features],
    ['best_features_60', best_features_60],
]


# ## Import Data

# In[4]:

# DB Sqlite connection
import sqlite3
db = "/Users/thibaultclement/Project/ligue1-predict/src/notebook/data/db/soccer_predict.sqlite"
conn = sqlite3.connect(db)
cur = conn.cursor()


# In[5]:

# Get all prematch data
df_all = pd.read_sql_query("SELECT * FROM pre_matchs ORDER BY INFO_Date ASC;", conn)
df_all = (df_all[df_all.columns.drop(['index'])])
df_all.shape


# In[6]:

# Remove all game between June (include) and October (include)
df_all['INFO_Date'] = pd.to_datetime(df_all['INFO_Date'])
df_all['INFO_Date'].dt.month
df_all = df_all[(df_all['INFO_Date'].dt.month < 6) | (df_all['INFO_Date'].dt.month >= 10)]
df_all.shape


# In[7]:

# Create a INFO_WIN column containing the gain if you bet the good result
df_all['INFO_WIN'] = 0
df_all.loc[df_all.INFO_FTR == 'H', 'INFO_WIN'] = df_all[odd_H]
df_all.loc[df_all.INFO_FTR == 'A', 'INFO_WIN'] = df_all[odd_A]
df_all.loc[df_all.INFO_FTR == 'D', 'INFO_WIN'] = df_all[odd_D]
df_all['INFO_WIN_P'] = 0
df_all.loc[df_all.INFO_FTR == 'H', 'INFO_WIN_P'] = df_all['INFO_PSH']
df_all.loc[df_all.INFO_FTR == 'A', 'INFO_WIN_P'] = df_all['INFO_PSA']
df_all.loc[df_all.INFO_FTR == 'D', 'INFO_WIN_P'] = df_all['INFO_PSD']


# ## Methods

# In[8]:

def get_dataset(league, season, historical_training_year, features):
    # Filter by league
    df = df_all[(df_all['INFO_Div'] == league)]
    # Keep season for test and filter by number of historical season used to train
    date_start_learn = datetime.date(season-historical_training_year, 8, 1)
    date_end_learn = datetime.date(season, 8, 1)
    date_start_test_season = datetime.date(season, 8, 1)
    date_end_test_season = datetime.date(season+1, 8, 1)
    df_test = df[(df['INFO_Date'] > date_start_test_season)]
    df_test = df_test[(df_test['INFO_Date'] < date_end_test_season)]
    df = df[(df['INFO_Date'] > date_start_learn)]
    df = df[(df['INFO_Date'] < date_end_learn)]
    # Filter by feature used to train
    X = pd.get_dummies(df[features])
    y = df[target]
    X_test_season = pd.get_dummies(df_test[features])
    y_test_season = df_test[target]
    # Impute of missing values (NaN) with the mean
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X)
    X = imp.transform(X)
    X_test_season = imp.transform(X_test_season)
    # Standardize features
    sc_X = StandardScaler().fit(X)
    X = sc_X.transform(X)
    X_test_season = sc_X.transform(X_test_season)
    return df, df_test, X, y, X_test_season, y_test_season


# In[9]:

def get_score(y, pred, probs):
    # Compute cross-entropy score
    ll = log_loss(y, probs)
    # Compute accuracy score
    acc = accuracy_score(y, pred)
    # Compute precision score
    prec = precision_score(y, pred, average=None)
    prec_A = prec[0]
    prec_D = prec[1]
    prec_H = prec[2]
    # Compute recall score
    rec = recall_score(y, pred, average=None)
    rec_A = rec[0]
    rec_D = rec[1]
    rec_H = rec[2]
    # Compute F1 score
    f1 = f1_score(y, pred, average=None)
    f1_A = f1[0]
    f1_D = f1[1]
    f1_H = f1[2]
    return {
        'll': ll,
        'acc': acc,
        'prec_A': prec_A,
        'prec_D': prec_D,
        'prec_H': prec_H,
        'rec_A': rec_A,
        'rec_D': rec_D,
        'rec_H': rec_H,
        'f1_A': f1_A,
        'f1_D': f1_D,
        'f1_H': f1_H
    }


# In[10]:

def get_money(df_test, pred_season, prob_season):
    # Join odd and prediction together
    df_test_season = df_test
    df_test_season['probs_A'] = prob_season[:,0]
    df_test_season['probs_D'] = prob_season[:,1]
    df_test_season['probs_H'] = prob_season[:,2]
    df_test_season['probs'] = df_test_season[['probs_A','probs_D','probs_H']].max(axis=1)
    df_test_season['pred'] = pred_season
    df_test_season['WIN'] = -1
    df_test_season.loc[df_test_season.INFO_FTR == df_test_season.pred, 'WIN'] = df_test_season['INFO_WIN']-1
    df_test_season['WIN_P'] = -1
    df_test_season.loc[df_test_season.INFO_FTR == df_test_season.pred, 'WIN_P'] = df_test_season['INFO_WIN_P']-1
    df_test_season['INFO_ODD'] = 0
    df_test_season.loc[df_test_season.pred == 'A', 'INFO_ODD_BET'] = df_test_season[odd_A]
    df_test_season.loc[df_test_season.pred == 'D', 'INFO_ODD_BET'] = df_test_season[odd_D]
    df_test_season.loc[df_test_season.pred == 'H', 'INFO_ODD_BET'] = df_test_season[odd_H]
    df_test_season['prob_less_bet'] = 0
    df_test_season.loc[df_test_season.pred == 'A', 'prob_less_bet'] = df_test_season['probs'] - df_test_season[odd_A].apply(lambda x: 1/x)
    df_test_season.loc[df_test_season.pred == 'D', 'prob_less_bet'] = df_test_season['probs'] - df_test_season[odd_D].apply(lambda x: 1/x)
    df_test_season.loc[df_test_season.pred == 'H', 'prob_less_bet'] = df_test_season['probs'] - df_test_season[odd_H].apply(lambda x: 1/x)
    # calculate money I can get following different scenario
    # Bet on all
    bet_all = df_test_season.WIN.mean()
    # Bet under 1.9
    bet_lte_19 = df_test_season[df_test_season['INFO_ODD_BET'] < 1.9].WIN.mean()
    # Bet under 4
    bet_lte_4 = df_test_season[df_test_season['INFO_ODD_BET'] < 4].WIN.mean()
    # Bet between 1.9 and 4
    bet_btw_19_4 = df_test_season[(df_test_season['INFO_ODD_BET'] > 1.9) & (df_test_season['INFO_ODD_BET'] < 4)].WIN.mean()
    # Bet between 1.9 and 5
    bet_btw_19_5 = df_test_season[(df_test_season['INFO_ODD_BET'] > 1.9) & (df_test_season['INFO_ODD_BET'] < 5)].WIN.mean()
    # Bet between 1.5 and 4
    bet_btw_15_4 = df_test_season[(df_test_season['INFO_ODD_BET'] > 1.5) & (df_test_season['INFO_ODD_BET'] < 4)].WIN.mean()
    # Bet between 1.5 and 5
    bet_btw_15_5 = df_test_season[(df_test_season['INFO_ODD_BET'] > 1.5) & (df_test_season['INFO_ODD_BET'] < 5)].WIN.mean()
    # Bet prob higher than 50%
    bet_pred_gte_50 = df_test_season[df_test_season.probs > 0.5].WIN.mean()
    # Bet prob higher than 60%
    bet_pred_gte_60 = df_test_season[df_test_season.probs > 0.6].WIN.mean()
    # Bet prob higher than 70%
    bet_pred_gte_70 = df_test_season[df_test_season.probs > 0.7].WIN.mean()
    return {
        'bet_all': bet_all,
        'bet_lte_19': bet_lte_19,
        'bet_lte_4': bet_lte_4,
        'bet_btw_19_4': bet_btw_19_4,
        'bet_btw_19_5': bet_btw_19_5,
        'bet_btw_15_4': bet_btw_15_4,
        'bet_btw_15_5': bet_btw_15_5,
    }


# ## Loop on league

# In[ ]:

# Init dataframe
result_df = pd.DataFrame(columns=[
    'league',
    'season',
    'historical_training_year',
    'features',
    'll',
    'acc',
    'prec_A',
    'prec_D',
    'prec_H',
    'rec_A',
    'rec_D',
    'rec_H',
    'f1_A',
    'f1_D',
    'f1_H',
    'bet_all',
    'bet_lte_19',
    'bet_lte_4',
    'bet_btw_19_4',
    'bet_btw_19_5',
    'bet_btw_15_4',
    'bet_btw_15_5'
])


# In[ ]:

for league in league_list:
    for season in season_list:
        for historical_training_year in historical_training_year_list:
            for features in features_list:
                print league,str(season),str(historical_training_year),features[0]
                df, df_test, X, y, X_test_season, y_test_season = get_dataset(league, season, historical_training_year, features[1])
                # train model
                parameters = {
                    'learning_rate': [0.01],
                    'n_estimators': np.arange(100, 600, 30).tolist(),
                    'max_depth': np.arange(3, 9).tolist(),
                    'min_child_weight': np.arange(1, 9).tolist(),
                    'gamma': np.arange(0.01,1,0.03).tolist(),
                    'subsample': np.arange(0.5,1,0.05).tolist(),
                    'colsample_bytree': np.arange(0.5,1,0.05).tolist(),
                    'objective': ['multi:softprob'],
                    'scale_pos_weight': [1],
                    'reg_alpha':np.arange(0,1,0.03).tolist()
                }
                clf = RandomizedSearchCV(
                    estimator=XGBClassifier(nthread=-1, seed=15),
                    param_distributions=parameters,
                    #scoring=make_scorer(log_loss, greater_is_better=False, needs_proba=True),
                    scoring='accuracy',
                    cv=8,
                    n_jobs=-1,
                    verbose=1,
                    n_iter=10)
                clf.fit(X, y)
                # Predict target values
                y_pred = clf.predict(X_test_season)
                # Predict probabilities
                y_probs = clf.predict_proba(X_test_season)
                # get scores
                score_dict = get_score(y_test_season, y_pred, y_probs)
                # get money earned
                money_dict = get_money(df_test, y_pred, y_probs)
                # Add all info to result dataframe
                result_df.loc[len(result_df.index)] = [
                    league,
                    season,
                    historical_training_year,
                    features[0],
                    score_dict['ll'],
                    score_dict['acc'],
                    score_dict['prec_A'],
                    score_dict['prec_D'],
                    score_dict['prec_H'],
                    score_dict['rec_A'],
                    score_dict['rec_D'],
                    score_dict['rec_H'],
                    score_dict['f1_A'],
                    score_dict['f1_D'],
                    score_dict['f1_H'],
                    money_dict['bet_all'],
                    money_dict['bet_lte_19'],
                    money_dict['bet_lte_4'],
                    money_dict['bet_btw_19_4'],
                    money_dict['bet_btw_19_5'],
                    money_dict['bet_btw_15_4'],
                    money_dict['bet_btw_15_5']
                ]


# # Save result

# In[ ]:

result_df.to_csv('./report/XGBoost_FTR_BYLEAGUE_BEST_PARAM_LOG_LOSS.csv')


# ### Final result
# Best is with ??? and ??? years of history

# In[ ]:
