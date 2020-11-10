import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import os
import matplotlib.pyplot as plt


def get_player_df(X, player):
    return X[X['Player'] == player]


def get_all_data():
    indeces = []
    X = pd.DataFrame()
    for year in range(1992, 2020):
        filename = 'Fantasy_Data/yearly/' + str(year) + '.csv'
        A = pd.read_csv(filename)
        A['Year'] = year
        for i in range(0, A.shape[0]):
            indeces.append(A.at[i,'Player'] + str(year))
        X = pd.concat([X,A])
    indeces = pd.Series(indeces)
    X = X.set_index(indeces)

    X = X.drop(columns=['Unnamed: 0'])
    return X

def get_year_data(year):
    indeces = []
    X = pd.DataFrame()
    filepath = "Fantasy_Data/weekly/" + str(year)

    for fileindex in range(0,len(os.listdir(filepath))):
        filename = filepath + '/week' + str(fileindex + 1) + '.csv'

        A = pd.read_csv(filename)
        A['Week'] = fileindex + 1
        for i in range(0, A.shape[0]):
            indeces.append(A.at[i,'Player'] + str(fileindex + 1))
        X = pd.concat([X,A])

    indeces = pd.Series(indeces)
    X = X.set_index(indeces)


    return X


def get_ridge_regression(X, Variable, Position):
    QB_features = ['Age','G','GS', 'Cmp', 'Int', 'Fumbles','FumblesLost','PassingYds','PassingTD','PassingAtt','RushingYds','RushingTD','RushingTD']
    RB_features = ['Age','G','GS', 'Fumbles','FumblesLost','RushingYds','RushingTD','RushingTD','Tgt','Rec','ReceivingYds','ReceivingTD']
    WR_features = ['Age','G','GS','Fumbles','FumblesLost','Tgt','Rec','ReceivingYds','ReceivingTD']
    TE_features = ['Age','G','GS','Fumbles','FumblesLost','Tgt','Rec','ReceivingYds','ReceivingTD']
    y = X["FantasyPoints"]

    if Position == 'QB':
        features = QB_features
    elif Position == 'RB':
        features = RB_features
    elif Position == 'WR':
        features = WR_features
    elif Position == 'TE':
        features = TE_features

    X = X[features]

    alphas = [0,20,50,100,200]
    
    cs = ['red', 'green', 'blue', 'orange','black','yellow','purple','cyan','grey','navy']
    coef = []
    for a, c in zip(alphas, cs):
        rr = Ridge(alpha=a, fit_intercept=False)
        rr.fit(X, y)
        model = rr.predict(X)
        plt.plot(sorted(X.iloc[:, X.columns.get_loc(Variable)]), model[np.argsort(X.iloc[:, 0])], c, label='Alpha: {}'.format(a))
        coef.append(rr.coef_)
    plt.title("Expected Fantasy Points based on " + Variable + " using Ridge Regression")
    plt.xlabel(Variable)
    plt.ylabel("Expected Fantasy Points")
    plt.legend()
    plt.show()

def get_polynomial_regression(X, Variable, Position):
    QB_features = ['Age','G','GS', 'Cmp', 'Int', 'Fumbles','FumblesLost','PassingYds','PassingTD','PassingAtt','RushingYds','RushingTD','RushingTD']
    RB_features = ['Age','G','GS', 'Fumbles','FumblesLost','RushingYds','RushingTD','RushingTD','Tgt','Rec','ReceivingYds','ReceivingTD']
    WR_features = ['Age','G','GS','Fumbles','FumblesLost','Tgt','Rec','ReceivingYds','ReceivingTD']
    TE_features = ['Age','G','GS','Fumbles','FumblesLost','Tgt','Rec','ReceivingYds','ReceivingTD']
    y = X["FantasyPoints"]

    if Position == 'QB':
        features = QB_features
    elif Position == 'RB':
        features = RB_features
    elif Position == 'WR':
        features = WR_features
    elif Position == 'TE':
        features = TE_features

    X = X[features]



def get_average(X):
    pass

X = get_all_data()

Y = get_player_df(X, "Tom Brady")
# Y = X[X["Age"] >= 28]


# Y = X[X["Player"] == 'Deshaun Watson']
get_ridge_regression(Y, "PassingYds", "QB")
# get_linear_regression(Y)
# coeff = get_ridge_regression(Y)
# plt.plot(coeff)
# plt.show()
# get_linear_regression1(Y)
