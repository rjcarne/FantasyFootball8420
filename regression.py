import pandas as pd
import os


def get_player_df(X, player):
    return X[X['Player'] == player]


def get_all_data():
    indeces = []
    X = pd.DataFrame()
    for year in range(1992, 2020):
        filename = 'Fantasy_Data/yearly/' + str(year) + '.csv'
        A = pd.read_csv(filename)
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
        for i in range(0, A.shape[0]):
            indeces.append(A.at[i,'Player'] + str(fileindex + 1))
        X = pd.concat([X,A])

    indeces = pd.Series(indeces)
    X = X.set_index(indeces)


    return X

