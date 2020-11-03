import os
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
from sklearn.metrics import silhouette_score
import pandas as pd
from plotnine import *


def append_data(X, filepath, year):
    # loads file
    with open(filepath,'r') as dest_f:
        data_iter = csv.reader(dest_f,
                            delimiter = ',',
                            quotechar = '"')
        data = [data for data in data_iter]
    data_array = np.asarray(data, dtype = None)

    # remove column for ranking within year
    data_array = np.delete(data_array,0,1)

    # number of players + 1 because of title row
    PLAYER_NUM = data_array.shape[0]

    # create feature for key
    key_col = np.empty((PLAYER_NUM,1))
    key_col = key_col.astype('str') 
    for i in range(1,PLAYER_NUM):
        key_col[i] = data_array[i][0] + str(year)
    key_col[0] = "Key"
    data_array = np.hstack((data_array[:,:0], key_col, data_array[:,0:]))
    
    # 1991 and under didn't include targets for receptions. 
    if year <= 1991:
        PLAYER_NUM = data_array.shape[0]
        new_col = np.empty((PLAYER_NUM,1))
        for current_player in range(1,PLAYER_NUM):
            new_col[current_player] = data_array[current_player][12]
        data_array = np.hstack((data_array[:,:12], new_col, data_array[:,12:]))
    data_array = np.delete(data_array, 0, axis=0)
    
    # add year to current data
    X = np.append(X, values=data_array,axis=0)
    return X

def get_labels():
    with open(filepath,'r') as dest_f:
        data_iter = csv.reader(dest_f,
                            delimiter = ',',
                            quotechar = '"')
        data = [data for data in data_iter]
    data_array = np.asarray(data, dtype = None)
    data_array = np.delete(data_array,0,1)
    PLAYER_NUM = data_array.shape[0]
    key_col = np.empty((PLAYER_NUM,1))
    key_col = key_col.astype('str') 
    for i in range(1,PLAYER_NUM):
        key_col[i] = data_array[i][0] + str(year)
    key_col[0] = "Key"
    data_array = np.hstack((data_array[:,:0], key_col, data_array[:,0:]))
    X_labels = data_array[0]
    return X_labels

def get_index(list, value):
    for i in range(0,len(list)):
        if value == list[i]:
            return i
    return -1

def get_data(X, column):
    teams = np.unique(X[:,0])
    
    positions = np.unique(X[:,1])
    X_data = np.empty(X.shape)
    NUM_PLAYERS = X.shape[0]


    for i in range(NUM_PLAYERS):
        X_data[i][0] = get_index(teams, X[i][0])
        X_data[i][1] = get_index(positions, X[i][1])
        for j in range(column,X.shape[1]):
            X_data[i][j] = X[i][j]

    return X_data, teams, positions
   


# X = all yearly data
X = np.empty((0,28))


yearly_directory = 'Fantasy_Data/yearly/'
for year in range(1970,2020):
    filepath = yearly_directory + str(year) + '.csv'
    X = append_data(X, filepath, year)
    
X_labels = get_labels()

# remove player name feature
X = np.delete(X,1,1)

# remove player name label
X_labels = np.delete(X_labels,1,0)

# get keys and delete key column from X and labels
keys = X[:,0]
X = np.delete(X,0,1)
X_labels = np.delete(X_labels,0,0)

X_data, teams, positions = get_data(X, 2)
# print(X_data[0])
# print(keys)
# print(X_labels)

A = pd.DataFrame(data=X_data, index= keys, columns=X_labels)

QB_features = ['Tm','Age','G','GS','PassingYds','PassingTD','PassingAtt','Int','Cmp','Att','RushingYds','RushingTD','RushingAtt', 'Fumbles','FumblesLost','FantasyPoints']
QB = A[A['Pos'] == 1]
QB = QB[QB_features]

RB_features = ['Tm','Age','G','GS', 'RushingYds','RushingTD','RushingAtt', 'Tgt','Rec','ReceivingYds','Y/R','ReceivingTD', 'Fumbles','FumblesLost','FantasyPoints']
RB = A[A['Pos'] == 2]
RB = A[RB_features]

WR_features = ['Tm','Age','G','GS','Tgt','Rec','ReceivingYds','Y/R','ReceivingTD', 'Fumbles','FumblesLost','FantasyPoints']
WR = A[A['Pos'] == 3]
WR = A[WR_features]

TE_features = ['Tm','Age','G','GS','Tgt','Rec','ReceivingYds','Y/R','ReceivingTD', 'Fumbles','FumblesLost','FantasyPoints']
TE = A[A['Pos'] == 4]
TE = A[TE_features]

QB = QB[QB['GS'] >= 8]

z = StandardScaler()

QB[QB_features] = z.fit_transform(QB)

EM = mixture.GaussianMixture(n_components = 3)

EM.fit(QB)

cluster = EM.predict(QB)

QB['cluster'] = cluster

g = ggplot(QB, aes(x = 'FantasyPoints', y= 'PassingYds', color = 'cluster')) + geom_point()

file_name = "test"
ggsave(plot = g, filename = file_name)