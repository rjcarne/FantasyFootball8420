import os
import numpy as np
import csv




def append_data(X, filepath, year):
    with open(filepath,'r') as dest_f:
        data_iter = csv.reader(dest_f,
                            delimiter = ',',
                            quotechar = '"')
        data = [data for data in data_iter]
    data_array = np.asarray(data, dtype = None)
    data_array = np.delete(data_array,0,1)
    if year <= 1991:
        PLAYER_NUM = data_array.shape[0]
        new_col = np.empty((PLAYER_NUM,1))
        for current_player in range(1,PLAYER_NUM):
            new_col[current_player] = data_array[current_player][12]
        data_array = np.hstack((data_array[:,:12], new_col, data_array[:,12:]))
        data_array = np.delete(data_array, 0, axis=0)
        
    X = np.append(X, values=data_array,axis=0)
    return X

X = np.empty((0,27))


yearly_directory = 'Fantasy_Data/yearly/'
for year in range(1970,2020):
    filepath = yearly_directory + str(year) + '.csv'
    X = append_data(X, filepath, year)

with open(yearly_directory + '2019.csv','r') as dest_f:
        data_iter = csv.reader(dest_f,
                            delimiter = ',',
                            quotechar = '"')
        data = [data for data in data_iter]
        data_array = np.asarray(data, dtype = None)
        X_labels = data_array[0]

print(X.shape)