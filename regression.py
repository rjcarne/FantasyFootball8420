import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
import numpy as np
import os
import matplotlib.pyplot as plt
import auxilary


def get_player_df(X, player):
    return X[X['Player'] == player]

def get_position_df(X, position):
    return X[X['Pos'] == position]

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

def get_ridge_regression(X, Y, alphas):
   
    coefs = []
    for a in alphas:
        reg = Ridge(alpha = a, normalize=True)
        reg.fit(X, Y)  
        coefs.append(reg.coef_)
        Y_pred = reg.predict(X)  
        plt.plot(X, Y_pred, label= "Ridge w alpha = " + str(a))
   
    return "Ridge"            # Why return a string? is this just to print?

def get_ridge_regression_weights(X, Y, alphas):
    ridge = Ridge(normalize = True)
    coefs = []
    
    for a in alphas:
        ridge.set_params(alpha = a)
        ridge.fit(X, Y)
        coefs.append(ridge.coef_)
        
    
    
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.plot(alphas, coefs)
    plt.title("Weight of Features as Alpha Increases")
    plt.show()

    for a in alphas:
        X_train, X_test , y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
        ridge = Ridge(alpha = a, normalize = True)
        ridge.fit(X_train, y_train)             
        pred2 = ridge.predict(X_test)           
        print("alpha: ", a)
        print("Ridge Coeffecients")
        print(pd.Series(ridge.coef_, index = X.columns)) 
        print("MSE: " + str(mean_squared_error(y_test, pred2)))          
        print("*******************************")

def get_polynomial_regression(X, Y, color):
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, Y)

    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color=color, label="Polynomial")
    return "Polynomial"

def get_linear_regression(X, Y, color):
    
    reg = LinearRegression()  
    reg.fit(X, Y)  
    Y_pred = reg.predict(X)
    plt.plot(X, Y_pred, color='red', label="Linear")
    return "Linear"

def get_features(Position):
    QB_features = ['FantasyPoints','Year','Age','G','GS', 'Cmp', 'Int', 'Fumbles','FumblesLost','PassingYds','PassingTD','PassingAtt','RushingYds','RushingTD','RushingTD']
    RB_features = ['FantasyPoints','Year','Age','G','GS', 'Fumbles','FumblesLost','RushingYds','RushingTD','RushingTD','Tgt','Rec','ReceivingYds','ReceivingTD']
    WR_features = ['FantasyPoints','Year','Age','G','GS','Fumbles','FumblesLost','Tgt','Rec','ReceivingYds','ReceivingTD']
    TE_features = ['FantasyPoints','Year','Age','G','GS','Fumbles','FumblesLost','Tgt','Rec','ReceivingYds','ReceivingTD']

    if Position == 'QB':
        return QB_features
    elif Position == 'RB':
        return RB_features
    elif Position == 'WR':
        return WR_features
    elif Position == 'TE':
        return TE_features

def get_x_and_y(data, X_var, Y_var, color):    
    features = get_features(data["Pos"][0])
   
    data = data[features]

    X = data[X_var].values.reshape(-1, 1)  
    Y = data[Y_var].values.reshape(-1, 1)  
    plt.scatter(X,Y, color = color)
    return X,Y

# def main():    
#     mydf = get_all_data()
    
#     # Tom Brady Data
#     playerdf = get_player_df(mydf, 'Tom Brady')
#     bradyPoints = []
#     bradyYears = []
#     for value in playerdf['FantasyPoints']:
#         bradyPoints.append(value)
#     for value in playerdf['Year']:
#         bradyYears.append(value)  
        
        
#     # Peyton Manning Data
#     playerdf = get_player_df(mydf, 'Peyton Manning')
#     manningPoints = []
#     manningYears = []
#     for value in playerdf['FantasyPoints']:
#         manningPoints.append(value)
#     for value in playerdf['Year']:
#         manningYears.append(value)
        
#     # Matt Cassel
#     playerdf = get_player_df(mydf, 'Matt Cassel')
#     casselPoints = []
#     casselYears = []
#     for value in playerdf['FantasyPoints']:
#         casselPoints.append(value)
    # for value in playerdf['Year']:
    #     casselYears.append(value)
    
    # # Visualizations
    # plt.plot(bradyYears, bradyPoints, 'o', color='red')
    # plt.plot(manningYears, manningPoints, 'x', color='blue')
    # plt.plot(casselYears, casselPoints, '+', color='green')
    # plt.title("QB Fantasy Points by Year")
    # plt.xlabel("Years") 
    # plt.ylabel("Points")
#     plt.show()
    

def testmain():

    X = get_all_data()
    
    data = get_player_df(X, "Tom Brady")
    # X = X[X["Pos"] == "QB"]
    # data = X[X["Age"] <= 28]

    X_var = "Year"
    Y_var = "FantasyPoints"
    features = get_features(data["Pos"][0])


    X, Y = get_x_and_y(data, X_var, Y_var, "blue")
    # X = data[features]
    # Y = data["FantasyPoints"]
    # X = X.drop(columns='FantasyPoints')
    alphas = [10, 1000000]

    # alphas = 10**np.linspace(10,-2,100)*0.5
    reg_type = get_ridge_regression(X, Y, alphas)
    reg_type = get_linear_regression(X, Y, "red")
    
    # get_ridge_regression_weights(X, Y, alphas)

    reg_type = get_polynomial_regression(X, Y, "blue")

    # We may want to set the boundaries and tick marks of the plots so that the years are not half years
    
    plt.title("Expected " + Y_var + " based on " + X_var)
    plt.xlabel(X_var)
    plt.ylabel(Y_var)
    plt.legend()
    plt.show()
    

# testmain()
# main()


def featureSpecificLinearReg(maindf, playerName):
    reg = LinearRegression()                                # Regression method
    playerdf = get_player_df(maindf, playerName)            # Get player Dataframe
    playerFeatures = get_features(playerdf["Pos"][0])       # Get player features
    predArray = np.zeros((1, len(playerFeatures)))          # create numpy array to return

    test_val = playerdf[playerFeatures].iloc[-1]["FantasyPoints"]
    playerdf = playerdf[playerdf["Year"] < 2019]

    if len(playerdf.columns) >= 5:
        minAge = playerdf.iloc[0]["Age"]
        maxAge = playerdf.iloc[-1]["Age"]

        ageRange = maxAge - minAge

        desiredMinAge = minAge + (ageRange * 0.2)

        playerdf = playerdf[playerdf["Age"] >= desiredMinAge]


    print("Player Statistics")
    print(playerdf)
    
    index = 0
    for feature in playerFeatures:
        featureList = []
        yearList = []
        for rowCounter, row in playerdf.iterrows():
            featureList.append(row[feature])
            yearList.append(int(row["Year"]))
        result = stats.linregress(yearList, featureList)        # Linear Regression
        predArray[0][index] = result[0] * 2019 + result[1]    # y = mx+b
        index = index + 1
    print("Predicted Stats: ")
    print(predArray)
    print("************")
    print("Predicted Fantasy Points Next Year")
    prediction = auxilary.predict_fantasy_output(playerdf[playerFeatures], playerdf["FantasyPoints"], predArray, 0.005)
    print(prediction)
    print("************")

    print("Error Calculations")
    difference = test_val - prediction
    # print(len(test_row.to_numpy()))
    # print(len(predArray))
    print("Difference with 2019 data: ", difference)

featureSpecificLinearReg(get_all_data(), "Tom Brady")

