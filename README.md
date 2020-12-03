Data included in Fantasy_Data folder

Most functionalites are put into functions, there is a simple function call at the bottom that you can change to get results for different players. Simply replacing "Tom Brady" with a different player will perform their calculations

****** Function Descriptions ******

* Gets dataframe with player's career stats *
get_player_df(X, player):

* Gets dataframe with all players of a certain position *
get_position_df(X, position):

* Loads in all data from dataset into one dataframe *
get_all_data():

* Gets dataframe from a certain year *
get_year_data(year):

* Performs ridge regression and plots *
get_ridge_regression(X, Y, alphas):

* Performs ridge regression using many alphas, plots the weights as alpha changes and shows MSE for each alpha *
get_ridge_regression_weights(X, Y, alphas):

* Performs polynomial regression and plots *
get_polynomial_regression(X, Y, color):

* Performs linear regression and plots *
get_linear_regression(X, Y, color):
    
* Returns list of features relevant to a certain position in football *
get_features(Position):

* Gets X and Y based on desired input features and plots *
get_x_and_y(data, X_var, Y_var, color):    

* Performs linear regression for all stats, uses those stats in ridge regression for next season's projection *
predict_fantasy_output(X, Y, X_test, alpha):

* Creates graph of linear, polynomial and ridge regression using only one feature and plots *
regressionGraph():

* Performs linear regression for all features *
featureSpecificLinearReg(maindf, playerName):

* Performs all calculations based on a players name
def get_info(player):

