# Replace line 20 with the ticker
# Replace line 21 with the number of days you want to predict 
# Select desired ml algorithm from lines 64-68

import pandas as pd # for dataframes
import pandas_datareader.data as pdr # data reader to create dataframe from yahoo finance
import numpy as np # for arrays
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split # to split training/learning data and testing data
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import datetime # for dates
import matplotlib.pyplot as plt # for graphing
from matplotlib import style

plt.style.use('bmh')
ticker = 'TSLA' # Takes any yahoo finance tickers
prediction_out = 300 # Num of days you want to predict



# GET DATA
# Create dataframe with stock data Jan 1, 2000  -  present
START = datetime.datetime(2000, 1, 1)
END = datetime.datetime.now()
df = pdr.DataReader( 
    name=ticker, 
    data_source='yahoo', 
    start=START, 
    end=END 
    )
df.to_csv('{}.csv'.format(ticker))
df = pd.read_csv('{}.csv'.format(ticker), parse_dates=True, index_col=0) # save to csv, read 
df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']] # keep these features of dataframe
df['HL_PCT'] = (df['High']-df['Low'])/df['Low']*100.0  # new col for high - low percent: High - Low / Low, X 100
df['PCT_change'] = (df['Adj Close']-df['Open'])/df['Open']*100.0 # new col for daily percent change: New - Old / Old X 100
df = df[['Open', 'High', 'Low', 'Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
prediction_col = 'Adj Close' # only going to predict adj close col
df['Adj Close'].plot(color = 'green') # plot adj close in green, prediction will be diff colour
df.fillna(0) # Replace all NaN elements with 0s, will just be treated as outliers
df['label'] = df[prediction_col].shift(-prediction_out) # new col (the prediction). Set to adj close col but shifted __ days into the future. Num days determined by prediction_out 



# DEFINE FEATURES & LABELS
# features = X, label = y
# X = all cols in dataframe except 'label'. turn to npmy array
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X) # Preprocessing
X_lately = X[-prediction_out:] # X_lately contains the most recent features, which we're going to predict against. Syntax [start : end : step] => return a new list containing the requested elements
X = X[:-prediction_out:] # Set X equal to X to the point of -prediction_out. [:-prediction_out:] means from index 0 to -prediction_out
df.dropna(inplace=True) # Drop any still NaN's still from the dataframe
# y = 'label' col in dataframe is label. turn to numpy array
y = np.array(df['label'])
X_train , X_test , y_train , y_test = train_test_split(X, y, train_size=0.9, test_size = 0.1) # Split above^ to get training set of features and labels & testing set of features and labels



# SPOT-CHECK ALGORITHMS
# clf = LinearRegression(n_jobs=-1)
clf = SVR(kernel='rbf', C=1e3, gamma=0.1) # Defining the model
#clf = RandomForestRegressor(n_estimators=100)
#clf = GradientBoostingRegressor(n_estimators=200)
#clf = DecisionTreeRegressor() 
clf.fit(X_train, y_train) # Fit to model
accuracy = clf.score(X_test, y_test) # predictor.score(X,Y) internally calculates Y'=predictor.predict(X) and then compares Y' against Y to give an accuracy measure



# PREDICT
prediction_set = clf.predict(X_lately) # predict against X_lately
print(prediction_set) # the array of predictions (The Adj Close of the next __ days)
print(accuracy) 
print(prediction_out) # num of days to predict



# GRAPH
df['Prediction'] = np.nan # new col, NaN, change later
last_date = df.iloc[-1].name # Need to get the last day in the DataFrame and begin assigning each new prediction to a new day, this is the very last date, get name of it
last_unix = last_date.timestamp() # number of seconds that have passed since January 1, 1970
next_unix = last_unix + 86400 # next_unix is next day. 86400 is seconds in 1 day. Now have next day we wish to use
for i in prediction_set: # Add the prediction to the existing DataFrame
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix = next_unix + 86400 #next_unix set to itself plus the value of 1 day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
# Iterate through prediction_set, take each prediction and day and then set those as the values in the DataFrame
# Last line sets NaN for all of the columns except the last column, sets the last column to [i] which is the prediction for that day
df['Prediction'].plot(color = 'red') # plot 'Prediction' column in red so user can see difference real Adj Close and predicted Adj Close
plt.legend(loc=2) # Set legend to position 2 (top-left)
plt.title('{}'.format(ticker)) # Title is the ticker
plt.xlabel('Date') # X axis label is Date
plt.ylabel('Adj Close Price (USD)') # Y axis label is Adj Close Price
plt.show() # Show plot