import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import os
import numpy as np
import pandas as pd
import random

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


"""
Loads Data from Yahoo Finance
	ticker (str): the ticker you want to load, examples include AAPL, TESL, etc.
	n_steps (int): the historical sequence length (i.e window size) used to predict, default is 25
	scale (bool): whether to scale prices from 0 to 1, default is True
	shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
	predict_step (int): the future lookup step to predict, default is 1 (e.g next day)
	split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
			to False will split datasets in a random way
	test size(float) ratio of test to train
"""
def loadData(ticker, n_steps=25, scale=True, predict_step=1, split_by_date=True, test_size=0.2):

	feature_columns=['adjclose', 'volume', 'open', 'high', 'low']
	#Use all features from yahooFinance

	#Load from yahoo finance
	df = si.get_data(ticker)

	result = {}
	result['df'] = df.copy()

	#add date as a column
	if "date" not in df.columns:
		df["date"] = df.index

	#Scale Data
	if scale:
		column_scaler= {}

		#scale from 0to1
		for column in feature_columns:
			scaler = preprocessing.MinMaxScaler()
			df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
			column_scaler[column] = scaler
		#add to results
		result["column_scaler"]= column_scaler

	#add target column by shifting my predict step
	df['future']=df['adjclose'].shift(-predict_step)

	#last lookup contains nan in future column
	# get before dropping NaNs
	last_sequence = np.array(df[feature_columns].tail(predict_step))

	df.dropna(inplace=True)
	sequence_data = []
	sequences = deque(maxlen=n_steps)
	for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
		sequences.append(entry)
		if len(sequences) == n_steps:
			sequence_data.append([np.array(sequences), target])

	#last sequence get by appending last nstep with predictstep
	# for instance, if n_steps=50 and predict_step=10, last_sequence should be of 60 (that is 50+10) length
	# this last_sequence will be used to predict future stock prices that are not available in the dataset
	last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
	last_sequence = np.array(last_sequence).astype(np.float32)


	result['last_sequence'] = last_sequence


  	#form NN data
	X, y = [], []

	for seq, target in sequence_data:
		X.append(seq)
		y.append(target)

	X=np.array(X)
	y=np.array(y)

	if split_by_date:
		# split the dataset into training & testing sets by date (not randomly splitting)
		train_samples = int((1 - test_size) * len(X))
		result["X_train"] = X[:train_samples]
		result["y_train"] = y[:train_samples]
		result["X_test"]  = X[train_samples:]
		result["y_test"]  = y[train_samples:]
	else:	
		# split the dataset randomly
		result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=False)

	#get tesat dates
	dates = result["X_test"][:, -1, -1]

	#results from test tades
	result["test_df"]= result["df"].loc[dates]

	# Kill Dupes
	result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

	#remove dates from train test
	result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
	result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

	return result


if __name__=="__main__":
	ticker="GME"
	data = loadData(ticker)
	print(data)