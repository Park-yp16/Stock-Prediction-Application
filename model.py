import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import pytz
import os
import numpy as np
import pandas as pd
import random
import os
import time
from datetime import date, datetime
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt

import sqlite3 as sql
import database


np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

def shuffle_in_unison(a, b):
		# shuffle two arrays in the same way
		state = np.random.get_state()
		np.random.shuffle(a)
		np.random.set_state(state)
		np.random.shuffle(b)


#class for the model
class MyModel: 
	


	def load_data(self, ticker, n_steps=50, lookup_step=1, split_by_date=True,
					test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
		"""
		Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
		Params:
			ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
			n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
			lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
			split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
				to False will split datasets in a random way
			test_size (float): ratio for test data, default is 0.2 (20% testing data)
			feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
		"""
		# see if ticker is already a loaded stock from yahoo finance
		if isinstance(ticker, str):
			# load it from yahoo_fin library
			df = si.get_data(ticker)
		elif isinstance(ticker, pd.DataFrame):
			# already loaded, use it directly
			df = ticker
		else:
			raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
		# this will contain all the elements we want to return from this function
		result = {}
		# we will also return the original dataframe itself
		result['df'] = df.copy()
		# make sure that the passed feature_columns exist in the dataframe
		for col in feature_columns:
			assert col in df.columns, f"'{col}' does not exist in the dataframe."
		# add date as a column
		if "date" not in df.columns:
			df["date"] = df.index
		column_scaler = {}
		self.current = df['adjclose'].iloc[-1]
		print(self.current)
		# scale the data (prices) from 0 to 1
		for column in feature_columns:
			scaler = preprocessing.MinMaxScaler()
			df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
			column_scaler[column] = scaler
			# add the MinMaxScaler instances to the result returned
		result["column_scaler"] = column_scaler
		# add the target column (label) by shifting by `lookup_step`
		df['future'] = df['adjclose'].shift(-lookup_step)
		# last `lookup_step` columns contains NaN in future column
		# get them before droping NaNs
		last_sequence = np.array(df[feature_columns].tail(lookup_step))
		# drop NaNs
		df.dropna(inplace=True)
		sequence_data = []
		sequences = deque(maxlen=n_steps)
		for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
			sequences.append(entry)
			if len(sequences) == n_steps:
				sequence_data.append([np.array(sequences), target])
		# get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
		# for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
		# this last_sequence will be used to predict future stock prices that are not available in the dataset
		last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
		last_sequence = np.array(last_sequence).astype(np.float32)
		# add to result
		result['last_sequence'] = last_sequence
		# construct the X's and y's
		X, y = [], []
		for seq, target in sequence_data:
			X.append(seq)
			y.append(target)
		# convert to numpy arrays
		X = np.array(X)
		y = np.array(y)
		if split_by_date:
			# split the dataset into training & testing sets by date (not randomly splitting)
			train_samples = int((1 - test_size) * len(X))
			result["X_train"] = X[:train_samples]
			result["y_train"] = y[:train_samples]
			result["X_test"]  = X[train_samples:]
			result["y_test"]  = y[train_samples:]

		# shuffle the datasets for training 
			shuffle_in_unison(result["X_train"], result["y_train"])
			shuffle_in_unison(result["X_test"], result["y_test"])
		else:	
			# split the dataset randomly
			result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size)
		# get the list of test set dates
		dates = result["X_test"][:, -1, -1]
		# retrieve test features from the original dataframe
		result["test_df"] = result["df"].loc[dates]
		# remove duplicated dates in the testing dataframe
		result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
		# remove dates from the training/testing sets & convert to float32
		result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
		result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
		return result

	#Create LSTM Model for prediction
	def create_model(self, sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop"):
		self.model = Sequential()
		for i in range(n_layers):
			if i ==0:
				#first
				 self.model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
			elif i == n_layers - 1:
				#last
				 self.model.add(cell(units, return_sequences=False))
			else:
				 self.model.add(cell(units, return_sequences=True))
			#add dropout
		self.model.add(Dropout(dropout))
		self.model.add(Dense(1, activation="linear"))
		self.model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
		return  self.model

	#Define Parameters to set up stock
	def define_model(self, lookup_step, stock, n_steps=50, test_size=0.2, n_layers = 2, units=256, dropout=0.4, epochs=50):

		#Number of days used for each prediction
		 self.N_STEPS = n_steps

		# days to look into future ( 1 is next day) 
		 self.LOOKUP_STEP = lookup_step

		#Split of train/test
		 TEST_SIZE = test_size

		#Features to use
		 FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

		#current date
		 tz = pytz.timezone("US/Eastern")
		 date_now = datetime.now(tz).date()

		##model params
		 N_LAYERS = n_layers


		#want LSTM model
		 CELL = LSTM

		#Nueron count
		 UNITS = units

		#Dropout rate, default 40%
		 DROPOUT= dropout

		#Training params
		# Mean absolute error loss
		# LOSS = "mae"
		#Huber loss
		 LOSS = "huber_loss"
		 self.LOSS = LOSS
		 OPTIMIZER = "adam"

		 BATCH_SIZE = 64

		 EPOCHS = epochs

		 self.ticker = stock
		 self.epochs = EPOCHS
		 self.date = date_now
		#model name to save
		 ticker_d_filename = os.path.join("data", f"{self.ticker}_{date_now}_{self.LOOKUP_STEP}days.csv")
		 self.model_name = f"{date_now}_{self.ticker}_{self.LOOKUP_STEP}days"

		 # At 6 months and longer a larger range of test data becomes more and more important than just simply using hte end of the tail
		 split_by_date = True
		 if self.LOOKUP_STEP > 179:
		 	split_by_date = False

		 self.data = self.load_data(self.ticker, self.N_STEPS, lookup_step= self.LOOKUP_STEP, test_size= TEST_SIZE, feature_columns=FEATURE_COLUMNS, split_by_date=split_by_date)


		 #save dataframe
		 self.data["df"].to_csv(ticker_d_filename)

		 #make model
		 self.model = self.create_model(self.N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER)

		 #savepoints
		 checkpointer = ModelCheckpoint(os.path.join("results", self.model_name + ".h5"), save_weights_only = True,save_best_only=True, verbose=1)
		 tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))

		 #train the model and save weights as we see 
		 # a new optimal model
		 history = self.model.fit(self.data["X_train"], self.data["y_train"],
					batch_size=BATCH_SIZE,
					epochs=EPOCHS,
					validation_data=(self.data["X_test"], self.data["y_test"]),
					callbacks=[checkpointer, tensorboard],
					verbose=1)


	#plolt true close price (future) along with predicted close (future) price in blue and red respectfully
	def plot_graph(self, test_df):
		plt.figure()
		plt.plot(test_df[f'true_adjclose_{self.LOOKUP_STEP}'], c='b')
		plt.plot(test_df[f'adjclose_{self.LOOKUP_STEP}'], c='r')
		plt.xlabel("Date of Prediction")
		plt.ylabel("Future Price")
		plt.title(self.ticker + " {} day predictions".format(self.LOOKUP_STEP))
		plt.legend(["Actual Future Price", "Predicted Future Price"])
		#plt.show()
		plt.savefig("static/docs/upload/plots/{}_{}days_{}.png".format(self.ticker, self.LOOKUP_STEP, self.date), dpi=300)
		#plt.close()
		return plt


	#takes mode and data dict to consctuct
	# a final df that includes features with 
	# true and predicted prices of dataset

	def get_final_dataframe(self):

		#if predicted future price is higher than current
		# then calculate the true future price - curent price to get buy profit
		buy_profit = lambda current, true_f, pred_f: true_f - current if pred_f	> current else 0

		# if pred future price is lower than current price
		# then subtract the true future price from current price
		sell_profit = lambda current, true_f, pred_f: current- true_f if pred_f < current else 0

		X_test = self.data["X_test"]
		y_test = self.data["y_test"]

		#perform prediction and get prices
		y_pred = self.model.predict(X_test)

		#adjust for scale
		y_test = np.squeeze(self.data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
		y_pred = np.squeeze(self.data["column_scaler"]["adjclose"].inverse_transform(y_pred))

		test_df = self.data["test_df"]

		# add pred future prices to the dataframe
		test_df[f"adjclose_{self.LOOKUP_STEP}"] = y_pred

		#add true to dataprice
		test_df[f"true_adjclose_{self.LOOKUP_STEP}"] = y_test



		#sort the df by date
		test_df.sort_index(inplace=True)
		final_dataframe = test_df
		#add the buy profit column
		final_dataframe["buy_profit"] = list(map(buy_profit, 
									final_dataframe["adjclose"], 
									final_dataframe[f"adjclose_{self.LOOKUP_STEP}"], 
									final_dataframe[f"true_adjclose_{self.LOOKUP_STEP}"])
									)

		#add sell profit column
		final_dataframe["sell_profit"] = list(map(sell_profit, 
									final_dataframe["adjclose"], 
									final_dataframe[f"adjclose_{self.LOOKUP_STEP}"], 
									final_dataframe[f"true_adjclose_{self.LOOKUP_STEP}"])


									)
		return final_dataframe


	def predict(self):

		#get last sequence from data
		last_sequence = self.data["last_sequence"][-self.N_STEPS:]

		#expand dimension
		last_sequence = np.expand_dims(last_sequence, axis=0)

		#get prediction scaled from 0 to 1
		pred = self.model.predict(last_sequence)

		pred_price = self.data["column_scaler"]["adjclose"].inverse_transform(pred)[0][0]
		self.prediction = pred_price
		return pred_price



	def load_model(self, path=""):
		if path == "":
			model_path = os.path.join("results", self.model_name + ".h5")
		else:
			model_path=path
		self.model.load_weights(model_path)

	def evaluate_model(self):

		loss, mae = self.model.evaluate(self.data["X_test"], self.data["y_test"], verbose = 0)

		self.mean_absolute_error = self.data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]


		#get final df for testing
		final_df = self.get_final_dataframe()

		#predict future price
		future_price = self.predict()
		#self.prediction= future_price


		#calculate accuracy by counting number of positive profits
		accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)

		# calculating total buy & sell profit
		total_buy_profit  = final_df["buy_profit"].sum()
		total_sell_profit = final_df["sell_profit"].sum()

		total_profit = total_buy_profit + total_sell_profit


		# dividing total profit by number of testing samples (number of trades)
		profit_per_trade = total_profit / len(final_df)

		self.loss = loss
		self.accuracy_score = accuracy_score
		self.total_buy_profit = total_buy_profit
		self.total_sell_profit = total_sell_profit
		self.total_profit = total_profit
		self.future_price = future_price
		self.profit_per_trade = profit_per_trade

		self.plot = self.plot_graph(final_df)

	def getFuturePrice(self):
		return self.future_price
	#Create from scratch and display
	def __init__(self, lookup_step, stock, n_steps=50, test_size=0.2, n_layers = 2, units=256, dropout=0.3, epochs=50):


		if not os.path.isdir("results"):
			os.mkdir("results")
		if not os.path.isdir("logs"):
			os.mkdir("logs")
		if not os.path.isdir("data"):
			os.mkdir("data")
		if not os.path.isdir("static/docs/upload/plots"):
			os.mkdir("static/docs/upload/plots")


		self.define_model(lookup_step, stock, n_steps=n_steps, test_size=test_size, n_layers=n_layers, units=units, dropout=dropout, epochs=epochs)
		self.future_price = 0
		self.load_model()

		self.evaluate_model()

		print(f"Ticker: {self.ticker}")
		print(f"Future price after {self.LOOKUP_STEP} days is {self.future_price:.2f}$")
		print(f"{self.LOSS} loss:", self.loss)
		print("Mean Absolute Error:", self.mean_absolute_error)
		print("Accuracy score:", self.accuracy_score)
		print("Total buy profit:", self.total_buy_profit)
		print("Total sell profit:", self.total_sell_profit)
		print("Total profit:", self.total_profit)
		print("Profit per trade:", self.profit_per_trade)

		#add to db
		try:
			with sql.connect("database.db") as con:
				cur = con.cursor()

				#insert
				sss = ("INSERT INTO models (lookup_step, stock, n_steps, updateDate, modelName) VALUES (?,?,?,?,?);")

				ss = (int(self.LOOKUP_STEP), self.ticker, int(self.N_STEPS), self.date.strftime("%Y-%d-%m"), self.model_name)

				cur.execute(sss, ss)
				print("Successfully inserted model into database")
				con.commit()
		except:
			print("Something went wrong attempting to insert model into database")
		finally:
			return
		#self.plot.show()

	#Have saved model
	@classmethod
	def fromModel(self, model_name, loss = "huber_loss", n_steps = 50, feature_columns = ["adjclose", "volume", "open", "high", "low"], units = 256, dropout = 0.4, epochs=50):

		#create self
		self = self.__new__(self)
		
		#get model name format
		splitModelName = model_name.split("_")
		print(splitModelName)
		date_file = splitModelName[0] 
		stock = splitModelName[1]
		s_lookup_step = splitModelName[2].split("d")[0]
		lookup_step = int(s_lookup_step)
		self.LOSS = loss
		self.model_name = model_name
		FEATURE_COLUMNS = feature_columns
		#print(date)
		print(stock)
		print(lookup_step)

		#these should exist but jsut in case
		if not os.path.isdir("results"):
			os.mkdir("results")
		if not os.path.isdir("logs"):
			os.mkdir("logs")
		if not os.path.isdir("data"):
			os.mkdir("data")
		if not os.path.isdir("static/docs/upload/plots"):
			os.mkdir("static/docs/upload/plots")

		print(len(FEATURE_COLUMNS))
		#build needed variables
		self.N_STEPS = n_steps 
		self.LOOKUP_STEP = lookup_step
		self.ticker = stock
		self.date = datetime.strptime(date_file, "%Y-%m-%d").date()
		self.model = self.create_model(sequence_length=n_steps, n_features=5, loss = loss, units = units)

		#load model from save
		self.load_model(path=os.path.join("results", self.model_name + ".h5"))

		#update date for new data and updated info
		tz = pytz.timezone("US/Eastern")
		self.date = datetime.now(tz).date()

		#load data
		split_by_date = True
		if self.LOOKUP_STEP > 179:
		 	split_by_date = False

		self.data = self.load_data(self.ticker, self.N_STEPS, lookup_step=self.LOOKUP_STEP, split_by_date=split_by_date)



		self.evaluate_model()

		print(f"Ticker: {self.ticker}")
		print(f"Future price after {self.LOOKUP_STEP} days is {self.future_price:.2f}$")
		print(f"{self.LOSS} loss:", self.loss)
		print("Mean Absolute Error:", self.mean_absolute_error)
		print("Accuracy score:", self.accuracy_score)
		print("Total buy profit:", self.total_buy_profit)
		print("Total sell profit:", self.total_sell_profit)
		print("Total profit:", self.total_profit)
		print("Profit per trade:", self.profit_per_trade)

		return self
		#self.plot.show()		



