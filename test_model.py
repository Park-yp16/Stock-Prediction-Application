from model import *
from math import ceil
plt.ion()
def makeModel(ticker, days=30, n_step=50):
	return MyModel(days, ticker, epochs=50, n_steps=50) #n_steps=int(ceil(days*(10.0/3.0))))

days = [15, 30, 60, 90, 180,365,365*2]
#days = [365,365*2,365*5,365*10, 365*15]
for day in days: 
	AMZN = makeModel("AMZN",day)
	TSLA = makeModel("TSLA",day)
	NASDAQ = makeModel("^IXIC",day)
	DJI = makeModel("^DJI",day)
	AAPL= makeModel("AAPL",day)
	MSFT = makeModel("MSFT",day)
	if day < 730:
		NIO = makeModel("NIO",day)
	NVDA = makeModel("NVDA",day)
	FB = makeModel("FB",day)
	TWTR = makeModel("TWTR",day)
	WMT= makeModel("WMT",day)
	SP500 = makeModel("^GSPC",day)
	GME = makeModel("GME",day)

#model = MyModel(30, "AMZN", epochs=2)

#a = MyModel.fromModel("2021-04-26_^DJI_180days")