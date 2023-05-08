import numpy as np

# define how the market is cleared
def market_clearing(self, bid_price, bid_volume, date):
    '''
        A function that calculates the output the day-ahead market would give when the selcted bid is submitted [EUR]
       
        Return: overall profit received from market in EUR and realised market price in EUR/MWh
    '''


    # calculate revenue
    
    if bid_price <= self.prices[date]:
        #bid is sucessful
        profit = bid_volume * (self.prices[date] - self.mc)

    else:
        #bid is not sucessfull
        profit = 0


    return profit, self.prices[date]
