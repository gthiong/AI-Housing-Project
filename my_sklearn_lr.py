#Remember our linear regression example? Let's train it locally and register it

#prepare the environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sklearn
#load the data
from sklearn.datasets import load_boston
boston_dataset = load_boston()


#prepare the data
from sklearn.model_selection import train_test_split
num_Rooms_Train, num_Rooms_Test, med_price_Train, med_Price_Test = train_test_split(boston_dataset.data[:,5].reshape(-1,1), boston_dataset.target.reshape(-1,1))


#implement linear regression model
from sklearn.linear_model import LinearRegression
price_room = LinearRegression()
price_room.fit (num_Rooms_Train,med_price_Train)

# Let's save this model fit out
from sklearn.externals import joblib
os.makedirs("outputs", exist_ok=True)
joblib.dump(value=price_room, filename="outputs/bh_lr.pkl")
