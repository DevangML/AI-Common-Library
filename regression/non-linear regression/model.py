import numpy as num
import pandas as pds
import matplotlib.pyplot as plot


df = pds.read_csv("regressionchina_gdp.csv")


def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y


beta1 = 0.10
beta2 = 1990.0

ypred = sigmoid(x_data, beta1, beta2)

plot.plot(x_data, ypred * 16000000000000.)
plot.plot(x_data, y_data, 'go')
