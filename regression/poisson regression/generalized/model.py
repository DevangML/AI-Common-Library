import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Create a pandas DataFrame for the counts data set.
df = pd.read_csv('nyc_bb_bicyclist_counts.csv', header=0,
                 infer_datetime_format=True, parse_dates=[0], index_col=[0])

# We'll add a few derived regression variables to the X matrix.
ds = df.index.to_series()
df['MONTH'] = ds.dt.month
df['DAY_OF_WEEK'] = ds.dt.dayofweek
df['DAY'] = ds.dt.day

# Let's print out the first few rows of our data set to see how it looks like
print(df.head(10))

# Let's create the training and testing data sets.
mask = np.random.rand(len(df)) < 0.8
df_train = df[mask]
df_test = df[~mask]
print('Training data set length='+str(len(df_train)))
print('Testing data set length='+str(len(df_test)))

# Setup the regression expression in Patsy notation.
# We are telling patsy that BB_COUNT is our dependent variable y and it depends on the regression variables X:
# DAY, DAY_OF_WEEK, MONTH, HIGH_T, LOW_T and PRECIP.
expr = 'BB_COUNT ~ DAY  + DAY_OF_WEEK + MONTH + HIGH_T + LOW_T + PRECIP'

# Let's use Patsy to carve out the X and y matrices for the training and testing data sets:
y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

# Using the statsmodels GLM class, train the Poisson regression model on the training data set.
poisson_training_results = sm.GLM(
    y_train, X_train, family=sm.families.Poisson()).fit()

# Print the training summary.
print(poisson_training_results.summary())

# Let's print out the variance and mean of the data set
print('variance='+str(df['BB_COUNT'].var()))
print('mean='+str(df['BB_COUNT'].mean()))

# Build Consul's Generalized Poison regression model, know as GP-1
gen_poisson_gp1 = sm.GeneralizedPoisson(y_train, X_train, p=1)

# Fit the model
gen_poisson_gp1_results = gen_poisson_gp1.fit()

# print the results
print(gen_poisson_gp1_results.summary())

# Get the model's predictions on the test data set
gen_poisson_gp1_predictions = gen_poisson_gp1_results.predict(X_test)

predicted_counts = gen_poisson_gp1_predictions
actual_counts = y_test['BB_COUNT']

fig = plt.figure()
fig.suptitle('Predicted versus actual bicyclist counts on the Brooklyn bridge')
predicted, = plt.plot(X_test.index, predicted_counts,
                      'go-', label='Predicted counts')
actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()

# Build Famoye's Restricted Generalized Poison regression model, know as GP-2
gen_poisson_gp2 = sm.GeneralizedPoisson(y_train, X_train, p=2)

# Fit the model
gen_poisson_gp2_results = gen_poisson_gp2.fit()

# print the results
print(gen_poisson_gp2_results.summary())
