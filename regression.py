import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# %matplotlib inline

dataset = pd.read_csv('/home/ehigie/Downloads/Compressed/weatherHistory.csv')

print(dataset.shape)

print(dataset.describe())

dataset.plot(x='Temperature (C)', y='Humidity', style='o')
plt.title('Temp Vs Humidity')
plt.xlabel('Temperature (C)')
plt.ylabel('Humidity')
plt.show()

plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.displot(['Humidity'])
plt.show()

x = dataset['Temperature (C)'].values.reshape(-1,1)
y = dataset['Humidity'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print('Intercept:', regressor.intercept_)

print('Coefficient:', regressor.coef_)

y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(df)

df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5',color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5',color='black')
plt.show()

plt.scatter(x_test, y_test, color='gray')
plt.plot(x_test,y_pred, color='red', linewidth=2)
plt.show()

# evaluate the performance of the algorithm on the dataset 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))