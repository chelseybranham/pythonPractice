import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('/content/sample_data/california_housing_train.csv')
test=pd.read_csv('/content/sample_data/california_housing_test.csv')

train=train.append(test)
train

sns.distplot(train['median_house_value'])

var= 'median_income'
data=pd.concat([train['median_house_value'], train[var]], axis=1)
data.plot.scatter(x=var, y='median_house_value', ylim=(0, 600000), s=32);


X = data.drop('median_house_value', axis=1)  # Features
y = data['median_house_value']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print('Mean squared error:', mse)