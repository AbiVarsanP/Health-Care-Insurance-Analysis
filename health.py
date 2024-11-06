import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
hosp = pd.read_csv('Hospitalisation_details.csv')
medi = pd.read_csv('Medical Examinations.csv')
name = pd.read_excel('Names.xlsx')
hosp.info()
medi.info()
name.info()

table1 = pd.merge(hosp,medi, on = "Customer ID")
dataset = pd.merge(table1,name, on = "Customer ID")
dataset

dataset.isnull().sum()

trivial_rows = dataset[dataset == "?"].count(axis=1).sum()
trivial_rows

total_rows = dataset.shape[0]
total_rows

percentage = (trivial_rows / total_rows) * 100
percentage

print("Percentage of trivial rows: {:.2f}%".format(percentage))

dataset = dataset[dataset != "?"].dropna()
dataset.shape

dataset_cat = dataset.select_dtypes(exclude='number')
dataset_cat.columns

dataset['Heart Issues'].value_counts()

dataset['Any Transplants'].value_counts()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
dataset["Heart Issues"]=le.fit_transform(dataset["Heart Issues"])
dataset["Any Transplants"]=le.fit_transform(dataset["Any Transplants"])
dataset["Cancer history"]=le.fit_transform(dataset["Cancer history"])
dataset["smoker"]=le.fit_transform(dataset["smoker"])
dataset["Heart Issues"].value_counts()

dataset['Hospital tier'].value_counts()

dataset['Hospital tier'] = dataset['Hospital tier'].str.replace('tier', '')
#df['col1'] = df['col1'].str.replace('example-', '')
dataset['Hospital tier'].value_counts()

dataset['Hospital tier'] = dataset['Hospital tier'].str.replace("-", "")
dataset['Hospital tier'] = dataset['Hospital tier'].astype(int)
dataset


dataset['City tier'] = dataset['City tier'].str.replace("tier", "")
dataset['City tier'] = dataset['City tier'].str.replace("-", "")
dataset


dataset['City tier'] = dataset['City tier'].astype(int)
dataset



dataset['state_group'] = np.where((dataset['State ID'] == 'R1011') | (dataset['State ID'] == 'R1012') | (dataset['State ID'] == 'R1013'), dataset['State ID'], 'other')
state_dummies = pd.get_dummies(dataset['state_group'], prefix='state')
df = pd.concat([dataset, state_dummies], axis=1)
dataset


dataset=dataset[dataset["State ID"].isin(['R1011','R1012','R1013'])]
dataset.shape
dataset["State ID"]=le.fit_transform(dataset["State ID"])
dataset["State ID"].unique()

dataset['state_group'].value_counts()

 
dataset["state_group"].replace('R1011',1,inplace=True)
dataset["state_group"].replace('R1012',2,inplace=True)
dataset["state_group"].replace('R1013',3,inplace=True)
dataset["state_group"].replace('other',0,inplace=True)

dataset

dataset["NumberOfMajorSurgeries"].replace('No major surgery',0,inplace=True)

dataset

month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
dataset['month'] = dataset['month'].map(month_dict)

dataset


dataset.year = dataset.year.astype(int)

dataset['age'] = 2023 - dataset.year

dataset

gender= ['0' if 'Mr.' in name else '1' for name in dataset['name']]
dataset["Gender"]=gender
dataset.head()

plt.figure(figsize=(15,8))
sns.histplot(dataset['charges'])
plt.title('Distribution of cost')
plt.show()

plt.figure(figsize=(15,8))
sns.boxplot(dataset['charges'])
plt.title('Distribution of cost')
plt.show()


plt.figure(figsize=(15,8))
sns.swarmplot(dataset['charges'])
plt.title('Distribution of cost')
plt.show()


plt.figure(figsize=(15,8))
sns.boxplot(x = 'charges', y = 'Gender',data = dataset)
plt.title('Distribution of cost')
plt.show()

plt.figure(figsize = (15,5))
sns.boxplot(x = "City tier",y = "charges", data = dataset)
plt.show()

median = dataset.groupby('Hospital tier')[['charges']].median().reset_index()
import plotly.express as px
fig = px.line_polar(median, r='charges', theta='Hospital tier', line_close=True)
fig.show()
table = pd.crosstab(dataset['City tier'], dataset['Hospital tier'])
print(table)

table.plot(kind='bar', stacked=True)
plt.xlabel('City tier')
plt.ylabel('Hospital tier')
plt.title('Count of People in Different Tiers of Cities and Hospitals')
plt.show()


import scipy.stats as stats
print('Null Hypothesis => Average hospitalization costs for the three types of hospitals are not significantly different.')
f_val, p_val = stats.f_oneway(dataset[dataset['Hospital tier'] == 'tier,1']['charges'],
                              dataset[dataset['Hospital tier'] == 'tier,2']['charges'],
                              dataset[dataset['Hospital tier'] == 'tier,3']['charges'])
print('P-value :',p_val)
if p_val < 0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")

print('Null Hypothesis => Average hospitalization costs for the three types of cities are not significantly different.')
f_val, p_val = stats.f_oneway(dataset[dataset['City tier'] == 'tier,1']['charges'],
                              dataset[dataset['City tier'] == 'tier,2']['charges'],
                              dataset[dataset['City tier'] == 'tier,3']['charges'])
print('P-value :',p_val)
if p_val < 0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")

print('Null Hypothesis => Average hospitalization costs for smokers is not significantly different from the average cost for nonsmokers.')
t_val, p_val = stats.ttest_ind(dataset[dataset['smoker'] == 'yes']['charges'],
                              dataset[dataset['smoker'] == 'no']['charges'])
                          
print('P-value :',p_val)
if p_val < 0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")

from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(dataset['smoker'], dataset['Heart Issues'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f'P-value = {p}')
if p < 0.05:
    print("Reject the null hypothesis, Smoking and heart issues are independent.")
else:
    print("Accept null hypothesis, Smoking and heart issues are independent.")

dataset.drop(["Customer ID",'name'], inplace=True, axis=1)


plt.figure(figsize=(15,10))
sns.heatmap(dataset.corr(),square=True,annot=True,linewidths=1)


from sklearn.model_selection import train_test_split
x = dataset.drop(["charges"], axis=1)
y = dataset['charges']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.20,random_state=10)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2,0.3,0.4,0.5,
0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
9.0,10.0,20,50,100,500,1000],
'penalty': ['l2', 'l1', 'elasticnet']}
sgd = SGDRegressor()
# Cross Validation
folds = 5
model_cv = GridSearchCV(estimator = sgd,
param_grid = params,
scoring = 'neg_mean_absolute_error',
cv = folds,
return_train_score = True,
verbose = 1)
model_cv.fit(x_train,y_train)

model_cv.best_params_
sgd = SGDRegressor(alpha= 100, penalty= 'l1')
sgd.fit(x_train, y_train)
sgd.score(x_test, y_test)
y_pred = sgd.predict(x_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
sgd_mae = mean_absolute_error(y_test, y_pred)
sgd_mse = mean_squared_error(y_test, y_pred)
sgd_rmse = sgd_mse*(1/2.0)
print("MAE:", sgd_mae)
print("MSE:", sgd_mse)
print("RMSE:", sgd_rmse)