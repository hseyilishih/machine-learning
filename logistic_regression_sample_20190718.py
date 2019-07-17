# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:58:17 2019

@author: 000147A

direct marketing campaigns (phone calls) of a Portuguese banking institution. 
The classification goal is to predict 
whether the client will subscribe (1/0) 
to a term deposit (variable y)
"""


# coding: utf-8

#https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import locale
locale.setlocale(locale.LC_ALL, '')  # Use '' for auto, or force e.g. to 'en_US.UTF-8'

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import sklearn
print('sklearn: %s' % sklearn.__version__)


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.feature_selection import RFE

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE


import statsmodels.api as sm


#%matplotlib inline
#%matplotlib auto (reset 回來)
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'auto')

# # The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe (1/0) a term deposit (variable y).
# # This dataset provides the customer information. It includes 41188 records and 21 fields.

data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
data.head()


# #### Input variables
# 
# 1 - age (numeric)
# 
# 2v - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 
# 3x - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 
# 4 - education (categorical: 
#'basic.4y','basic.6y','basic.9y',---> 重新歸為Basic
#'high.school','illiterate','professional.course','university.degree','unknown')
# 
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# 
# 8 - contact: contact communication type (categorical: 'cellular','telephone')
# 
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 
# 15v - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# 
# 16 - emp.var.rate: employment variation rate - (numeric)
# 
# 17 - cons.price.idx: consumer price index - (numeric)
# 
# 18 - cons.conf.idx: consumer confidence index - (numeric)
# 
# 19 - euribor3m: euribor 3 month rate - (numeric)

# 20 - nr.employed: number of employees - (numeric)

# #### Predict variable (desired target):
# y - has the client subscribed a term deposit? (binary: '1','0')

# The education column of the dataset has many categories and we need to reduce the categories for a better modelling. The education column has the following categories:


data['education'].unique()


#  group "basic.4y", "basic.9y" and "basic.6y" together and call them "basic".


data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

# After grouping, this is the columns.

data['education'].unique()


#=================================================================== ### Data exploration


data['y'].value_counts()

sns.countplot(x='y',data=data, palette='hls')
plt.show()



count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])

print('0 count_no_sub=', count_no_sub)
print('1 count_sub=', count_sub)

pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)

print("percentage of no subscription is", pct_of_no_sub*100)

pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)


# Our classes are imbalanced, and the ratio of no-subscription to subscription instances is 89:11. Before we go ahead to balance the classes, Let's do some more exploration.
#使用y 的內含值來分類,看看每個變數x的平均值
data.groupby('y').mean()


# Observations:
# 
# The average age of customers who bought the term deposit is higher than that of the customers who didn't. The pdays (days since the customer was last contacted) is understandably lower for the customers who bought it. The lower the pdays, the better the memory of the last call and hence the better chances of a sale. Surprisingly, campaigns (number of contacts or calls made during the current campaign) are lower for customers who bought the term deposit.
# 
# We can calculate categorical means for other categorical variables such as education and marital status to get a more detailed sense of our data.


data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()

#====================================================================================== Visualizations

# 2v - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')

# The frequency of purchase of the deposit depends a great deal on the job title. 
#Thus, the job title can be a good predictor of the outcome variable.


table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')

# the marital status does not seem a strong predictor for the outcome variable.


table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')
# Education seems a good predictor of the outcome variable.

pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')
# Day of week may not be a good predictor of the outcome.


pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')
# Month might be a good predictor of the outcome variable.


# Most customers of the bank in this dataset are in the age range of 30-40.

data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')


pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_pout_bar')

# Poutcome seems to be a good predictor of the outcome variable.

# ===================================================================================== Create dummy variables



data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
print('shape before pd.get_dummies=', data.shape)
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])




# 如果以下是我們經過資料探索 然後覺得有機會的 X 變數
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

cnt_dummy_var = 0 
for x in cat_vars:
    print(x, len(data[x].unique()),'\n',  data[x].unique(),'\n')
    cnt_dummy_var += len(data[x].unique())

print('********************** total cat_vars=', len(cat_vars), ', conver to total dummy var=', cnt_dummy_var)

#for x in cat_vars:
# 改用為 automatic counter of loop
for idx, x in enumerate(cat_vars):
    print("cat_vars sequence=", idx+1, 'value=', x)
    cat_list='var'+'_'+ x
    cat_list = pd.get_dummies(data[x], prefix=x)
    #cat_list is dataframe, one column become n column depends on how many unique value in a column
    data1=data.join(cat_list)  # use old data to join and assign to a temp data1
    data=data1 #replace to old data with temp data1
print('shape after pd.get_dummies=', data.shape)

#==============================================================================================================================
# 如果以下是我們經過資料探索 然後覺得有機會的 X categorical類別變數 10個
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
print(len(cat_vars))



#把原來所有的欄位放到 data_vars這個 list, 72 個, 包含Y
data_vars=data.columns.values.tolist()
print(type(data_vars), len(data_vars), '\n', data_vars )

#把 有轉 dummy variable 的原 x 去掉, 只留下 沒有轉dummy 以及 生出來的dummy variables
to_keep=[i for i in data_vars if i not in cat_vars]
print(type(to_keep), len(to_keep), '\n', to_keep )
# 應該剩下 72 - 10個cat_vars = 62

data_final=data[to_keep] #把這些要keep欄位 轉成 temp data frame 名為 data_final
data_final.columns.values  #df.columns.values 就可以show 出所有field name, 沒寫.tolist()所以data type 是 array


X = data_final.loc[:, data_final.columns != 'y'] #把非y的欄位 複製到大寫 X
y = data_final.loc[:, data_final.columns == 'y'] #把y的欄位 複製到 小寫y

# ===============================================================================### Over-sampling using SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
print(type(columns), columns) #奇怪 只有16 columns
'''
<class 'pandas.core.indexes.base.Index'> 
Index(['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
       'month_apr', 'month_aug', 'month_dec','month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov','month_oct', 
       'poutcome_failure', 'poutcome_success'],
      dtype='object')
'''

#X_train.columns.values 


os = SMOTE(random_state=0) 
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

#  Check the numbers of our data
print("Number of oversampled data is ",  f'{len(os_data_X) :n}'     )

#int(len(os_data_y[os_data_y['y']==0]) )
print("Number of no subscription in oversampled data",  int(len(os_data_y[os_data_y['y']==0]) ))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))

print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


# ### Recursive feature elimination

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]

#https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
#logreg = LogisticRegression(solver='lbfgs')

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# The Recursive Feature Elimination (RFE) has helped us select the following features: 
#"previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", "poutcome_failure", "poutcome_nonexistent", "poutcome_success".

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']


# ### Implementing the model

logit_model=sm.Logit(y,X)
result=logit_model.fit() #訓練
print(result.summary2()) 

# The p-values for four variables are very high, therefore, we will remove them.

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
      'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 
	  'poutcome_failure', 'poutcome_success'] 
	  
	#重新準備 X, y
X=os_data_X[cols]
y=os_data_y['y']


logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# ### Logistic Regression Model Fitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#logreg = LogisticRegression(solver='lbfgs')
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# ### Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


print(classification_report(y_test, y_pred))


# #### Interpretation:

# Of the entire test set, 74% of the promoted term deposit were the term deposit that the customers liked. Of the entire test set, 74% of the customer's preferred term deposit were promoted.





logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

