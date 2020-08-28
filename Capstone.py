#!/usr/bin/env python
# coding: utf-8

# Load dataset

# In[1]:


import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_excel(r"C:\Users\Sushant\Desktop\Capstone\Customer_Data.xlsx")
df2 = pd.read_csv(r"C:\Users\Sushant\Desktop\Capstone\Final_invoice.csv")
df3 = pd.read_csv(r"C:\Users\Sushant\Desktop\Capstone\JTD.csv")


# In[2]:


df1.head()


# In[3]:


df2.head()


# In[4]:


data = df1.merge(df2,how='inner',on=['Customer No.'])
data.head()
data = data.rename({'Unnamed: 0':'DBM Order'}, axis=1)
data.columns


# In[5]:


final_data = data.merge(df3, how='inner',on=['DBM Order'])
final_data.info()


# In[6]:


final_data.drop(columns=['CGST(14%)','CGST(2.5%)','CGST(6%)','CGST(9%)'], axis = 1, inplace=True)
final_data.drop(columns=['IGST(12%)','IGST(18%)','IGST(28%)','IGST(5%)'], axis = 1, inplace=True)
final_data.drop(columns=['SGST/UGST(14%)','SGST/UGST(2.5%)','SGST/UGST(6%)', 'SGST/UGST(9%)'], axis = 1, inplace=True)
final_data.drop(columns=['ODN No.','Partner Type','Technician Name','Description','Data Origin','Gate Pass Date','Labor Value Number'],axis = 1, inplace=True)


# # Treating on missing data

# In[7]:


null_per = final_data.isnull().sum() / len(final_data)
missing_features = null_per[null_per > 0.80].index
final_data.drop(missing_features, axis=1, inplace=True)
final_data['Total Value'] = final_data['Total Value'].fillna((final_data['Total Value'].mean()))
final_data['Amt Rcvd From Custom'] = final_data['Amt Rcvd From Custom'].fillna((final_data['Amt Rcvd From Custom'].mean()))
final_data['Amt Rcvd From Ins Co'] = final_data['Amt Rcvd From Ins Co'].fillna((final_data['Amt Rcvd From Ins Co'].mean()))
for column in final_data['Model']:
    final_data['Model'].fillna(final_data['Model'].mode()[0], inplace=True)
final_data['Outstanding Amt'] = final_data['Outstanding Amt'].fillna((final_data['Outstanding Amt'].mean()))
final_data['TDS amount'] = final_data['TDS amount'].fillna((final_data['TDS amount'].mean()))
final_data['Total CGST'] = final_data['Total CGST'].fillna((final_data['Total CGST'].mean()))
final_data['Total GST'] = final_data['Total GST'].fillna((final_data['Total GST'].mean()))
final_data['Total IGST'] = final_data['Total IGST'].fillna((final_data['Total IGST'].mean()))
final_data['Total SGST/UGST'] = final_data['Total SGST/UGST'].fillna((final_data['Total SGST/UGST'].mean()))
for column in final_data['Material']:
    final_data['Material'].fillna(final_data['Material'].mode()[0], inplace=True)
final_data.info()


# # Q.1 Which area has more cars? (Top 10 states)

# In[8]:


area_car = df2['District'].value_counts().sort_values(ascending = False).head(10)
print(area_car)
# bar chart
plt.figure(figsize=[10,10])
plt.xlabel("States")
plt.ylabel("Number of cars")
plt.title("State wise ditribution of cars")
area_car.plot(kind = 'bar')
# display plot
plt.show()


# 
# 
# # Q.2 Which cars are more popular? (Top 50 cars)

# In[9]:


popular_car = df2['Model'].value_counts().sort_values(ascending=False).head(50)
print(popular_car)
plt.figure(figsize=[15,15])
plt.xlabel("Car Model")
plt.ylabel("Number of occurances")
plt.title("Most owned car models")
popular_car.plot(kind = 'bar')
plt.show()


# # Lowest owened car models

# In[10]:


lowest_owned_car = final_data['Model'].value_counts().sort_values(ascending=True).head(10)
print(lowest_owned_car)


# # Q.3 What is service structure for particular car/make?

# In[11]:


popular_service = final_data.groupby(['Order Type','Make']).size().unstack()
print(popular_service)
popular_service.plot(kind='bar', stacked=True, figsize=(15,10))
plt.title("Service structure for particular make")
plt.show()


# # Which order type gives more revenue?

# In[12]:


revenue = final_data.groupby('Order Type')['Total Value'].sum().astype(int)
print(revenue)
plt.figure(figsize=[10,10])
plt.xlabel("Order Type")
plt.ylabel("Total Value")
plt.title("Order type wise revenue generation")
revenue.plot(kind = 'bar')
plt.show()


# # Q.4 Which type of service is popular in a certain area?

# In[13]:


print(final_data['District'].value_counts())
popular_service = final_data.groupby(['District','Order Type']).size().unstack()
print(popular_service)
popular_service.plot(kind='bar', stacked=True, figsize=(15,10))
plt.title("Area wise popularity of service")
plt.show()


# # Area wise Revenue generated through Labour cost 

# In[14]:


Labour_total = pd.pivot_table(final_data,index=['District','Order Type'], values=['Labour Total'],aggfunc= np.sum)
print(Labour_total)


# # What is the difference between each service labour costing?

# In[15]:


df2['Labour Total'] = df2['Labour Total'].astype(int)
labour = df2.groupby('Order Type')['Labour Total'].sum()
print(labour)
order_labour = labour.sort_values(ascending=False).plot(kind='bar')
plt.xlabel("Order Type")
plt.ylabel("Labour cost")
plt.title("Difference in each service labour costing")


# # Which service is popular regading certain car/make?

# In[16]:


popular_service_make = df2.groupby(['Make','Order Type']).size().unstack()
print(popular_service_make)
popular_service_make.plot(kind='bar', stacked=True, figsize=(15,10))
plt.title("Make wise popularity of service")
plt.show()


# In[17]:


total = df2.pivot_table(index=['Cust Type','Order Type'], values=['Total Amt Wtd Tax.'],aggfunc=np.mean)
print(total)
total.plot(kind='bar',figsize=(15,10))
plt.title("Customer Type")
plt.show()


# In[18]:


repeat_customers = df2.groupby(by = ['Customer No.']).size().sort_values(ascending =False).head(10)
print(repeat_customers)


# In[19]:


import seaborn as sns
total = final_data.pivot_table(index=['District'],columns='Order Type',values=['Total Value'])
total = total.replace(np.nan, 0)
print(total)
heat_map=sns.heatmap(total)
plt.figure(figsize=[20,20])
plt.show()


# In[20]:


final_data.columns


# In[21]:


cust_type = final_data[['Cust Type', 'Total Amt Wtd Tax.']].groupby(['Cust Type'], as_index=False).sum().sort_values(by='Total Amt Wtd Tax.', ascending=False)
print(cust_type)
cust_type.plot(kind='barh',figsize=(15,10))
plt.xlabel("Total Amt Wtd Tax.")
plt.ylabel("Cust Type")
plt.title("Customer type vs. Revenue generated")
plt.show()


# In[22]:


Dist = final_data[['District', 'Total Amt Wtd Tax.']].groupby(['District'], as_index=False).sum().sort_values(by='Total Amt Wtd Tax.', ascending=False)
print(Dist)
Dist.plot(kind='barh',figsize=(15,10))
plt.xlabel("Total Amt Wtd Tax.")
plt.ylabel("District")
plt.title("District vs. Revenue generated")
plt.show()


# In[23]:


Make = final_data[['Make', 'Total Amt Wtd Tax.']].groupby(['Make'], as_index=False).sum().sort_values(by='Total Amt Wtd Tax.', ascending=False)
print(Make)
Make.plot(kind='barh',figsize=(15,10))
plt.xlabel("Total Amt Wtd Tax.")
plt.ylabel("Make")
plt.title("Make vs. Revenue generated")
plt.show()


# In[24]:


final_data[['Order Type', 'Total Value']].groupby(['Order Type'], as_index=False).sum().sort_values(by='Total Value', ascending=False)


# In[25]:


X = final_data[['Order Type', 'Total Value']]
#Visualise data points
plt.scatter(X['Order Type'],X['Total Value'],c='black')
plt.xlabel('Order Type')
plt.ylabel('Total Value')
plt.show()


# In[26]:


final_data.columns


# In[27]:


matrix = final_data.pivot_table(index='Customer No.', 
columns='Order Type',values='Total Value')
matrix.fillna(0, inplace=True)
matrix.reset_index(inplace=True)
matrix.head(5)


# In[28]:


from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=3, init='k-means++', 
max_iter=300, n_init=10, random_state=0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix.head(5)


# In[29]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
clusters = matrix.iloc[:,:]
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')


# In[30]:


matrix = final_data.pivot_table(index='Customer No.', 
columns='Cust Type',values='Total Value')
matrix.fillna(0, inplace=True)
matrix.reset_index(inplace=True)
matrix.head(5)


# In[31]:


from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=3, init='k-means++', 
max_iter=300, n_init=10, random_state=0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix.head(5)


# In[32]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
clusters = matrix.iloc[:,:]
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')


# In[33]:


matrix = final_data.pivot_table(index='Customer No.', 
columns='Make',values='Total Value')
matrix.fillna(0, inplace=True)
matrix.reset_index(inplace=True)
matrix.head(5)


# In[34]:


from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=3, init='k-means++', 
max_iter=300, n_init=10, random_state=0)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix.head(5)


# In[35]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
clusters = matrix.iloc[:,:]
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')


# In[36]:


final_data.columns


# # RFM Model

# In[37]:


from datetime import timedelta
import squarify
final_data['Invoice Date'] = pd.to_datetime(final_data['Invoice Date'])
# Create snapshot date
snapshot_date = final_data['Invoice Date'].max() + timedelta(days=1)
print(snapshot_date)
# Grouping by CustomerID
data_process = final_data.groupby(['Customer No.']).agg({
        'Invoice Date': lambda x: (snapshot_date - x.max()).days,
        'Invoice No': 'count',
        'Total Value': 'sum'})
# Rename the columns 
data_process.rename(columns={'Invoice Date': 'Recency',
                         'Invoice No': 'Frequency',
                         'Total Value': 'MonetaryValue'}, inplace=True)
print(data_process.head())
print('{:,} rows; {:,} columns'
      .format(data_process.shape[0], data_process.shape[1]))


# In[38]:


# Plot RFM distributions
plt.figure(figsize=(12,10))# Plot distribution of R
plt.subplot(3, 1, 1); sns.distplot(data_process['Recency'])# Plot distribution of F
plt.subplot(3, 1, 2); sns.distplot(data_process['Frequency'])# Plot distribution of M
plt.subplot(3, 1, 3); sns.distplot(data_process['MonetaryValue'])# Show the plot
plt.show()


# In[39]:


# --Calculate R and F groups--# Create labels for Recency and Frequency
r_labels = range(4, 0, -1); f_labels = range(1, 5)# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(data_process['Recency'], q=4, labels=r_labels)# Assign these labels to 4 equal percentile groups 
f_groups = pd.qcut(data_process['Frequency'], q=4, labels=f_labels)# Create new columns R and F 
data_process = data_process.assign(R = r_groups.values, F = f_groups.values)
data_process.head()


# In[40]:


# Create labels for MonetaryValue
m_labels = range(1, 5)# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(data_process['MonetaryValue'], q=4, labels=m_labels)# Create new column M
data_process = data_process.assign(M = m_groups.values)


# In[41]:


def join_rfm(x): return str(x['R']) + str(x['F']) + str(x['M'])
data_process['RFM_Segment_Concat'] = data_process.apply(join_rfm, axis=1)
rfm = data_process
rfm.head()


# In[42]:


# Count num of unique segments
rfm_count_unique = rfm.groupby('RFM_Segment_Concat')['RFM_Segment_Concat'].nunique()
print(rfm_count_unique.sum())


# In[43]:


# Calculate RFM_Score
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)
print(rfm['RFM_Score'].head())


# In[44]:


# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Can\'t Loose Them'
    elif ((df['RFM_Score'] >= 8) and (df['RFM_Score'] < 9)):
        return 'Champions'
    elif ((df['RFM_Score'] >= 7) and (df['RFM_Score'] < 8)):
        return 'Loyal'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 7)):
        return 'Potential'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 6)):
        return 'Promising'
    elif ((df['RFM_Score'] >= 4) and (df['RFM_Score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Require Activation'# Create a new variable RFM_Level
rfm['RFM_Level'] = rfm.apply(rfm_level, axis=1)# Print the header with top 5 rows to the console
rfm.head()


# In[45]:


# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_level_agg = rfm.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
}).round(1)# Print the aggregated dataset
print(rfm_level_agg)


# In[46]:


rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=rfm_level_agg['Count'], 
              label=['Can\'t Loose Them',
                     'Champions',
                     'Loyal',
                     'Needs Attention',
                     'Potential', 
                     'Promising', 
                     'Require Activation'], alpha=.6 )
plt.title("RFM Segments",fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()


# In[47]:


final_data.columns


# In[48]:


data=final_data[['Customer No.','Invoice Date', 'Invoice No','Total Value']]
total_days = (data['Invoice Date'].max()-final_data['Invoice Date'].min()).days
data_group = data.groupby('Customer No.').agg({'Invoice Date':lambda date:total_days,
                                               'Invoice No':lambda num: len(num),
                                              'Total Value': lambda price: price.sum()})
data_group.columns=['num_days','num_transactions','spent_money']
data_group.head()


# CLTV Using formula

# In[49]:


# Average Order Value
data_group['avg_order_value']=data_group['spent_money']/data_group['num_transactions']
#Purchase Frequency 
purchase_frequency=sum(data_group['num_transactions'])/data_group.shape[0]
# Repeat Rate
repeat_rate= data_group[data_group.num_transactions > 1].shape[0]/data_group.shape[0]
#Churn Rate
churn_rate=1-repeat_rate
print('Purchase Frequency:',purchase_frequency)
print('repeat_rate:',repeat_rate)
print('churn_rate',churn_rate)


# Profit Margin & CLTV (Let us assume 25% profit margin)

# In[50]:


# Profit Margin
data_group['profit_margin']= data_group['spent_money']*0.25
# Customer Value
data_group['CLTV']= (data_group['avg_order_value']*purchase_frequency)/churn_rate
#Customer Lifetime Value
data_group['cust_lifetime_value']=data_group['CLTV']*data_group['profit_margin']
data_group.head()


# Prediction Model for CLTV

# In[51]:


# Extract month and year from InvoiceDate
data['month_yr'] = data['Invoice Date'].apply(lambda x: x.strftime('%b-%Y'))
sale = data.pivot_table(index=['Customer No.'],columns=['month_yr'],values='Total Value',aggfunc='sum',fill_value=0).reset_index()
# sum all the months sale
sale['CLV']=sale.iloc[:,2:].sum(axis=1) 
sale.head()


# regression model for existing customers

# In[52]:


sale.columns


# In[57]:


X = sale.drop(['Customer No.','CLV'], axis = 1)
y = sale['CLV']
#split training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=6)
cols = X_train.columns

#Let's check the scatter_plot for different features vs target variable 'CLV'
fig, axes = plt.subplots(nrows=3 , ncols=3, figsize=(20,20))

for i in range(0, 3):
    for j in range(0,3):
        col = cols[ i * 3 + j]
        axes[i, j].set_title(col)
        axes[i, j].scatter(X_train[col], y_train)
        axes[i, j].set_xlabel(col)
        axes[i, j].set_ylabel('CLV')

plt.show()


# In[58]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('mean_squared_error:',mse)

r2 = r2_score(y_test, y_pred)
print('r2_score:',r2)

mae = mean_absolute_error(y_test,y_pred)
print('mean_absolute_error:',mae)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('root_mean_squared_error:',rmse)


# In[ ]:




