#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import requests
from bs4 import BeautifulSoup
import itertools 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


# # Data Acquisition
# For this section , we used BeautifulSoup libray as learned.
# For each column , we created a list and insrted the relavent data.
# At the end we combined all lists to one list of lists. This is the dataset.

# In[2]:


main_url= 'https://www.truecar.com/new-cars-for-sale/listings/'
response = requests.get(main_url)
soup = BeautifulSoup(response.content, 'html.parser')


# In[3]:


car_urls = []
overview_questions = []

cars = soup.find_all("div", {"class": "card-content order-3 vehicle-card-body"})
for car in cars:
    car_url = "https://www.truecar.com"+car.find("a")["href"]
    car_urls.append(car_url)


# In[4]:


Style = []
Exterior_Color=[]
Interior_Color=[]
Engine=[]
Drive_Type=[]
Fuel_Type=[]
Transmission=[]
Cab_Type=[]
Price=[]
Moonroof=[]
Backup_Camera=[]
Parking_Sensors=[]
Bluetooth=[]
Navigation=[]
Front_Heated_Seats=[]
Names = []
Brand=[]
Year=[]
car_names=[]

myList= [Names,Brand,Year, Style, Exterior_Color, Interior_Color, Engine, Drive_Type, Fuel_Type, Transmission,Cab_Type, Price, 
         Moonroof, Backup_Camera, Parking_Sensors, Bluetooth, Navigation, Front_Heated_Seats]


# In order to get the relavent data for our research, we built the get_attributes function and than called it 300 times , as each times the data is taken from all 30 cars in each page.

# In[5]:


def get_attributes ():

    for car_url in car_urls:    
        response = requests.get(car_url)
        cars_soup = BeautifulSoup(response.content, 'html.parser')
        vehicle_overview = cars_soup.find("div",{"class": "py-5"})
        price = cars_soup.find("div",{"data-test": "vdpPreProspectPrice"})
        questions = vehicle_overview.find_all("div",{"class": "heading-4"})
        answers = vehicle_overview.find_all("p",{"class": "text-base"})
        popular_features = cars_soup.find("div",{"class": "bg-light py-4"})
        features = popular_features.find_all("p",{"class": "text-base"})
        list1 = []
        list2 = []
        
        
        name = car.find("span",{"class": "truncate"}).get_text()
        car_names.append(name)
        
        for question, answer in zip(questions, answers):
            q = question.get_text()
            a = answer.get_text()
            if (q == 'Style'):
                myList[3].append(a)
            if (q == 'Exterior Color'):
                myList[4].append(a)
            if (q == 'Interior Color'):
                myList[5].append(a)
            if (q == 'Engine'):
                myList[6].append(a)
            if (q == 'Drive Type'):
                myList[7].append(a)
            if (q == 'Fuel Type'):
                myList[8].append(a)
            if (q == 'Transmission'):
                myList[9].append(a)
            if (q == 'Cab Type'):
                myList[10].append(a)

                       
        myList[11].append(price.get_text())
        
        for feat in features:
            f = feat.get_text()
            list1.append(f)
    
        if 'Moonroof' in list1:
            Moonroof.append(1)
        else:
            Moonroof.append(0)
        if 'Backup Camera' in list1:
            Backup_Camera.append(1)
        else:
            Backup_Camera.append(0)
        if 'Parking Sensors' in list1:
            Parking_Sensors.append(1)
        else:
            Parking_Sensors.append(0)
        if 'Bluetooth' in list1:
            Bluetooth.append(1)
        else:
            Bluetooth.append(0)
        if 'Navigation' in list1:
            Navigation.append(1)
        else:
            Navigation.append(0)
        if 'Front Heated Seats' in list1:
            Front_Heated_Seats.append(1)
        else:
            Front_Heated_Seats.append(0)
                
        car_info = cars_soup.find("div",{"data-test": "vdpBreadcrumbs"})
        info = car_info.find_all("a")
        for i in info:
            s = i.get_text() 
            list2.append(s)
        Brand.append(list2[1])
        Names.append(list2[2])
        Year.append(list2[3])


# In[6]:


button_next_urls = []
for i in range(1,300):
    button_next = soup.find("li", {"data-test": "paginationDirectionalItem"})
    next_page_url = "https://www.truecar.com"+button_next.find("a")["href"]
    button_next_urls.append(next_page_url)
for button in button_next_urls:
    get_attributes()
   
 


# # Handling the Data
# In order to be able to analyze the data further, we needed to convert it to numeric values later in the code. Therefore, we removed unwanted characters such as '$' and ',' . We also dropped columns with irrelevant data to our analysis, and rows with missing data, such as price.

# In[127]:


for i in range(len(Price)):
    Price[i] = Price[i].replace("$", "")
    Price[i] = Price[i].replace(",", "")
    Price[i] = Price[i].replace("No Price Available", "0")


# In[128]:


df = pd.DataFrame(myList)
df2= df.transpose()

df2.columns =['Names','Brand','Year', 'Style', 'Exterior_Color', 'Interior_Color', 'Engine', 'Drive_Type', 'Fuel_Type', 'Transmission','Cab_Type', 'Price', 
         'Moonroof', 'Backup_Camera', 'Parking_Sensors', 'Bluetooth', 'Navigation', 'Front_Heated_Seats']
df2.to_csv(("/Users/edenbenzikri/projcsv.csv"))


# In[129]:


data = pd.read_csv("/Users/edenbenzikri/projcsv.csv" , names = ['Names','Brand','Year', 'Style', 'Exterior_Color', 'Interior_Color', 'Engine', 'Drive_Type', 'Fuel_Type', 'Transmission','Cab_Type', 'Price', 
         'Moonroof', 'Backup_Camera', 'Parking_Sensors', 'Bluetooth', 'Navigation', 'Front_Heated_Seats'] , header = None)
#replacing non-numeric cab type values to NaN for dropna
data['Cab_Type'] = pd.to_numeric(df2['Cab_Type'], errors='coerce')
data['Price'] = pd.to_numeric(df2['Price'], errors='ignore')
data['Year'] = pd.to_numeric(df2['Year'], errors='ignore')

data.loc[data.Price < 200, "Price"] = np.nan
data.dropna(axis=1 , thresh=5 , inplace= True)
data.dropna(axis=0 , how='any' , inplace= True)

data.to_csv("/Users/edenbenzikri/finalData.csv", header=None)
data.head()


# # basic statistics
# In this section we ran some basic statistics functions to better understand our data with more information. The information contains the number of columns, column names,  the number of cells in each column (non-null values), the amount of unique values of each column and more.

# In[130]:


data.info()


# In[131]:


data.duplicated().sum()
data[data.duplicated()]
data.drop_duplicates()

data.describe(include = 'O')



# # EDA & Visualiztion
# In this section ,we used the Seaborn data visualization library to display the data so we can understand it better.
# Each visualization has a title which describes what it presents.

# In[132]:


brand_prices = data.groupby('Brand')['Price'].mean().sort_values(ascending=False)
top_brands = brand_prices.head(10)
sns.barplot(x=top_brands.index, y=top_brands.values,palette="ch:.25")
plt.title("Top 10 Brands by Average Price", fontsize = 8)
plt.xlabel("Brand", fontsize = 6)
plt.ylabel("Average Price", fontsize = 6)
plt.tick_params(axis='both', which='major', labelsize=6)
plt.show()


# In[212]:


var = "Brand"
plt.figure(figsize=(7, 4))
sns.catplot(x=var, kind="count", palette="ch:.25", height=5, aspect=1, data=data );
plt.title("Brands and the Counts", fontsize = 8)
plt.xticks(rotation=90);


# In[213]:


plt.figure(figsize=[7,4])
sns.set_palette("ch:.25")
sns.distplot(data['Price'],kde=True )
plt.title("The Denisity of Cars by its Prices", fontsize = 8)



# In[135]:


#The autopct parameter is used to show the percentage on each section of the chart


sns.set_palette("ch:.25")
fuel_counts = data['Fuel_Type'].value_counts()
ax = fuel_counts.plot.pie(autopct='%1.1f%%',fontsize = 6, startangle=90, figsize = (7,4))
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

plt.title("Fuel Type Distribution" , fontsize = 10)
plt.show()


# In[136]:


sns.set_palette("ch:.25")
transmission_counts = data['Transmission'].value_counts()
ax = transmission_counts.plot.pie(autopct='%1.1f%%',fontsize = 6, startangle=90, figsize = (7,4))
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
plt.title("Transmission Distribution" , fontsize = 10)
plt.show()


# In[137]:


sns.set_palette("ch:.25")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

data.groupby(["Drive_Type"])["Price"].mean().plot(kind="bar", ax=ax1)
ax1.set_xlabel("Drive Type")
ax1.set_ylabel("Average Price")
ax1.set_title("Bar plot of Drive Type vs. Price")

sns.boxplot(x='Drive_Type', y='Price', data=data)
ax2.set_xlabel("Drive Type")
ax2.set_ylabel("Price")
ax2.set_title("Box plot of Drive Type vs. Price")
plt.show()


# In[190]:


from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

sns.set_palette("ch:.25")

plt.figure(figsize=(50, 12))

plt.subplot(1,3,1)
plt1 = data.Brand.value_counts().plot(kind="bar")
plt.title('Brands Histogram', fontsize=50)
plt1.set(xlabel = 'Brand', ylabel='Frequency of company')
plt1.set_yticklabels(plt1.get_yticklabels(), fontsize=35)
plt1.set_xticklabels(plt1.get_xticklabels(), fontsize=35)
plt1.xaxis.label.set_size(50)
plt1.yaxis.label.set_size(50)


plt.subplot(1,3,2)
plt1 = data.Fuel_Type.value_counts().plot(kind="bar")
plt.title('Fuel Type Histogram', fontsize=50)
plt1.set(xlabel = 'Fuel_Type', ylabel='Frequency of fuel type')
plt1.xaxis.label.set_size(50)
plt1.yaxis.label.set_size(50)
plt1.set_yticklabels(plt1.get_yticklabels(), fontsize=35)
plt1.set_xticklabels(plt1.get_xticklabels(), fontsize=35)


plt.subplot(1,3,3)
plt1 = data.Style.value_counts().plot(kind="bar")
plt.title('Car Type Histogram', fontsize=50)
plt1.set(xlabel = 'Type', ylabel='Frequency of type')
plt1.set_yticklabels(plt1.get_yticklabels(), fontsize=35)
plt1.set_xticklabels(plt1.get_xticklabels(), fontsize=35)
plt1.xaxis.label.set_size(50)
plt1.yaxis.label.set_size(50)

plt.show()


# In[139]:


sns.set_palette("ch:.25")

#how the numeric features distribution on the price
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
features = ['Moonroof', 'Backup_Camera', 'Parking_Sensors', 'Bluetooth', 'Navigation', 'Front_Heated_Seats']
for i, feature in enumerate(features):
    sns.boxplot(x=feature, y='Price', data=data, ax=axs[i])


# In[162]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data_encoded = data.copy()

non_numeric_cols = data.columns[data.dtypes == object]

for col in non_numeric_cols:
    le.fit(data_encoded[col])
    
    data_encoded[col] = le.transform(data_encoded[col])


# In[164]:


data_encoded.head()


# # Finding the correlations

# A heat map is a graphical representation of data where the values of a matrix are represented as colors.
# 
# Heat maps are useful because they can quickly reveal patterns and trends in large data sets, as well as areas of high and low concentration. 
# 
# Using the heat map we looked for relavent correlations to our research.
# 

# In[226]:


corr = data_encoded.corr()
sns.set_palette("ch:.25")
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr,annot=True,ax=ax)
ax.set_title("Correlation between all data ")

plt.show()


# In[165]:


corr_matrix = data_encoded.corr()
print(corr_matrix)


# In the follwing code we revealed the relavent correlations for us.
# We looked for correlations that are higher than 0.5 and also related to the price.
# In this way, we figured which elements from our data effects the Car Prices the most.

# In[166]:


# Find the highest correlations among all data categories
correlations = []
for i in range(corr.values.shape[0]):
    for j in range(corr.values.shape[1]):
        if i < j and corr.values[i][j] >= 0.5:
            correlations.append(corr.values[i][j])
tuple_arr = []
for i in range(corr.values.shape[0]):
    for j in range(corr.values.shape[1]):
        if i < j and corr.values[i][j] >= 0.5:
            tuple_arr.append((i,j))
print("Correlations: ", correlations)
print("Pairs found: ", tuple_arr)

cols_for_corr = ['Names','Brand','Year', 'Style', 'Exterior_Color', 'Interior_Color', 'Engine', 'Drive_Type', 'Fuel_Type', 'Transmission', 'Price', 
         'Moonroof', 'Backup_Camera', 'Parking_Sensors', 'Bluetooth', 'Navigation', 'Front_Heated_Seats']
corr_sort = np.argsort(correlations)
corr_pairs = []
for n_correlation in corr_sort:
    col_x, col_y = tuple_arr[n_correlation]
    if cols_for_corr[col_x] == 'Price' or cols_for_corr[col_y] == 'Price':
        col_name_x, col_name_y = cols_for_corr[col_x], cols_for_corr[col_y]
        print(f"Correlation between '{col_name_x}' and '{col_name_y}' is {correlations[n_correlation]}")
        corr_pairs.append((col_name_x,col_name_y))
        
x_cols = ['Price']
y_cols = []
for p1,p2 in corr_pairs:
    if p1 != 'Price':
        y_cols.append(p1)
    elif p2 != 'Price':
        y_cols.append(p2)
        
x = d[x_cols]
y = d[y_cols]


# # Machine Learning
# # Models
# In this sction we used Regression Models for the Machine Learning.
# This was needed because the element we were checking is continues numeric variable,

# In[193]:


X = data_encoded.drop(columns=["Price"])
y = data_encoded["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print("Test score:", score)

# evaluate the model using cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)

mean_cv_score = np.mean(cv_scores)
print("Mean of cross-validation scores:", mean_cv_score)

std_cv_score = np.std(cv_scores)
print("Standard deviation of cross-validation scores:", std_cv_score)


# The test score is a measure of how well the model generalizes to new, unseen data. In this case, the decision tree model achieves a high test score of 0.989, indicating that it is able to predict the price of products accurately on new, unseen data.
# 
# The cross-validation scores are: [0.775, 0.882, 0.693, 0.991, 0.944]. These values represent the R^2 coefficient of determination, a measure of how well the model fits the data, with 1.0 being a perfect fit and 0 indicating that the model does not explain any of the variability in the data.
# 
# The mean of the cross-validation scores is 0.857, which indicates that the model performs well overall, but not as well as the test score.
# 
# n summary, the decision tree model seems to be a good fit for the data, with a high test score and reasonably good cross-validation scores.

# In[221]:


train_x, test_x, train_y, test_y = train_test_split(d.drop("Price", axis=1), data_encoded["Price"], test_size=0.2, random_state=0)
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(train_x, train_y)
scores = cross_val_score(model, train_x, train_y, cv=5)
score = model.score(test_x, test_y)
print("Test score:", score)
print("Cross-validation scores:", scores)
print("Mean of cross-validation scores:", cv_scores.mean())
print("Standard deviation of cross-validation scores:", cv_scores.std())


# Overall, the results suggest that the random forest model is a good fit for the data, with a high R^2 score on the test set. However, the low mean of the cross-validation scores and high variability across folds suggest that the model may be overfitting the training data, and that its performance may not generalize well to new, unseen data.

# In[224]:


X = data_encoded.drop(columns=["Price"])
y = data_encoded["Price"]

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # reducing test_size to 0.1
model.fit(X, y)

scores = cross_val_score(model, X, y, cv=5)
score = model.score(X_test, y_test)
y_pred = model.predict(X_test)

print("Test score:", score)
print("Cross-validation scores:", scores)
print("Mean of cross-validation scores:" ,scores.mean())
print("Standard deviation of cross-validation scores::" ,scores.std())



# Overall, the model appears to be reasonably effective at predicting prices based on the other features in the dataset.
