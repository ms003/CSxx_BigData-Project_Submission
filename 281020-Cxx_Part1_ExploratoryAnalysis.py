"""
Created on Wed Oct 14 15:31:16 2020

@author: mihalis
"""
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
import pandas as pd
from sklearn.preprocessing import scale
import seaborn as sns

# In[2]:


print("read data")
br_data = pd.read_csv("br_data.csv")

print("show first and last 10 rows")
print(br_data.head(10))  
print(br_data.tail(10))
print("show shape of data")
print(br_data.shape)
print("columns")
print(br_data.columns)
print("checking for nulls")
print(br_data.isnull())  # check if there are missing values.Returns Boolean [False=no null, True=null]
print(br_data.isnull().any())  # returns which columns are intact and which not

# In[3]:


print("Dropping ID column")

br_data = br_data.drop("id", axis=1)
br_data0 = br_data.iloc[:, [0, 21, 22,23,24,25,26,27,28,29,30]]
print(br_data0.columns)
print(br_data0.shape)
print("Dropped column")

# In[4]:


print("Correlation Analysis")

print("Heatmap inidcating correlation between variables")
print (br_data0.head())
corr = br_data0.corr()
sns.heatmap(corr)
plt.show()






print("Grouping data by diagnosis")
group_by = br_data0.groupby(['diagnosis'])
# print(Metrics)
print("describe data grouped by diagnosis")
print(group_by.describe())
print("convert raw data to pandas dataframe")


from scipy import stats
from scipy.stats import ttest_ind

print("Metrics")
metrics0 = group_by.agg(["mean", "min", "max", "median"])



print("Sorting the data by Diagnosis and then splitting them into two lists")

br_data = br_data0.sort_values(by="diagnosis")
diagnosis_B = br_data.iloc[:357, :]
diagnosis_M = br_data.iloc[357:, :]
big_list = [diagnosis_B, diagnosis_M]

print(diagnosis_B.head())
print(diagnosis_M.tail())



print("Statistical Analysis: T-Test")

print("T-test: metric,statistic,pvalue")

ttest_results = []
for (columnName, columnData) in br_data.iteritems():
  if columnName != 'diagnosis':
    current_result = stats.ttest_ind(diagnosis_B[columnName], diagnosis_M[columnName], equal_var=False)
    ttest_results.append([columnName,current_result.statistic,current_result.pvalue])
    print("{0},{1},{2}".format(columnName, current_result.statistic, current_result.pvalue))

ttest_results = pd.DataFrame(ttest_results, columns = ["metric", "statistic", "p-value"])
ttest_results1 = ttest_results.sort_values(by="p-value")

print(ttest_results.head(31))


# In[6]:


print("Radius-mean Comparison")
plt.figure(figsize=(5, 5), dpi=50)
plt.hist(diagnosis_B["concave points_worst"], bins=40, color="cyan", label="benign", histtype="step")
plt.hist(diagnosis_M["concave points_worst"], bins=40, color="darkviolet", label="malignant", histtype="step")
plt.legend()
plt.text(60, .025, "Comparison")
plt.xlabel("Radius_mean")
plt.ylabel("No of samples")
plt.title("Radius_mean Comparison")
plt.show()



print("Histograms showing the feture distrbution between benign and malignant sample")
for (columnName, columnData) in br_data.iteritems():
        if(columnName != 'diagnosis'):
                plt.hist(diagnosis_B[columnName], bins=40,  color= "cyan", label= "benign", histtype= "step")
                plt.hist(diagnosis_M[columnName], bins=40, color= "darkviolet", label= "malignant", histtype= "step")
                plt.legend()
                plt.xlabel(columnName)
                plt.ylabel("No of samples")
                plt.title("Sample distribution for benign and malignant samples")
                plt.figure(figsize=(1,1))
                plt.show()





print("Violin plot showing tendencybetween Benign and Malignant sample")
plt.figure(figsize=(5, 5))
sns.violinplot(x='diagnosis', y='concave points_worst', data=br_data0)
plt.show()

plt.figure(figsize=(5, 5))
sns.violinplot(x='diagnosis', y='perimeter_worst', data=br_data0)
plt.show()

plt.figure(figsize=(5, 5))
sns.violinplot(x='diagnosis', y='radius_worst', data=br_data0)
plt.show()

plt.figure(figsize=(5, 5))
sns.violinplot(x='diagnosis', y='symmetry_worst', data=br_data0)
plt.show()

plt.figure(figsize=(5, 5))
sns.violinplot(x='diagnosis', y='fractal_dimension_worst', data=br_data0)
plt.show()




from scipy.stats import pearsonr

print("Scatter plot for linear correlation between the features")

results = []
for col1 in br_data0.columns:
    for col2 in br_data0.columns:
        if col1 == "diagnosis" or col2 == "diagnosis": 
            continue
        sns.scatterplot(x=col1, y=col2, data=br_data0, hue="diagnosis",legend='full')
        plt.show()
        corr = pearsonr(br_data0[col1], br_data0[col2])
        results.append([col1, col2,  corr[0], corr[1]])
        print(col1, col2, corr[0], corr[1], sep=",")
results = pd.DataFrame(results, columns = ["fig1", "fig2", "corr", "p-val"])
print(results)   
 
   
    


