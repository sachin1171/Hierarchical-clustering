#hirecial clustering 1
################################Problem - 1###########################################
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import excel
airlines = pd.read_excel("C:/Users/usach/Desktop/14.Clustering-Hierarchical Clustering/EastWestAirlines.xlsx", header = 0)
type(airlines)
pd.DataFrame(airlines)
airlines = pd.read_excel("C:/Users/usach/Desktop/14.Clustering-Hierarchical Clustering/EastWestAirlines.xlsx", sheet_name="data")
airlines.dtypes
def norm_func(i) :
    x = (i - i.min())/(i.max()) - (i.min())
    return x
airlines_norm = norm_func(airlines.iloc[:,1:10])
airlines_norm.head()
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(airlines_norm, method = "complete" , metric = "euclidean")
plt.figure(figsize = (15,5)) ; plt.title("H_Clustering_Dendrogram");plt.xlabel("index");plt.ylabel("Distance")
sch.dendrogram(z)
plt.show()
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete',affinity = "euclidean").fit(airlines_norm)
cluster_labels  = pd.Series(h_complete.labels_)
airlines["Cluster"] = cluster_labels
airlines = airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
airlines.Cluster.value_counts()
airlines.iloc[:,2:13].groupby(airlines.Cluster).mean()
airlines_C_average = airlines.iloc[:,2:12].groupby(airlines.Cluster).mean()
awards = airlines.iloc[:,12].groupby(airlines.Cluster).sum()
airlines_C_average["Awards"] = awards

# Cluster 0 clients have the least demand for flights and travel the least, giving them so many free flights is not encouraging them to fly more as thir demand of trabel is less, the free flight system here should be replaces with discounts on the flights booked and not give unnecessary free flights.
# Cluster 1 clients do travle a lot and have the highest number of flights booked in the past 12 months, free flights or awards can be increased to encourage them to fly with the airlines.
# Cluster 2 customers are the highest users of the frequent flier credit card and are the oldest customers. These customers should be given a an between the free flight ticket award or discount system to chose from.

###################################Problem-2##############################
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
crime = pd.read_csv("C:/Users/usach/Desktop/14.Clustering-Hierarchical Clustering/crime_data.csv")
crime.head()
def norm_func(i) :
    x = (i - i.min())/(i.max() - i.min())
    return x
crime_norm = norm_func(crime.iloc[:, 1:5 ])
crime_norm.head()
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(crime_norm, method = "complete" , metric = "euclidean")
plt.figure(figsize = (15,5)) ; plt.title("H_Clustering_Dendrogram");plt.xlabel("index");plt.ylabel("Distance")
sch.dendrogram(z)
plt.show()
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete',affinity = "euclidean").fit(crime_norm)
cluster_labels  = pd.Series(h_complete.labels_)
crime_norm["Cluster"] = cluster_labels
crime["Cluster"] = cluster_labels
crime = crime.iloc[ : , [5,0,1,2,3,4]]
cluster = crime.iloc[:,2:6].groupby(crime.Cluster).mean()


# Custer 2 states seems to be the safest places of all the rest clusters. It is also the safest place for the females too.
# Cluster 1 state comes after the cluster 2 as the population is also high and crime rate in comparision with Cluster 0 and 3 is the least.
# Cluster 0  is the most unsafe state for the population. for the current population in the cluster, it seems like crime is very common, especially rape and assault cases.
# Cluster 3 also not safe to stay however is safer than cluster 0. Cluster 3 can be determined as the most unsafe place for the female population.


########################################Problem-3###############################################
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np

telco_data = pd.read_excel("C:/Users/usach/Desktop/14.Clustering-Hierarchical Clustering/Telco_customer_churn.xlsx")
telco_data.drop(['Count' , 'Quarter'] , axis=1 , inplace=True)

new_telco_data = pd.get_dummies(telco_data)

dupis = telco_data.duplicated()
sum(dupis)

telco_data = telco_data.drop_duplicates()

from sklearn.preprocessing import  OneHotEncoder

OH_enc = OneHotEncoder()

new_telco_data2 = pd.DataFrame(OH_enc.fit_transform(telco_data).toarray())

from sklearn.preprocessing import  LabelEncoder
L_enc = LabelEncoder()
telco_data['Referred a Friend'] = L_enc.fit_transform(telco_data['Referred a Friend'])
telco_data['Offer'] = L_enc.fit_transform(telco_data['Offer'])
telco_data['Phone Service'] = L_enc.fit_transform(telco_data['Phone Service'])
telco_data['Multiple Lines'] = L_enc.fit_transform(telco_data['Multiple Lines'])
telco_data['Internet Service'] = L_enc.fit_transform(telco_data['Internet Service'])
telco_data['Internet Type'] = L_enc.fit_transform(telco_data['Internet Type'])
telco_data['Online Security'] = L_enc.fit_transform(telco_data['Online Security'])
telco_data['Online Backup'] = L_enc.fit_transform(telco_data['Online Backup'])
telco_data['Device Protection Plan'] = L_enc.fit_transform(telco_data['Device Protection Plan'])
telco_data['Premium Tech Support'] = L_enc.fit_transform(telco_data['Premium Tech Support'])
telco_data['Streaming TV'] = L_enc.fit_transform(telco_data['Streaming TV'])
telco_data['Streaming Movies'] = L_enc.fit_transform(telco_data['Streaming Movies'])
telco_data['Streaming Music'] = L_enc.fit_transform(telco_data['Streaming Music'])
telco_data['Unlimited Data'] = L_enc.fit_transform(telco_data['Unlimited Data'])
telco_data['Contract'] = L_enc.fit_transform(telco_data['Contract'])
telco_data['Paperless Billing'] = L_enc.fit_transform(telco_data['Paperless Billing'])
telco_data['Payment Method'] = L_enc.fit_transform(telco_data['Payment Method'])

telco_data.isna().sum()

telco_data.columns

seaborn.boxplot(telco_data["Tenure in Months"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Avg Monthly Long Distance Charges"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Avg Monthly GB Download"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Monthly Charge"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Charges"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Total Refunds"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Extra Data Charges"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Long Distance Charges"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Revenue"]);plt.title("Boxplot");plt.show()

plt.scatter(telco_data["Tenure in Months"] , telco_data["Total Extra Data Charges"])
plt.scatter(telco_data["Monthly Charge"] , telco_data["Avg Monthly Long Distance Charges"])
plt.scatter(telco_data["Total Long Distance Charges"] , telco_data["Total Revenue"])

IQR = telco_data["Avg Monthly GB Download"].quantile(0.75) - telco_data["Avg Monthly GB Download"].quantile(0.25)
L_limit_Avg_Monthly_GB_Download = telco_data["Avg Monthly GB Download"].quantile(0.25) - (IQR * 1.5)
H_limit_Avg_Monthly_GB_Download = telco_data["Avg Monthly GB Download"].quantile(0.75) + (IQR * 1.5)
telco_data["Avg Monthly GB Download"] = pd.DataFrame(np.where(telco_data["Avg Monthly GB Download"] > H_limit_Avg_Monthly_GB_Download , H_limit_Avg_Monthly_GB_Download ,
                                    np.where(telco_data["Avg Monthly GB Download"] < L_limit_Avg_Monthly_GB_Download , L_limit_Avg_Monthly_GB_Download , telco_data["Avg Monthly GB Download"])))
seaborn.boxplot(telco_data["Avg Monthly GB Download"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Refunds"].quantile(0.75) - telco_data["Total Refunds"].quantile(0.25)
L_limit_Total_Refunds = telco_data["Total Refunds"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Refunds = telco_data["Total Refunds"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Refunds"] = pd.DataFrame(np.where(telco_data["Total Refunds"] > H_limit_Total_Refunds , H_limit_Total_Refunds ,
                                    np.where(telco_data["Total Refunds"] < L_limit_Total_Refunds , L_limit_Total_Refunds , telco_data["Total Refunds"])))
seaborn.boxplot(telco_data["Total Refunds"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Extra Data Charges"].quantile(0.75) - telco_data["Total Extra Data Charges"].quantile(0.25)
L_limit_Total_Extra_Data_Charges = telco_data["Total Extra Data Charges"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Extra_Data_Charges = telco_data["Total Extra Data Charges"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Extra Data Charges"] = pd.DataFrame(np.where(telco_data["Total Extra Data Charges"] > H_limit_Total_Extra_Data_Charges , H_limit_Total_Extra_Data_Charges ,
                                    np.where(telco_data["Total Extra Data Charges"] < L_limit_Total_Extra_Data_Charges , L_limit_Total_Extra_Data_Charges , telco_data["Total Extra Data Charges"])))
seaborn.boxplot(telco_data["Total Extra Data Charges"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Long Distance Charges"].quantile(0.75) - telco_data["Total Long Distance Charges"].quantile(0.25)
L_limit_Total_Long_Distance_Charges = telco_data["Total Long Distance Charges"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Long_Distance_Charges = telco_data["Total Long Distance Charges"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Long Distance Charges"] = pd.DataFrame(np.where(telco_data["Total Long Distance Charges"] > H_limit_Total_Long_Distance_Charges , H_limit_Total_Long_Distance_Charges ,
                                    np.where(telco_data["Total Long Distance Charges"] < L_limit_Total_Long_Distance_Charges , L_limit_Total_Long_Distance_Charges , telco_data["Total Long Distance Charges"])))
seaborn.boxplot(telco_data["Total Long Distance Charges"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Revenue"].quantile(0.75) - telco_data["Total Revenue"].quantile(0.25)
L_limit_Total_Revenue = telco_data["Total Revenue"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Revenue = telco_data["Total Revenue"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Revenue"] = pd.DataFrame(np.where(telco_data["Total Revenue"] > H_limit_Total_Revenue , H_limit_Total_Revenue ,
                                    np.where(telco_data["Total Revenue"] < L_limit_Total_Revenue , L_limit_Total_Revenue , telco_data["Total Revenue"])))
seaborn.boxplot(telco_data["Total Revenue"]);plt.title('Boxplot');plt.show()

def std_fun(i):
    x = (i-i.mean()) / (i.std())
    return (x)

telco_data_norm = std_fun(new_telco_data)

str(telco_data_norm)

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage

telco_single_linkage = linkage(telco_data_norm , method="single" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_single_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_complete_linkage = linkage(telco_single_linkage , method="complete" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_single_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_average_linkage = linkage(telco_complete_linkage , method="average" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_average_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_centroid_linkage = linkage(telco_data_norm , method="centroid" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_centroid_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

from sklearn.cluster import  AgglomerativeClustering

telco_single = AgglomerativeClustering(n_clusters=3 , linkage="single" , affinity="euclidean").fit(telco_data_norm)
cluster_telco_single = pd.Series(telco_single.labels_)
telco_data["cluster"] = cluster_telco_single

telco_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete" , affinity="euclidean").fit(telco_data_norm)
cluster_telco_complete = pd.Series(telco_complete.labels_)
telco_data["cluster"] = cluster_telco_complete

telco_average = AgglomerativeClustering(n_clusters=3 , linkage="average" , affinity="euclidean").fit(telco_data_norm)
cluster_telco_average = pd.Series(telco_average.labels_)
telco_data["cluster"] = cluster_telco_average

telco_ward = AgglomerativeClustering(n_clusters=3 , linkage="ward" ,  affinity="euclidean").fit(telco_data_norm)
cluster_telco_ward = pd.Series(telco_ward.labels_)
telco_data["cluster"] = cluster_telco_ward

telco_data.iloc[: , 0:29].groupby(telco_data.cluster).mean()

import os

telco_data.to_csv("final_telco_data.csv" , encoding="utf-8")

os.getcwd()

import  gower
from scipy.cluster.hierarchy import fcluster , dendrogram
gowers_matrix = gower.gower_matrix(telco_data)
gowers_linkage = linkage(gowers_matrix)
gcluster = fcluster(gowers_linkage , 3 , criterion = 'maxclust')
dendrogram(gowers_linkage)
telco_data["cluster"] = gcluster
telco_data.iloc[: , 0:29].groupby(telco_data.cluster).mean()

import os

telco_data.to_csv("final2_telco_data.csv" , encoding="utf-8")

os.getcwd()
################################Problem 4######################################
import pandas as pd
import seaborn
import numpy as np
import matplotlib.pyplot as plt

auto_data = pd.read_csv("C:/Users/usach/Desktop/14.Clustering-Hierarchical Clustering/AutoInsurance.csv")

auto_data.drop(['Customer'] , axis= 1 , inplace = True)

new_auto_data = auto_data.iloc[ : ,1:]

new_auto_data.isna().sum()

new_auto_data.columns

duplis = new_auto_data.duplicated()
sum(duplis)
dummy_auto_data = pd.get_dummies(new_auto_data)

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

auto_data_norm = norm_func(dummy_auto_data)

from sklearn.cluster import AgglomerativeClustering

auto_single = AgglomerativeClustering(n_clusters=3 , linkage="single" , affinity="euclidean").fit(auto_data_norm)
cluster_auto_single = pd.Series(auto_single.labels_)
new_auto_data["cluster"] = cluster_auto_single

auto_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete" , affinity="euclidean").fit(auto_data_norm)
cluster_auto_complete = pd.Series(auto_complete.labels_)
new_auto_data["cluster"] = cluster_auto_complete

auto_average = AgglomerativeClustering(n_clusters=3 , linkage="average" , affinity="euclidean").fit(auto_data_norm)
cluster_auto_average = pd.Series(auto_average.labels_)
new_auto_data["cluster"] = cluster_auto_average

auto_ward = AgglomerativeClustering(n_clusters=3 , linkage="ward" , affinity="euclidean").fit(auto_data_norm)
cluster_auto_ward = pd.Series(auto_ward.labels_)
new_auto_data["cluster"] = cluster_auto_ward

new_auto_data.iloc[: ,:23].groupby(new_auto_data.cluster).mean()

import os

new_auto_data.to_csv("final_auto_data.csv" , encoding="utf-8")

os.getcwd()
