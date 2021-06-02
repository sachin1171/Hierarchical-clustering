#Hierarchical Clustering
############################Problem 1######################################
library(data.table)
library(readxl)
EastWestAirlines <- read_excel(file.choose(), sheet = "data")
#View(EastWestAirlines)
colnames(EastWestAirlines)
ncol(EastWestAirlines)
sub_EastWestAirlines <- EastWestAirlines[,2:12]
norm_airline <- scale(sub_EastWestAirlines)
# Hirerachical CLustering
distanct_airline <- dist(norm_airline,method="euclidean")
str(distanct_airline)
airline_clust <- hclust(distanct_airline, method = "complete")
#plot(airline_clust, hang = -1)
group_airline <- cutree(airline_clust,k=5)
EastWestAirlines_New <- cbind(EastWestAirlines, group_airline)
setnames(EastWestAirlines_New, 'group_airline', 'group_hclust')
aggregate(EastWestAirlines_New[,2:12],by= list(EastWestAirlines_New$group_hclust), FUN = mean)
# install.packages("cluster")
library(cluster)
# Using Clara function(Clustering for Large Applications) to find cluster
xcl <- clara(norm_airline,5) #Using Centroid
clusplot(xcl)
#using Partition Arround Medoids to find cluster
xpm <- pam(norm_airline,5) # Using Medoids
clusplot(xpm)



###############################Problem 2######################################         
library(data.table)
crime_data = read.csv(file.choose())
ncol(crime_data)
crime_data_sub <- crime_data[,2:5]
# Normalized the data

norm_crime_data_sub <- scale(crime_data_sub)
# calculating distance

d <- dist(norm_crime_data_sub, method = "euclidean")
str(d)              
crime_cluse <- hclust(d, method = "complete")
plot(crime_cluse, hang=-1)
rect.hclust(crime_cluse,plot(crime_cluse,hang=-1),k=4,border="red")
groups <- cutree(crime_cluse,k=4)
crime_data_final <- cbind(crime_data, groups)
aggregate(crime_data_final[,2:6], by=list(crime_data_final$groups), FUN = mean)
#as per summary we can say group 2 have the higher rate of crime.
#########################################Problem 3###########################
install.packages("readxl")
library(readxl)

telco_data <- read_excel("C:/Users/usach/Desktop/14.Clustering-Hierarchical Clustering/Telco_customer_churn.xlsx")
my_telco_data <-  telco_data[ , c(-1,-2,-3)]
sum(is.na(my_telco_data))

summary(my_telco_data)

dups <- duplicated(my_telco_data)
sum(dups)
my_telco_data <- my_telco_data[!dups , ]
str(my_telco_data)

boxplot(my_telco_data["Tenure in Months"])
boxplot(my_telco_data["Avg Monthly Long Distance Charges"])
boxplot(my_telco_data["Avg Monthly GB Download"])
boxplot(my_telco_data["Monthly Charge"])
boxplot(my_telco_data["Total Charges"])
boxplot(my_telco_data["Total Refunds"])
boxplot(my_telco_data["Total Extra Data Charges"])
boxplot(my_telco_data["Total Long Distance Charges"])
boxplot(my_telco_data["Total Revenue"])

qunt_Avg_Monthly_GB_Download <- quantile(my_telco_data$"Avg Monthly GB Download" , probs = c(.25 , .75))
winso_Avg_Monthly_GB_Download <- quantile(my_telco_data$"Avg Monthly GB Download" , probs = c(.01 , .93) , na.rm = TRUE)
H_Avg_Monthly_GB_Download <- 1.5*IQR(my_telco_data$"Avg Monthly GB Download" , na.rm = TRUE)
my_telco_data$"Avg Monthly GB Download"[my_telco_data$"Avg Monthly GB Download"<(qunt_Avg_Monthly_GB_Download[1]-H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[1]
my_telco_data$"Avg Monthly GB Download"[my_telco_data$"Avg Monthly GB Download">(qunt_Avg_Monthly_GB_Download[2]+H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[2]
boxplot(my_telco_data$"Avg Monthly GB Download")

qunt_Total_Refunds <- quantile(my_telco_data$"Total Refunds" , probs = c(.25 , .75))
winso_Total_Refunds <- quantile(my_telco_data$"Total Refunds" , probs = c(.01 , .92) , na.rm = TRUE)
H_Total_Refunds <- 1.5*IQR(my_telco_data$"Total Refunds" , na.rm = TRUE)
my_telco_data$"Total Refunds"[my_telco_data$"Total Refunds"<(qunt_Total_Refunds[1]-H_Total_Refunds)] <- winso_Total_Refunds[1]
my_telco_data$"Total Refunds"[my_telco_data$"Total Refunds">(qunt_Total_Refunds[2]+H_Total_Refunds)] <- winso_Total_Refunds[2]
boxplot(my_telco_data$"Total Refunds")

qunt_Total_Extra_Data_Charges <- quantile(my_telco_data$"Total Extra Data Charges" , probs = c(.25 , .75))
winso_Total_Extra_Data_Charges <- quantile(my_telco_data$"Total Extra Data Charges" , probs = c(.01 , .85) , na.rm = TRUE)
H_Total_Extra_Data_Charges <- 1.5*IQR(my_telco_data$"Total Extra Data Charges" , na.rm = TRUE)
my_telco_data$"Total Extra Data Charges"[my_telco_data$"Total Extra Data Charges"<(qunt_Total_Extra_Data_Charges[1]-H_Total_Extra_Data_Charges)] <- winso_Total_Extra_Data_Charges[1]
my_telco_data$"Total Extra Data Charges"[my_telco_data$"Total Extra Data Charges">(qunt_Total_Extra_Data_Charges[2]+H_Total_Extra_Data_Charges)] <- winso_Total_Extra_Data_Charges[2]
boxplot(my_telco_data$"Total Extra Data Charges")

qunt_Total_Long_Distance_Charges <- quantile(my_telco_data$"Total Long Distance Charges" , probs = c(.25 , .75))
winso_Total_Long_Distance_Charges <- quantile(my_telco_data$"Total Long Distance Charges" , probs = c(.01 , .95) , na.rm = TRUE)
H_Total_Long_Distance_Charges <- 1.5*IQR(my_telco_data$"Total Long Distance Charges" , na.rm = TRUE)
my_telco_data$"Total Long Distance Charges"[my_telco_data$"Total Long Distance Charges"<(qunt_Total_Long_Distance_Charges[1]-H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[1]
my_telco_data$"Total Long Distance Charges"[my_telco_data$"Total Long Distance Charges">(qunt_Total_Long_Distance_Charges[2]+H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[2]
boxplot(my_telco_data$"Total Long Distance Charges")

qunt_Total_Revenue <- quantile(my_telco_data$"Total Revenue" , probs = c(.25 , .75))
winso_Total_Revenue <- quantile(my_telco_data$"Total Revenue" , probs = c(.01 , .99) , na.rm = TRUE)
H_Total_Revenue <- 1.5*IQR(my_telco_data$"Total Revenue" , na.rm = TRUE)
my_telco_data$"Total Revenue"[my_telco_data$"Total Revenue"<(qunt_Total_Revenue[1]-H_Total_Revenue)] <- winso_Total_Revenue[1]
my_telco_data$"Total Revenue"[my_telco_data$"Total Revenue">(qunt_Total_Revenue[2]+H_Total_Revenue)] <- winso_Total_Revenue[2]
boxplot(my_telco_data$"Total Revenue")

install.packages("ggplot2")
library(ggplot2)
qplot(my_telco_data$"Total Revenue",my_telco_data$"Total Long Distance Charges" ,data = my_telco_data,geom = "point")
qplot(my_telco_data$"Total Charges",my_telco_data$"Total Refunds",data = my_telco_data,geom = "point")


install.packages("fastDummies")
library(fastDummies)

my_telco_data_dummy <- dummy_cols(my_telco_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

norm_telco_data <- scale(my_telco_data_dummy)
norm_telco_data <- as.data.frame(norm_telco_data)
summary(norm_telco_data)

dist_telco_data <- dist(norm_telco_data , method = "euclidean")
fit_telco_data <- hclust(dist_telco_data , method = "complete")
plot(fit_telco_data , hang = -1)

clust_telco_data <- cutree(fit_telco_data , k=3)
rect.hclust(fit_telco_data , k=3 , border = "green")

top_three_telco <- as.matrix(clust_telco_data)
final_telco_data <- data.frame(top_three_telco , telco_data)

aggregate(telco_data[,1:30], by = list(final_telco_data$top_three_telco), FUN = mean)
install.packages("readr")
library(readr)
write_csv(final_telco_data , "final_crime_data.csv")
getwd()

#daisy()
library(cluster)
telco_dist <- daisy(norm_telco_data , metric = "gower" )
summary(telco_dist)
telco_dist1 <- as.matrix(telco_dist)

fit_telco_data <- hclust(telco_dist , method = "complete")
plot(fit_telco_data , hang = -1)
clust_telco_data <- cutree(fit_telco_data , k = 3)
rect.hclust(fit_telco_data ,k=3 , border = "red")
###############################program 4###############################################
install.packages("readr")
library(readr)

auto_data <- read_csv("C:/Users/usach/Desktop/14.Clustering-Hierarchical Clustering/AutoInsurance.csv")
new_auto_data <- auto_data[-1]

sum(is.na(new_auto_data))

summary(new_auto_data)

dup<- duplicated(new_auto_data)
sum(dup)
new_auto_data <- new_auto_data[!dup , ]
str(new_auto_data)

boxplot(new_auto_data$`Customer Lifetime Value`)
boxplot(new_auto_data$Income)
boxplot(new_auto_data$`Monthly Premium Auto`)
boxplot(new_auto_data$`Months Since Last Claim`)
boxplot(new_auto_data$`Months Since Policy Inception`)
boxplot(new_auto_data$`Total Claim Amount`)

qunt_Customer_Lifetime_Value <- quantile(new_auto_data$`Customer Lifetime Value` , probs = c(.25 , .75))
winso_Customer_Lifetime_Value <- quantile(new_auto_data$`Customer Lifetime Value` , probs = c(.01 , .90) , na.rm = TRUE)
H_Customer_Lifetime_Value <- 1.5*IQR(new_auto_data$`Customer Lifetime Value` , na.rm = TRUE)
new_auto_data$`Customer Lifetime Value`[new_auto_data$`Customer Lifetime Value`<(qunt_Customer_Lifetime_Value[1]-H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[1]
new_auto_data$`Customer Lifetime Value`[new_auto_data$`Customer Lifetime Value`>(qunt_Customer_Lifetime_Value[2]+H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[2]
boxplot(new_auto_data$`Customer Lifetime Value`)

qunt_Monthly_Premium_Auto <- quantile(new_auto_data$`Monthly Premium Auto` , probs = c(.25 , .75))
winso_Monthly_Premium_Auto <- quantile(new_auto_data$`Monthly Premium Auto` , probs = c(.01 , .95) , na.rm = TRUE)
H_Monthly_Premium_Auto <- 1.5*IQR(new_auto_data$`Monthly Premium Auto` , na.rm = TRUE)
new_auto_data$`Monthly Premium Auto`[new_auto_data$`Monthly Premium Auto`<(qunt_Monthly_Premium_Auto[1]-H_Monthly_Premium_Auto)] <- winso_Monthly_Premium_Auto[1]
new_auto_data$`Monthly Premium Auto`[new_auto_data$`Monthly Premium Auto`>(qunt_Monthly_Premium_Auto[2]+H_Monthly_Premium_Auto)] <- winso_Monthly_Premium_Auto[2]
boxplot(new_auto_data$`Monthly Premium Auto`)

qunt_Total_Claim_Amount <- quantile(new_auto_data$`Total Claim Amount` , probs = c(.25 , .75))
winso_Total_Claim_Amount <- quantile(new_auto_data$`Total Claim Amount` , probs = c(.01 , .95) , na.rm = TRUE)
H_Total_Claim_Amount <- 1.5*IQR(new_auto_data$`Total Claim Amount` , na.rm = TRUE)
new_auto_data$`Total Claim Amount`[new_auto_data$`Total Claim Amount`<(qunt_Total_Claim_Amount[1]-H_Total_Claim_Amount)] <- winso_Total_Claim_Amount[1]
new_auto_data$`Total Claim Amount`[new_auto_data$`Total Claim Amount`>(qunt_Total_Claim_Amount[2]+H_Total_Claim_Amount)] <- winso_Total_Claim_Amount[2]
boxplot(new_auto_data$`Total Claim Amount`)


install.packages("ggplot2")
library(ggplot2)
qplot(new_auto_data$Income,new_auto_data$"Monthly Premium Auto" ,data = new_auto_data,geom = "point")
qplot(new_auto_data$"Total Claim Amount",new_auto_data$"Months Since Policy Inception",data = new_auto_data,geom = "point")


install.packages("fastDummies")
library(fastDummies)

my_auto_data_dummy <- dummy_cols(new_auto_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

norm_auto_data <- scale(my_auto_data_dummy)
summary(norm_auto_data)

dist_auto_data <- dist(norm_auto_data , method = "euclidean")

fit_auto_data <- hclust(dist_auto_data , method = "complete")
plot(fit_auto_data , hang = -1)

clust_auto_data <- cutree(fit_auto_data , k=3)
rect.hclust(fit_auto_data , k=3 , border = "green")

auto_top_three <- as.matrix(clust_auto_data)
final_auto_data <- data.frame(auto_top_three , new_auto_data)

aggregate(final_auto_data[, 1:24], by = list(final_auto_data$auto_top_three), FUN = mean)

write_csv(final_auto_data , "final_auto_data.csv")
getwd()

