# Automatically identify decoy priced houses

## problem
Complaints about decoy priced houses are constantly occurring. How to detect decoy priced houses automatically?

## objective
Create an algorithm that can automatically identify decoy priced houses from "property information data".

## method
1. K-means
2. Isolation Forest
3. OneClassSVM

## summary
In this code, I'm using K-means、Isolation Forest and One-Class SVM to detect outliers in the dataset. After detecting the outliers, we are removing them from the dataset and visualizing the clusters and outliers using the seaborn library. Finally, we are returning the cleaned dataset. The function takes a CSV file as input and returns a pandas dataframe. The example usage of the function reads the data from a CSV file called 'example_data.csv'.
