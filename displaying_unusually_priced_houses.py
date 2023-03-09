import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def decoy_property_detection(csv):
    df_original = pd.read_csv(csv)
    df = df_original.copy()

    # Reduce the impact of extremely high prices and long time to station on classification
    df = df.drop(df[(df['price_per_tsubo'] > df['price_per_tsubo'].quantile(q=0.75))].index)
    df = df.drop(df[(df['minute_to_station'] > df['minute_to_station'].quantile(q=0.95))].index)

    labelencoder = LabelEncoder()
    df['land_shape'] = labelencoder.fit_transform(df['land_shape'])
    df['frontal_road_direction'] = labelencoder.fit_transform(df['frontal_road_direction'])
    df['frontal_road_kind'] = labelencoder.fit_transform(df['frontal_road_kind'])

    data = df[['price_per_tsubo', 'minute_to_station', 'land_space', 'land_shape']]
    X = data.values

    # Standardization processing, mean is 0, standard deviation is 1
    X_std = StandardScaler().fit_transform(X)
    data = pd.DataFrame(X_std)

    # Reduce feature dimension to 2
    pca = PCA(n_components=3)
    data = pca.fit_transform(data)

    # Normalize the 2 new features after dimensionality reduction
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)

    # Train KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    df['cluster'] = kmeans.predict(data)
    df.index = data.index
    df['principal_feature1'] = data[0]
    df['principal_feature2'] = data[1]

    def get_distance_point(data, cluster_assignments, model):
        distance = pd.Series()
        for i in range(0, len(data)):
            Xa = np.array(data.loc[i])
            Xb = model.cluster_centers_[cluster_assignments[i]]
            distance.at[i] = np.linalg.norm(Xa - Xb)
        return distance

    # Set the outlier scale
    outliers_fraction = 0.05

    # To get the cluster assignments for each data point
    cluster_assignments = kmeans.predict(data)

    # To get the distance from each point to the assigned cluster center
    distance = get_distance_point(data, cluster_assignments, kmeans)

    # Sort the distances and get the top outlier_fraction points
    distance_sorted = distance.sort_values(ascending=False)
    n_outliers = int(outliers_fraction * len(distance))
    outliers_indices = distance_sorted[:n_outliers].index

    # Identify outliers using IsolationForest
    iso = IsolationForest(contamination=outliers_fraction)
    iso.fit(data)
    outlier_preds = iso.predict(data)
    outlier_indices = np.where(outlier_preds == -1)[0]

    # Identify outliers using One-Class SVM
    svm = OneClassSVM(nu=outliers_fraction)
    svm.fit(data)
    svm_preds = svm.predict(data)
    svm_outlier_indices = np.where(svm_preds == -1)[0]

    # Merge outlier indices from all 3 methods
    outlier_indices = list(set(outliers_indices) | set(outlier_indices) | set(svm_outlier_indices))

    # Add a new column to indicate whether a property is an outlier or not
    df['outlier'] = 0
    df.loc[outlier_indices, 'outlier'] = 1

    return df

import seaborn as sns

df_csv = 'd_bukken_test.csv'

# Call the decoy_property_detection function to get the cleaned dataset with outlier column
df = decoy_property_detection(df_csv)

# Remove the outliers from the dataset
df = df[df['outlier'] == 0]

# Visualize the clusters and outliers using seaborn
sns.scatterplot(data=df, x='principal_feature1', y='principal_feature2', hue='cluster', style='outlier')
