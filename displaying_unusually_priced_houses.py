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
    df['cluster'] = KMeans(n_clusters=3).fit_predict(data)
    df.index = data.index
    df['principal_feature1'] = data[0]
    df['principal_feature2'] = data[1]

    def get_distance_point(data, model):
        distance = pd.Series()
        for i in range(0, len(data)):
            Xa = np.array(data.loc[i])
            Xb = model.cluster_centers_[model.labels_[i]]
            distance.at[i] = np.linalg.norm(Xa - Xb)
        return distance

    # Set the outlier scale
    outliers_fraction = 0.05

    # To get the distance from each point to the cluster center
    distance = get_distance_point(data, KMeans(n_clusters=3).fit(data))

    # Calculate the number of outliers based on the outliers_fraction and set a threshold for outliers
    threshold = distance.nlargest(int(outliers_fraction * len(distance))).min()

    # Judging whether it is an abnormal value according to the threshold
    df['anomaly_KMeans'] = (distance >= threshold).astype(int)

    # Train isolation forest
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data)
    df['anomaly_IsolationForest'] = pd.Series(model.predict(data))

    # Train OneClassSVM
    model = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)
    model.fit(data)
    df['anomaly_OneClassSVM'] = pd.Series(model.predict(data))

    # Display exception ID
    print('Display abnormal ID (KMeans):', df.loc[df['anomaly_KMeans'] == 1]['id'].values)
    print('Display abnormal ID (IsolationForest):', df.loc[df['anomaly_IsolationForest'] == -1]['id'].values)
    print('Display abnormal ID (OneClassSVM):', df.loc[df['anomaly_OneClassSVM'] == -1]['id'].values)
    print('Display abnormal ID (Simultaneously):',
          df.loc[(df.anomaly_KMeans == 1) & (df.anomaly_IsolationForest == -1) & (df.anomaly_OneClassSVM == -1)][
              'id'].values)

decoy_property_detection('d_bukken_test.csv')