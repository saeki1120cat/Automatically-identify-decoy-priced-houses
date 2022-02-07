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
So far, we have detected anomalies using three different methods.Because our anomaly detection is unsupervised learning. After building the models, we have no idea how well it is doing as we have nothing to test it against. Hence, the results of those methods need to be tested in the field before placing them in the critical path.
