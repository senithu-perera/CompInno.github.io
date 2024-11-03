import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load the dataset
file_path = 'csv2.csv'  # Update with your actual file path
data = pd.read_csv(file_path)

# Selecting relevant numeric columns for clustering
numerical_features = ['Rooms', 'Price', 'Distance', 'Bathroom', 'Car', 'Landsize', 
                      'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
data_clustering = data[numerical_features]

# Handling missing values by imputing with the mean of each column
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_clustering)

# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Running KMeans with the optimal number of clusters
optimal_clusters = 3
final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = final_kmeans.fit_predict(data_scaled)

# Plotting the clusters spatially with color intensity based on Price
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['Longtitude'], data['Lattitude'], c=data['Price'], cmap='plasma', alpha=0.6)
plt.colorbar(scatter, label="Price (Darker = Higher Price)")
plt.title("Spatial Distribution of Properties with Price Intensity")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

#this Section of the code will only be used for saving the model after creation this commented out until needed
# Save the KMeans model and any preprocessor (scaler, imputer)
# with open('kmeans_model.pkl', 'wb') as model_file:
#     pickle.dump(final_kmeans, model_file)

# with open('scaler.pkl', 'wb') as scaler_file:
#     pickle.dump(scaler, scaler_file)

# with open('imputer.pkl', 'wb') as imputer_file:
#     pickle.dump(imputer, imputer_file)