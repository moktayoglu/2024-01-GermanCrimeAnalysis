import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from fastdtw import fastdtw


def pca(data, pca_n, kmeans_n):
    # Standardize the data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(data)
    print(data.shape)
    # Apply PCA
    pca = PCA(n_components=pca_n)  # Adjust the number of components as needed
    principal_components = pca.fit_transform(scaled_df)

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

    # Extract loadings
    loadings = pca.components_

    # Create a DataFrame with the loadings
    loadings_df = pd.DataFrame(loadings, columns=data.columns, index=[f'PC{i+1}' for i in range(pca_n)])

    # Print the explained variance ratio
    print(pca.explained_variance_ratio_)

    # Eigenvalues
    eigenvalues = pca.explained_variance_

    # Apply Kaiser Criterion
    n_components = sum(eigenvalue > 1 for eigenvalue in eigenvalues)

    print(f"Number of components to keep based on Kaiser Criterion: {n_components}")

    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_.cumsum(), 
            marker='o', linestyle='--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    print()

    # Display the loadings
    print(loadings_df)

    # Display the loadings as an sns.heatmap where the heatmap fits the figure size
    plt.figure(figsize=(40, 4))
    ax = sns.heatmap(loadings_df, annot=True, cmap='RdBu')
    # sns heatmap x row names should be rotated 45 degrees and font size should be 10
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    #plt.show()

    print(data.keys())

    # Do clustering on the loadings_df
    from sklearn.cluster import KMeans

    # Specify the number of clusters
    kmeans = KMeans(n_clusters=kmeans_n)

    loadings_df = loadings_df.T
    # Fit the data
    clusters = kmeans.fit_predict(loadings_df)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    loadings_df['Cluster'] = clusters
    #sort table by cluster
    loadings_df = loadings_df.sort_values(by=['Cluster'])

    print(loadings_df['Cluster'])

def kmeeans(data):
    # Specify the number of clusters
    kmeans = KMeans(n_clusters=kmeans_n, init='k-means++')

    # Fit the data
    clusters = kmeans.fit_predict(data.T)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    #put the clusters and the corresponding column names into a dictionary
    clusters_dict = {}
    for i in range(len(clusters)):
        clusters_dict[data.keys()[i]] = clusters[i]
    #print out the dictionary sorted by the clusters
    print(sorted(clusters_dict.items(), key=lambda x: x[1]))

def MixGauss(data):
    # Looking at mixture of gaussian model of the data
    # Create a list of silhouette scores for different number of clusters
    silhouette_scores = []
    for n in range(2, 11):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(data)
        silhouette_scores.append(silhouette_score(data, gmm.predict(data)))
    # Plot the silhouette scores
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by Number of Clusters')
    plt.show()

def spectral():
    # Specify the number of clusters
    spectral = SpectralClustering(n_clusters=kmeans_n, affinity='nearest_neighbors', assign_labels='kmeans')

    # Fit the data
    clusters = spectral.fit_predict(data.T)

    # Get the cluster centers
    cluster_centers = spectral.cluster_centers_

    #put the clusters and the corresponding column names into a dictionary
    clusters_dict = {}
    for i in range(len(clusters)):
        clusters_dict[data.keys()[i]] = clusters[i]
    #print out the dictionary sorted by the clusters
    print(sorted(clusters_dict.items(), key=lambda x: x[1]))

pca_n = 5
kmeans_n = 6
normalize = True

# Specify the file path
file_path = 'dat/data_crime_clear.xlsx'
# Read the Excel file
data = pd.read_excel(file_path)
data = data.set_index('Type of criminal offence').transpose()
# drop column 'Criminal offences, total'
data = data.drop('Criminal offences, total', axis=1)
# Drop the column where there is NaN value
data = data.dropna(axis=1)

#convert column values to float if possible, if not, change it to np.inf inside the table
for i in data.keys():
    for j in data[i]:
        try:
            data[i] = data[i].astype(float)
        except:
            data[i] = data[i].replace(j, np.NaN)

# if there is an np.NaN in the column, replace it with the average of the column - apart from np.NaN - in the table
for i in data.keys():
    data[i] = data[i].fillna(data[i].mean())

# if normalize is True, normalize the data so that it is between 0 and 1
if normalize:
    for i in data.keys():
        data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())


#print out the average for every column and sort them by the average
#print(data.mean().sort_values())

print(data.head())
print(np.asarray(data.T.iloc[0]).shape)

data = data.T

# Compute the DTW distance matrix using fastdtw
n = len(data)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        a = np.asarray(data.iloc[i])
        b = np.asarray(data.iloc[j])
        distance, _ = fastdtw(a, b,  dist=2)
        distance_matrix[i, j] = distance_matrix[j, i] = distance

# Perform Hierarchical Clustering
Z = linkage(distance_matrix, method='ward')

# Plot Dendrogram
plt.figure(figsize=(6, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('Distance')

# put the x axis labels in a variable
x_labels = plt.gca().get_xmajorticklabels()
indexes = []
for i in range(len(x_labels)):
    indexes.append(int(x_labels[i].get_text()))

# put the data.keys() in a list and sort it by the indexes
data_keys = list(data.T.keys())
data_keys = [data_keys[i] for i in indexes]
print(data_keys)
# set the x axis labels to the sorted data.keys() so that the labels have enough space
plt.gca().set_xticklabels(data_keys)
plt.xticks(rotation=90)


#print data.keys() with their index
for i in range(len(data.T.keys())):
    print(i, data.T.keys()[i])

plt.tight_layout()
plt.show()


