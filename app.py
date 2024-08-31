import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer

def intracluster_analysis(data, cluster_column):
    # Group data by cluster
    cluster_groups = data.groupby(cluster_column)
    
    # Iterate over each cluster
    for cluster, group_data in cluster_groups:
        st.write(f"\nIntracluster Analysis for Cluster {cluster}:")
        
        # Describe cluster characteristics
        cluster_stats = group_data.describe()
        st.write(cluster_stats)
        
        # Visualize cluster characteristics
        fig,ax = plt.subplots()
        ax = sns.boxplot(data=group_data.drop(cluster_column, axis=1), orient='h')
        plt.title(f"Distribution of Features in Cluster {cluster}")
        plt.xlabel("Feature Value")
        plt.ylabel("Feature")
        st.pyplot(fig)   





st.title("Uploaded CSV File")
st.sidebar.title("File Upload")
try:
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type = ['csv'])
    dataframe = pd.read_csv(uploaded_file)
    dataframe = dataframe.dropna(axis=1, how='all')
    if uploaded_file is not None:
        st.write(dataframe)   
except:
    st.write("Please upload a CSV file to continue")

if uploaded_file is not None:
    st.title("Select Desired Columns")
    try:
        columns = dataframe.columns
        kmean_columns = st.multiselect('Select the column',columns)
        if kmean_columns is not None:
            df_features_selected = dataframe[kmean_columns]
            st.write(df_features_selected)
        else:
            st.write("Select some columns to proceed")

    except:
        st.write("Select some columns to proceed")

    ## KDA
    scaler = MinMaxScaler()     
    df_features_selected = df_features_selected.dropna()##line added
    numerical_columns = df_features_selected.select_dtypes(include=['int32','int64','float64', 'float32']).columns
    categorical_columns = df_features_selected.select_dtypes(include=['object']).columns

    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df_features_selected[col] = label_encoder.fit_transform(df_features_selected[col])

    scaled_kmeans_df = df_features_selected.copy()
    for c in numerical_columns:
        scaled_kmeans_df[c] = scaler.fit_transform(df_features_selected[[c]])


    ##PCA
    from sklearn.decomposition import PCA
    if len(kmean_columns) > 2:
        pca = PCA(n_components=2)
        kmeans_pca = pca.fit_transform(scaled_kmeans_df)
        kmeans_pca_df = pd.DataFrame(kmeans_pca)
        kmeans_pca_df.rename(columns={0: 'col1'}, inplace=True)
        kmeans_pca_df.rename(columns={1: 'col2'}, inplace=True)
    else:
        kmeans_pca_df = scaled_kmeans_df      

    ################################Elbow

    inertia_lst = []
    silhouette_score_lst = []
    sil_dict = {}
    for i in range(2,20):
        kmeans = KMeans(n_clusters=i, init="k-means++")
        kmeans.fit(kmeans_pca_df)
        score = silhouette_score(kmeans_pca_df, kmeans.labels_, metric = 'euclidean')
        silhouette_score_lst.append(score)
        inertia_lst.append(kmeans.inertia_)
        sil_dict[i] = score      

    max_key = max(sil_dict, key=sil_dict.get)

    kmeans = KMeans(n_clusters=max_key, init="k-means++")
    cluster_labels = kmeans.fit_predict(kmeans_pca_df)

    # Add cluster labels to DataFrame
    kmeans_pca_df['kmeans_Cluster'] = cluster_labels

    kmenas_cluster_counts = pd.Series(kmeans.labels_).value_counts()


    ploting_cols = kmeans_pca_df.columns.tolist()


    #######################

    st.title("Choose the Clustering Method")
    user_menu = st.selectbox(
        'Select an Option',
        ('K-Means Clustering','Hierarchical Clustering','DBSCAN Clustering')
    )




    ###############################
    if user_menu == 'K-Means Clustering':
        st.header("K-means Clustering")
        fig,ax = plt.subplots()
        ax = plt.scatter(kmeans_pca_df[ploting_cols[0]], kmeans_pca_df[ploting_cols[1]], c = kmeans.labels_, cmap='viridis', s=50, alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        cbar = plt.colorbar()
        cbar.set_label('Cluster')
        cbar.set_ticks(range(len(set(kmeans.labels_))))
        cbar.set_ticklabels(range(len(set(kmeans.labels_))))
        st.pyplot(fig)

        intracluster_analysis(kmeans_pca_df, 'kmeans_Cluster')

    ################################hir

    # Perform hierarchical clustering
    if user_menu == 'Hierarchical Clustering':
        agg_clustering = AgglomerativeClustering(n_clusters=max_key)
        agg_clusters = agg_clustering.fit_predict(kmeans_pca_df)

        # Plot the clusters
        st.header("Hierarchical Clustering")
        fig,ax = plt.subplots()
        ax = plt.scatter(kmeans_pca_df['col1'], kmeans_pca_df['col2'], c=agg_clusters, cmap='viridis', s=50, alpha=0.7)
        plt.title('Hierarchical Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster')
        st.pyplot(fig)

        kmeans_pca_df['Hierar_Cluster'] = agg_clusters


        from scipy.cluster.hierarchy import dendrogram, linkage

        # Perform hierarchical clustering
        linked = linkage(kmeans_pca_df, method='ward')
        st.header("Dendogram")
        # Plot the dendrogram
        fig,ax = plt.subplots(figsize=(40, 20))
        ax = dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram', fontsize = 25)
        plt.xlabel('Index', fontsize = 25)
        plt.ylabel('Distance', fontsize = 25)
        st.pyplot(fig)

        intracluster_analysis(kmeans_pca_df, 'Hierar_Cluster')


    #################################DB
    if user_menu == 'DBSCAN Clustering':
        from sklearn.cluster import DBSCAN
        from sklearn.datasets import make_moons

        dbscan = DBSCAN(eps=0.2, min_samples=5)
        dbscan_clusters = dbscan.fit_predict(kmeans_pca_df)

        kmeans_pca_df['DB_Cluster'] = dbscan_clusters


        st.header("DBSCAN Clustering")
        fig,ax = plt.subplots()
        ax = plt.scatter(kmeans_pca_df[ploting_cols[0]], kmeans_pca_df[ploting_cols[1]], c=dbscan_clusters, cmap='viridis', s=50, alpha=0.7)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster')
        st.pyplot(fig)

        ##
        

        intracluster_analysis(kmeans_pca_df, 'DB_Cluster')

#######################################

