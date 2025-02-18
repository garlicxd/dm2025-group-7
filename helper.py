import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_wine
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import time
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.cluster.hierarchy import dendrogram, linkage

def dataframe_to_markdown(df):

    if df.empty:
        return ""

    try:
        # Get the column headers
        headers = [str(col) for col in df.columns]  # Convert column names to strings

        # Create the header row
        header_row = "| " + " | ".join(headers) + " |"

        # Create the separator row
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        # Create the data rows
        data_rows = []
        for _, row in df.iterrows():
            data_row = "| " + " | ".join([f"{item:.2f}" if isinstance(item, (int, float)) else str(item) for item in row.values]) + " |"
            data_rows.append(data_row)

        # Combine all rows into the Markdown table string
        markdown_table = "\n".join([header_row, separator_row] + data_rows)

        return markdown_table

    except Exception as e:  # Handle potential errors (e.g., type errors)
        print(f"Error converting DataFrame to Markdown: {e}")
        return ""

def generate_distance_matrix(data, metric='euclidean'):

    if isinstance(data, pd.DataFrame):
        data_values = data.values  # Convert to NumPy array if it's a DataFrame
    elif isinstance(data, np.ndarray):
        data_values = data
    else:
        print("Invalid input data. Please provide a NumPy array or Pandas DataFrame.")
        return None

    distance_matrix = pairwise_distances(data_values, metric=metric)

    # Create a Pandas DataFrame for better readability and labeling
    distance_df = pd.DataFrame(distance_matrix, index=data.index if isinstance(data, pd.DataFrame) else range(data_values.shape[0]), columns=data.index if isinstance(data, pd.DataFrame) else range(data_values.shape[0]))

    return distance_df

# load data
if True:
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target

    RANDOM_SEED = 132414
    toy = df.sample(n=15, random_state=RANDOM_SEED)
    toy = toy.reset_index(drop=True)

# dendogram
if False:
    first = 'ash'
    last = 'hue'
    X_kmeans = toy[[first, last]]

    X = X_kmeans.values

    linked = linkage(X, 'complete')  # 'complete' specifies complete linkage

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=toy.index + 1,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

# generate distance matrix
if True:
    X_kmeans = toy[['ash', 'hue']]

    distance_matrix = generate_distance_matrix(X_kmeans, metric='euclidean')

    if distance_matrix is not None:
        print("Distance Matrix:")
        print(dataframe_to_markdown(distance_matrix))
