#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings 
import re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import entropy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from wordcloud import STOPWORDS
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
warnings.filterwarnings('ignore')

def load_data(file_path):
    return pd.read_csv(file_path)


# EDA
def explore_data(df):
    """Prints dataset information and basic statistics."""
    print("Shape of dataset:", df.shape)
    print("\nFirst few rows:\n", df.head())
    print("\nDataset Info:\n")
    df.info()
    print("\nSummary Statistics:\n", df.describe())
    print("\nSummary of Null value:",df.isnull().sum())


def get_unique_values(df, column_name):
    """Returns and prints the unique values in a specified column."""
    unique_values = df[column_name].unique()
    print(f"Unique values in '{column_name}':\n", unique_values)
    return unique_values

def count_unique_values(df, column_name):
    """Returns and prints the number of unique values in a specified column."""
    unique_count = df[column_name].nunique()
    print(f"Number of unique values in '{column_name}':", unique_count)
    return unique_count


def extract_rater_demographics(df, demographic_fields):
    """Extracts unique raters with their demographic information."""
    rater_demographics = df[demographic_fields].drop_duplicates(subset=['rater_id'])
    return rater_demographics.iloc[:, 1:]


def merge_demographics(clustered_df, original_df, demographic_fields):
    """Merges demographic information into the clustered dataframe."""
    demographics_df = original_df.groupby('rater_id')[demographic_fields].first().reset_index()
    clustered_df['rater_id'] = demographics_df['rater_id'].values
    for field in demographic_fields:
        clustered_df[field] = demographics_df[field].values
    return clustered_df


# KModes Clustering based on raters demographics
def perform_kmodes_clustering(df, max_k=15):
    """Performs KModes clustering and determines the optimal number of clusters using the Elbow method."""
    categorical_data = df.astype(str)
    cost = []
    K = range(1, max_k)
    
    for k in K:
        kmodes = KModes(n_clusters=k, init='Huang', random_state=42)
        kmodes.fit(categorical_data)
        cost.append(kmodes.cost_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(K, cost, marker='o')
    plt.title('Elbow Method for Optimal k (KModes)')
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig("elbow_method_kmodes_demographics.pdf", format="pdf", bbox_inches="tight")

    
    plt.show()


def apply_kmodes_clustering(df, num_clusters=6):
    """Applies KModes clustering on the categorical dataset."""
    categorical_data = df.astype(str)
    kmodes = KModes(n_clusters=num_clusters, init='Huang', random_state=21)
    labels = kmodes.fit_predict(categorical_data)
    categorical_data['Cluster'] = labels
    return categorical_data


def visualize_cluster_counts(df):
    """Visualizes the count of records in each cluster."""
    cluster_counts = df['Cluster'].value_counts()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
    plt.title('Cluster Sizes using Kmodes (clustering on demographic features)')
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Number of Records', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig("cluster_counts_kmodes_demographics.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def visualize_categorical_distribution(df):
    """Visualizes the distribution of categorical variables across clusters."""
    for col in df.columns[:-1]:
        freq = df.groupby('Cluster')[col].value_counts(normalize=True).unstack().fillna(0)
        freq.plot(kind='bar', figsize=(10, 6), colormap='viridis', width=0.7)
        plt.title(f'Frequency of {col} in each Cluster', fontsize=14)
        plt.xlabel('Cluster', fontsize=14)
        plt.ylabel('Proportion', fontsize=14)
        plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.savefig(f"kmodes_categorical_distribution_{col}.pdf", format="pdf", bbox_inches="tight")

        
        plt.show()


def compute_within_demographic_cluster_disagreement(clustered_df):
    """Compute entropy for each question within each demographic cluster."""
    question_cols = [col for col in clustered_df.columns if "_Q" in col]
    
    results = []
    for cluster_id, group in clustered_df.groupby("Cluster"):
        cluster_entropy = group[question_cols].apply(compute_entropy, axis=0)  # Compute entropy per question
        cluster_avg_entropy = cluster_entropy.mean()
        
        results.append(pd.DataFrame({
            "Cluster": cluster_id,
            "Entropy": cluster_avg_entropy
        }).reset_index())
    
    return pd.concat(results, ignore_index=True)


def visualize_demographic_disagreement(entropy_df):
    """Visualize entropy (disagreement) across demographic clusters."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Cluster", y="Entropy", data=entropy_df, palette="viridis")
    plt.xlabel("Demographic Cluster")
    plt.ylabel("Disagreement (Entropy)")
    plt.title("Variation in Rating Behaviors Within Demographic Clusters")
    plt.show()


if __name__ == "__main__":
    file_path = "Desktop/cs-rit/last_term/capstone/pythonProject/diverse_safety_adversarial_dialog_350.csv"
    demographic_fields = ['rater_id','rater_gender', 'rater_race', 'rater_age', 'rater_education']
    
    df = load_data(file_path)
    rater_demographics = extract_rater_demographics(df, demographic_fields)
    print(rater_demographics.shape)  # Should be (123, 4)
    print(rater_demographics.head())
    
    perform_kmodes_clustering(rater_demographics, max_k=15)
    clustered_data = apply_kmodes_clustering(rater_demographics, num_clusters=6)
    visualize_cluster_counts(clustered_data)
    visualize_categorical_distribution(clustered_data) 


# Find the most disagreement item within and across the clusters

# Entropy is a measure of uncertainty or disorder in a system. If all raters in a cluster agree on a response, entropy is low ( closer
# to 0 ). If raters in a cluster have diverse opinions, entropy is high. Using Entropy for Disagreement Analysis
# Answers are categorical, variance is not appropriate.
# ANOVA (Analysis of Variance) compares mean values which does not apply to categorical data.
# Entropy measures the level of disagreement.
# Within-Cluster Disagreement: 
# For each item compute ave entropy over all questions
# For each cluster, rank items in order of decreasing average entropy reduction in cluster. Take top-ten (or twenty) items in each cluster 


# Compute Within-Cluster Disagreement
def compute_entropy(series):
    """Computes entropy of categorical responses."""
    # Counts occurrences of each category in the column and normalized them into probabilities
    value_counts = series.value_counts(normalize=True)
    # Compute entropy
    return entropy(value_counts, base=2)  


question_columns = [
    'Q2_harmful_content_medical_legal_financial_or_relationship_advice',
    'Q2_harmful_content_dangerous_content',
    'Q2_harmful_content_derogation_of_boat',
    'Q2_harmful_content_insensitive',
    'Q2_harmful_content_obscene_and_profane',
    'Q2_harmful_content_personal_information',
    'Q2_harmful_content_regulated_goods',
    'Q2_harmful_content_sexually_suggestive_content',
    'Q2_harmful_content_other', 'Q3_bias_targeting_beliefs',
    'Q3_bias_targeting_gender_sexual_orientation',
    'Q3_bias_targeting_inherited_attributes', 'Q3_bias_targeting_status',
    'Q3_bias_other', 'Q4_misinformation', 'Q5_political_affiliation',
    'Q6_policy_guidelines_polarizing_topics',
    'Q6_policy_guidelines_making_endorsement', 'Q6_policy_guidelines_other',
    'Q2_harmful_content_overall', 'Q3_bias_overall',
    'Q6_policy_guidelines_overall', 'Q_overall'
]


def compute_entropy(series):
    """Compute entropy of a categorical variable."""
    value_counts = series.value_counts(normalize=True)
    return -(value_counts * np.log2(value_counts + 1e-10)).sum()


def compute_global_entropy(df):
    """
    Compute the global entropy for each item across all clusters.
    """
    question_cols = [col for col in df.columns if "_Q" in col]
    item_ids = set([col.split("_Q")[0] for col in question_cols])

    global_entropy = {}

    for item_id in item_ids:
        item_question_cols = [col for col in question_cols if col.startswith(item_id + "_Q")]
        if not item_question_cols:
            continue

        # Compute entropy per question across entire dataset
        item_entropy_values = df[item_question_cols].apply(compute_entropy, axis=0)
        
        # Compute average entropy across all questions for this item
        global_entropy[item_id] = item_entropy_values.mean()

    return pd.Series(global_entropy, name="Global_Entropy")


def compute_within_cluster_disagreement(clustered_df, baseline_df):
    """Compute entropy reduction within each cluster and return the top 10 most disagreed items for each cluster."""
    clustered_df = clustered_df.copy()
    
    # Extract item IDs from question columns
    question_cols = [col for col in clustered_df.columns if "_Q" in col]
    item_ids = set([col.split("_Q")[0] for col in question_cols])  # Extract unique item IDs
    
    results = []
    for cluster_id, group in clustered_df.groupby("Cluster"):
        cluster_disagreement = {}
        
        for item_id in item_ids:
            item_question_cols = [col for col in question_cols if col.startswith(item_id + "_Q")]
            
            if not item_question_cols:
                continue 
            
            # Compute entropy per question
            item_entropy_values = group[item_question_cols].apply(compute_entropy, axis=0)
            
            # Compute the average entropy across all questions for this item
            avg_item_entropy = item_entropy_values.mean()
            
            # Compute entropy reduction
            global_entropy = baseline_df.loc[item_id] if item_id in baseline_df.index else 0
            entropy_reduction = global_entropy - avg_item_entropy  # Higher is better
            
            cluster_disagreement[item_id] = entropy_reduction  # Store entropy reduction per item

        # Convert dictionary to DataFrame
        cluster_df = pd.DataFrame(list(cluster_disagreement.items()), columns=["Item_ID", "Entropy_Reduction"])
        cluster_df["Cluster"] = cluster_id
        results.append(cluster_df)
    
    final_df = pd.concat(results).reset_index(drop=True)
    
    return final_df


# Rank items within each cluster and return top N items
def get_top_items_per_cluster(final_df, top_n=3):
    """
    Returns top `top_n` items per cluster based on Entropy_Reduction.
    Higher entropy eduction, most agreed.
    Smaller entropy_reduction, most disagreed.
    """
    top_items = final_df.groupby("Cluster").apply(
        lambda x: x.nlargest(top_n, "Entropy_Reduction")
    ).reset_index(drop=True)
    return top_items

# Customize stopwords
custom_stopwords = STOPWORDS.union({"USER", "LAMDA"})

# Define a custom color function that always returns black
def black_color_func(*args, **kwargs):
    return "black"

def generate_wordclouds_per_cluster(df, text_type="context"):
    """Generate word clouds for each cluster and save them as images."""
    os.makedirs("wordclouds", exist_ok=True) 
    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]

        if text_type == "context":
            text_data = cluster_df["context"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "response":
            text_data = cluster_df["response"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "both":
            context = cluster_df["context"].dropna().astype(str)
            response = cluster_df["response"].dropna().astype(str)
            text_data = context.str.cat(sep=" ") + " " + response.str.cat(sep=" ")
        else:
            raise ValueError("Invalid text_type. Choose from 'context', 'response', or 'both'.")

        wordcloud = WordCloud(width=800, height=400, background_color="white", color_func=black_color_func, stopwords=custom_stopwords).generate(text_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Top Agreed Terms in Cluster {cluster} – {text_type.capitalize()}")

        base_filename = f"wordclouds/cluster_{cluster}_{text_type}"
        # Save as PDF
        plt.savefig(f"{base_filename}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        print(f"WordCloud saved: {base_filename}.png")


# Cluster raters based on their response

# Using K-Means

# In[58]:


def reshape_responses(df):
    """Reshape response data where each row corresponds to a unique rater_id."""
    question_cols = [col for col in df.columns if col.startswith("Q")]
    relevant_cols = ["rater_id", "item_id"] + question_cols
    df_selected = df[relevant_cols]
    
    df_melted = df_selected.melt(id_vars=["rater_id", "item_id"], var_name="question", value_name="response")
    df_melted["item_question"] = df_melted["item_id"].astype(str) + "_" + df_melted["question"]
    
    reshaped_df = df_melted.pivot(index="rater_id", columns="item_question", values="response").reset_index()
    reshaped_df = reshaped_df.drop(columns=[col for col in reshaped_df.columns if "Q1_whole_conversation_evaluation" in col])
    return reshaped_df

def convert_responses_to_numerical(df):
    """Convert categorical responses to numerical values."""
    response_mapping = {'Yes': 1, 'No': -1, 'Unsure': 0}
    return df.replace(response_mapping)

def perform_kmeans_clustering(df, max_k=10):
    """Determine the optimal number of clusters using the Elbow method."""
    inertia = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method – Optimal k")
    plt.show()

def apply_kmeans_clustering(df, num_clusters=5):
    """Apply KMeans clustering on the numerical response dataset."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df)
    return df
    
def visualize_cluster_counts(df, output_path="kmeans_cluster_counts.pdf"):
    """Visualizes the count of records in each KMeans cluster."""
    cluster_counts = df['Cluster'].value_counts()
    # Create a bar plot
    plt.figure(figsize=(4, 3))
    ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, width=0.4, palette="Set2")
    
    # Add labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')
    
    # Add labels and title
    plt.xlabel("Cluster Label", fontsize=7)
    plt.ylabel("Number of Rows", fontsize=7)
    plt.title("Number of Rows in Each Cluster (KMeans)", fontsize=8)

    ax.tick_params(axis='both', labelsize=6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6) 

    ax.grid(False)
    ax.set_axisbelow(True)
    sns.despine() 
    
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

def visualize_demographic_distribution(df, demographic_features, output_dir="demographic_plots_kmeans"):
    """Visualizes how each demographic category is distributed across KMeans clusters and saves the plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="white")

    for feature in demographic_features:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(data=df, x=feature, hue='Cluster', palette="Set2")

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 2),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8, fontweight='bold')
                    
        y_max = max([bar.get_height() for container in ax.containers for bar in container] + [1])
        ax.set_ylim(top=y_max * 1.12)

    
        plt.title(f"Distribution of {feature} across Clusters Using KMeans", fontsize=10)
        plt.xlabel(feature, fontsize=9)
        plt.ylabel("Count", fontsize=9)
        plt.legend(title="Cluster", fontsize=8, title_fontsize=9)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()

        ax.legend(
            title="Cluster", 
            fontsize=8, 
            title_fontsize=9, 
            loc='upper left', 
            bbox_to_anchor=(0, 1),
            borderaxespad=0.1,
            frameon=False)
        # Save as PDF
        plt.savefig(f"{output_dir}/{feature}_kmeans_distribution.pdf", format="pdf", bbox_inches="tight")
        plt.close()
        
def main():
    file_path = "Desktop/cs-rit/last_term/capstone/pythonProject/diverse_safety_adversarial_dialog_350.csv"
    demographic_fields = ['rater_race', 'rater_gender', 'rater_age', 'rater_education']
    df = pd.read_csv(file_path)

    reshaped_df = reshape_responses(df)
    numerical_response_df = convert_responses_to_numerical(reshaped_df)
    
    # Perform KMeans Clustering
    perform_kmeans_clustering(numerical_response_df.iloc[:, 1:], max_k=10)
    clustered_response = apply_kmeans_clustering(numerical_response_df.iloc[:, 1:], num_clusters=5)
    clustered_response = merge_demographics(clustered_response, df, demographic_fields)

    visualize_cluster_counts(clustered_response,output_path="kmeans_cluster_counts.pdf")

    visualize_demographic_distribution(clustered_response, demographic_fields, output_dir="demographic_plots_kmeans")

    # Compute Global Entropy (Baseline)
    global_entropy_df = compute_global_entropy(numerical_response_df)

    # Compute Within-Cluster Disagreement
    within_cluster_disagreement = compute_within_cluster_disagreement(clustered_response, global_entropy_df)

    # Rank Items within Each Cluster based on Entropy Reduction
    top_items_per_cluster = get_top_items_per_cluster(within_cluster_disagreement, top_n=10)

    print("\nTop 10 Most Agreed Items Within Each KMeans Cluster (Entropy Reduction):")
    print(top_items_per_cluster)

    df["item_id"] = df["item_id"].astype(str)
    top_items_per_cluster["Item_ID"] = top_items_per_cluster["Item_ID"].astype(str)

    filtered_df = df.merge(top_items_per_cluster[["Item_ID", "Cluster"]], left_on="item_id", right_on="Item_ID", how="inner")
  
    # Print Conversations
    print("\nTop Agreed Items with Context and Responses:")
    print(filtered_df[["Cluster", "item_id", "context", "response"]].drop_duplicates())
    filtered_df[["Cluster", "item_id", "context", "response"]].drop_duplicates().rename(columns={
          "Cluster": "Cluster",
          "item_id": "Item ID",
          "context": "Context",
          "response": "Response"
      }).to_latex("top_disagreed_items_table_kmeans.tex", index=False, escape=True, longtable=True)


    # Generate Word Clouds for context, response, and both combined
    print("\nGenerating word clouds for user inputs (context)...")
    generate_wordclouds_per_cluster(filtered_df, text_type="context")
    
    print("\nGenerating word clouds for model responses (response)...")
    generate_wordclouds_per_cluster(filtered_df, text_type="response")
    
    print("\nGenerating word clouds for both context and response...")
    generate_wordclouds_per_cluster(filtered_df, text_type="both")


    # Save to CSV for Reference
    top_items_per_cluster.to_csv("top_agreed_items_per_KMeans cluster.csv", index=False, sep="\t", encoding="utf-8-sig")
    filtered_df[["item_id", "context", "response"]].drop_duplicates().to_csv("conversations_for_top_items.csv", index=False, sep="\t", encoding="utf-8-sig")

    print("Results saved: top_agreed_items_per_kmeans_cluster.csv & conversations_for_top_items_kmeans.csv")

  

if __name__ == "__main__":
    main()




# DBSCAN clustering based on raters' response

# In[62]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os

# Customize stopwords
custom_stopwords = STOPWORDS.union({"USER", "LAMDA"})

# Define a custom color function that always returns black
def black_color_func(*args, **kwargs):
    return "black"

def generate_wordclouds_per_cluster(df, text_type="context"):
    """Generate word clouds for each cluster and save them as images."""
    os.makedirs("wordclouds-dbscan", exist_ok=True) 

    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]

        if text_type == "context":
            text_data = cluster_df["context"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "response":
            text_data = cluster_df["response"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "both":
            context = cluster_df["context"].dropna().astype(str)
            response = cluster_df["response"].dropna().astype(str)
            text_data = context.str.cat(sep=" ") + " " + response.str.cat(sep=" ")
        else:
            raise ValueError("Invalid text_type. Choose from 'context', 'response', or 'both'.")

        wordcloud = WordCloud(width=800, height=400, background_color="white", color_func=black_color_func, stopwords=custom_stopwords).generate(text_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Top Agreed Terms in Cluster {cluster} – {text_type.capitalize()}")
        
        base_filename = f"wordclouds/cluster_{cluster}_{text_type}"
        plt.savefig(f"{base_filename}.png", format="png", bbox_inches="tight")
        plt.close()

        print(f"WordCloud saved: {base_filename}.png")


def compute_entropy(series):
    """Compute entropy of a categorical variable."""
    value_counts = series.value_counts(normalize=True)
    return -(value_counts * np.log2(value_counts + 1e-10)).sum()  # Avoid log(0)


def compute_global_entropy(df):
    """
    Compute the global entropy for each item across all clusters.
    """
    question_cols = [col for col in df.columns if "_Q" in col]
    item_ids = set([col.split("_Q")[0] for col in question_cols])

    global_entropy = {}

    for item_id in item_ids:
        item_question_cols = [col for col in question_cols if col.startswith(item_id + "_Q")]
        if not item_question_cols:
            continue

        # Compute entropy per question across entire dataset
        item_entropy_values = df[item_question_cols].apply(compute_entropy, axis=0)
        
        # Compute average entropy across all questions for this item
        global_entropy[item_id] = item_entropy_values.mean()

    return pd.Series(global_entropy, name="Global_Entropy")


def compute_within_cluster_disagreement(clustered_df, baseline_df):
    """
    Compute entropy reduction within each cluster and return the 
    top 10 most disagreed items for each cluster."""
    clustered_df = clustered_df.copy()
    
    # Extract item IDs from question columns
    question_cols = [col for col in clustered_df.columns if "_Q" in col]
    item_ids = set([col.split("_Q")[0] for col in question_cols])
    
    results = []
    for cluster_id, group in clustered_df.groupby("Cluster"):
        cluster_disagreement = {}
        
        for item_id in item_ids:
            item_question_cols = [col for col in question_cols if col.startswith(item_id + "_Q")]
            
            if not item_question_cols:
                continue 
            
            # Compute entropy per question
            item_entropy_values = group[item_question_cols].apply(compute_entropy, axis=0)
            
            # Compute the average entropy across all questions for this item
            avg_item_entropy = item_entropy_values.mean()
            
            # Compute entropy reduction
            global_entropy = baseline_df.loc[item_id] if item_id in baseline_df.index else 0
            entropy_reduction = global_entropy - avg_item_entropy
            
            cluster_disagreement[item_id] = entropy_reduction

        # Convert dictionary to DataFrame
        cluster_df = pd.DataFrame(list(cluster_disagreement.items()), columns=["Item_ID", "Entropy_Reduction"])
        cluster_df["Cluster"] = cluster_id
        results.append(cluster_df)
    
    final_df = pd.concat(results).reset_index(drop=True)
    
    return final_df


# Rank items within each cluster and return top N items
def get_top_items_per_cluster(final_df, top_n=10):
    """
    Returns the top agreed items per cluster based on Entropy_Reduction.
    
    """
    top_items = final_df.groupby("Cluster").apply(
        lambda x: x.nlargest(top_n, "Entropy_Reduction")
    ).reset_index(drop=True)
    return top_items


get_ipython().system('conda install -c conda-forge hdbscan -y')


import hdbscan
print("HDBSCAN successfully imported!")


# PCA assumes continuous, Gaussian-distributed features — not ideal for categorical values like -1/0/1.
# Customize stopwords
custom_stopwords = STOPWORDS.union({"USER", "LAMDA"})


def find_optimal_eps(df, min_samples=10):
    """Finds the optimal eps value for DBSCAN using the k-distance method."""
    X = df.to_numpy()
    
    # Fit Nearest Neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    
    # Sort distances in ascending order
    distances = np.sort(distances[:, -1])  
    
    # Create the plot
    plt.figure(figsize=(10, 6))  
    plt.plot(distances, linewidth=2)  
    plt.xlabel("Data points sorted by distance", fontsize=12)
    plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance", fontsize=12)
    plt.title("DBSCAN Elbow Method for Finding Optimal eps", fontsize=14)
    plt.grid(True)  
    plt.show()


def apply_hdbscan(df, min_cluster_size=5, min_samples=5):
    """Applies HDBSCAN clustering to the numerical response dataset."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    df['Cluster'] = clusterer.fit_predict(df)
    return df


def visualize_hdbscan_clusters(df, output_path="hdbscan_cluster_counts.png"):
    """Visualizes the count of records in each HDBSCAN cluster."""
    cluster_counts = df['Cluster'].value_counts()
    # Create a bar plot
    plt.figure(figsize=(4, 3))
    ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, width=0.4, palette="Set2")
    
    # Add labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')
    
    # Add labels and title
    plt.xlabel("Cluster Label", fontsize=7)
    plt.ylabel("Number of Rows", fontsize=7)
    plt.title("Number of Rows in Each Cluster (HDBSCAN)", fontsize=8)

    ax.tick_params(axis='both', labelsize=6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6) 

    ax.grid(False)
    ax.set_axisbelow(True)
    sns.despine() 
    
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def visualize_hdbscan_demographics(df, demographic_features, output_dir="demographic_plots_hdbscan"):
    """Visualizes how each demographic category is distributed across HDBSCAN clusters and saves the plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="white")

    for feature in demographic_features:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(data=df, x=feature, hue='Cluster', palette="Set2")

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 2),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8, fontweight='bold')
                    
        y_max = max([bar.get_height() for container in ax.containers for bar in container] + [1])
        ax.set_ylim(top=y_max * 1.12)

    
        plt.title(f"Distribution of {feature} across Clusters Using HDBSCAN", fontsize=10)
        plt.xlabel(feature, fontsize=9)
        plt.ylabel("Count", fontsize=9)
        plt.legend(title="Cluster", fontsize=8, title_fontsize=9)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()

        ax.legend(
            title="Cluster", 
            fontsize=8, 
            title_fontsize=9, 
            loc='upper left', 
            bbox_to_anchor=(0, 1),
            borderaxespad=0.1,
            frameon=False)
        # Save as PDF
        plt.savefig(f"{output_dir}/{feature}_hdbscan_distribution.pdf", format="pdf", bbox_inches="tight")
        plt.close()


def generate_wordclouds_per_cluster(df, text_type="context"):
    """Generate word clouds for each cluster and save them as images."""
    os.makedirs("wordclouds", exist_ok=True)

    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]

        if text_type == "context":
            text_data = cluster_df["context"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "response":
            text_data = cluster_df["response"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "both":
            context = cluster_df["context"].dropna().astype(str)
            response = cluster_df["response"].dropna().astype(str)
            text_data = context.str.cat(sep=" ") + " " + response.str.cat(sep=" ")
        else:
            raise ValueError("Invalid text_type. Choose from 'context', 'response', or 'both'.")

        wordcloud = WordCloud(width=800, height=400, background_color="white", color_func=black_color_func, stopwords=custom_stopwords).generate(text_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Top Agreed Terms in Cluster {cluster} – {text_type.capitalize()}")

        base_filename = f"wordclouds/cluster_{cluster}_{text_type}"
        plt.savefig(f"{base_filename}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        print(f"WordCloud saved: {base_filename}.pdf")


def main():
    file_path = "Desktop/cs-rit/last_term/capstone/pythonProject/diverse_safety_adversarial_dialog_350.csv"
    demographic_fields = ['rater_gender', 'rater_race', 'rater_age', 'rater_education']
    
    # Load and reshape
    df = load_data(file_path)
    reshaped_df = reshape_responses(df)
    print("Reshaped:", reshaped_df.shape)

    # Keep numerical question-response matrix
    question_response_df = reshaped_df.copy()  # no 'item_id' needed
    numerical_response_df = convert_responses_to_numerical(question_response_df)
    numerical_response_df = numerical_response_df.iloc[:, 1:]

    top_k = 1000
    variances = numerical_response_df.var()
    selected_cols = variances.sort_values(ascending=False).head(top_k).index
    reduced_df = numerical_response_df[selected_cols]

    # Hamming distance + HDBSCAN
    print("\nCalculating Hamming distance matrix...")
    hamming_dist = pairwise_distances(reduced_df, metric='hamming')


    print("Clustering with HDBSCAN using Hamming distance...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, cluster_selection_epsilon=0.3,alpha=1.0, min_samples=2,metric='precomputed')
    cluster_labels = clusterer.fit_predict(hamming_dist)

    clustered_response = numerical_response_df.copy()
    clustered_response['Cluster'] = cluster_labels

    # Print cluster summary
    print("\nHDBSCAN Cluster Counts (including noise as -1):")
    print(clustered_response['Cluster'].value_counts())


    # Merge back rater demographics
    clustered_response = merge_demographics(clustered_response, df, demographic_fields)

    # Visualize clusters and demographics
    visualize_hdbscan_clusters(clustered_response)
    visualize_hdbscan_demographics(clustered_response, demographic_fields)

    # Compute entropy on raw (unreduced) response matrix
    global_entropy_df = compute_global_entropy(question_response_df)
    within_cluster_disagreement = compute_within_cluster_disagreement(clustered_response, global_entropy_df)

    # Get top entropy-reducing items
    top_items_per_cluster = get_top_items_per_cluster(within_cluster_disagreement, top_n=10)
    print("\nTop 10 Most Disagreed Items Within Each HDBSCAN Cluster (Entropy Reduction):")
    print(top_items_per_cluster)

    # Prepare for word clouds
    df["item_id"] = df["item_id"].astype(str)
    top_items_per_cluster["Item_ID"] = top_items_per_cluster["Item_ID"].astype(str)
    filtered_df = df.merge(top_items_per_cluster[["Item_ID", "Cluster"]],
                           left_on="item_id", right_on="Item_ID", how="inner")

    print("\nTop Disagreed Items with Context and Responses:")
    print(filtered_df[["Cluster", "item_id", "context", "response"]].drop_duplicates())

    # Word Clouds
    print("\nGenerating word clouds for user inputs (context)")
    generate_wordclouds_per_cluster(filtered_df, text_type="context")

    print("\nGenerating word clouds for model responses (response)")
    generate_wordclouds_per_cluster(filtered_df, text_type="response")

    print("\nGenerating word clouds for both context and response")
    generate_wordclouds_per_cluster(filtered_df, text_type="both")

    clustered_response['Cluster'].value_counts().plot(kind='bar', title='Cluster Size Distribution')
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Raters")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

# K-Modes
# Customize stopwords
custom_stopwords = STOPWORDS.union({"USER", "LAMDA"})

def visualize_kmodes_clusters(df, output_path="kmodes_cluster_counts.pdf"):
    """Visualizes the count of records in each KModes cluster."""
    cluster_counts = df['Cluster'].value_counts()
    # Create a bar plot
    plt.figure(figsize=(4, 3))
    ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, width=0.4, palette="Set2")
    
    # Add labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')
    
    # Add labels and title
    plt.xlabel("Cluster Label", fontsize=7)
    plt.ylabel("Number of Rows", fontsize=7)
    plt.title("Number of Rows in Each Cluster (KModes)", fontsize=8)

    ax.tick_params(axis='both', labelsize=6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6) 

    ax.grid(False)
    ax.set_axisbelow(True)
    sns.despine() 
    
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def visualize_kmodes_demographics(df, demographic_features, output_dir="demographic_plots_kmodes"):
    """Visualizes how each demographic category is distributed across KModes clusters and saves the plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="white")

    for feature in demographic_features:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(data=df, x=feature, hue='Cluster', palette="Set2")

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 2),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8, fontweight='bold')
                    
        y_max = max([bar.get_height() for container in ax.containers for bar in container] + [1])
        ax.set_ylim(top=y_max * 1.12)

    
        plt.title(f"Distribution of {feature} across Clusters Using KModes", fontsize=10)
        plt.xlabel(feature, fontsize=9)
        plt.ylabel("Count", fontsize=9)
        plt.legend(title="Cluster", fontsize=8, title_fontsize=9)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()

        ax.legend(
            title="Cluster", 
            fontsize=8, 
            title_fontsize=9, 
            loc='upper left', 
            bbox_to_anchor=(0, 1),
            borderaxespad=0.1,
            frameon=False)
        # Save as PDF
        plt.savefig(f"{output_dir}/{feature}_kmodes_distribution.pdf", format="pdf", bbox_inches="tight")
        plt.close()


def generate_wordclouds_per_cluster(df, entropy_df, text_type="context"):
    """
    pass the entropy loss function to the word cloud funcion) to control how big the words 
    """
    os.makedirs("wordclouds-KModes", exist_ok=True)

    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]

        if text_type == "context":
            text_data = cluster_df["context"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "response":
            text_data = cluster_df["response"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "both":
            context = cluster_df["context"].dropna().astype(str)
            response = cluster_df["response"].dropna().astype(str)
            text_data = context.str.cat(sep=" ") + " " + response.str.cat(sep=" ")
        else:
            raise ValueError("Invalid text_type. Choose from 'context', 'response', or 'both'.")

        # Tokenize and lowercase words
        words = text_data.lower().split()

        # Get entropy reduction mapping for this cluster
        cluster_entropy = entropy_df[entropy_df["Cluster"] == cluster].set_index("Item_ID")["Entropy_Reduction"].to_dict()

        word_freq = {}
        for word in words:
            if word in cluster_entropy:
                word_freq[word] = word_freq.get(word, 0) + cluster_entropy[word]
            else:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Generate word cloud from weighted frequencies
        wordcloud = WordCloud(width=800, height=400, background_color="white",
                              color_func=black_color_func, stopwords=custom_stopwords).generate_from_frequencies(word_freq)

        # Save as PDF
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Entropy Reduction Terms in Cluster {cluster} – {text_type.capitalize()}")

        base_filename = f"wordclouds-KModes/cluster_{cluster}_{text_type}"
        plt.savefig(f"{base_filename}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        print(f"WordCloud saved: {base_filename}.pdf")


def main():
    file_path = "Desktop/cs-rit/last_term/capstone/pythonProject/diverse_safety_adversarial_dialog_350.csv"
    demographic_fields = ['rater_gender', 'rater_race', 'rater_age', 'rater_education']
    
    # Load and reshape
    df = load_data(file_path)
    reshaped_df = reshape_responses(df)
    print("Reshaped:", reshaped_df.shape)

    # Keep numerical question-response matrix
    question_response_df = reshaped_df.copy()
    numerical_response_df = convert_responses_to_numerical(question_response_df)
    numerical_response_df = numerical_response_df.iloc[:, 1:]

    # Convert to string since K-Modes requires categorical (not numeric) input
    kmodes_input_df = numerical_response_df.astype(str)

    # K-Modes Clustering
    print("\nClustering with K-Modes...")
    km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)
    cluster_labels = km.fit_predict(kmodes_input_df)

    clustered_response = numerical_response_df.copy()
    clustered_response['Cluster'] = cluster_labels

    # Print cluster summary
    print("\nK-Modes Cluster Counts:")
    print(clustered_response['Cluster'].value_counts())

    # Merge back rater demographics
    clustered_response = merge_demographics(clustered_response, df, demographic_fields)

    # Visualize clusters and demographics
    visualize_kmodes_clusters(clustered_response)
    visualize_kmodes_demographics(clustered_response, demographic_fields)

    # Compute entropy on raw response matrix
    global_entropy_df = compute_global_entropy(question_response_df)
    within_cluster_disagreement = compute_within_cluster_disagreement(clustered_response, global_entropy_df)

    # Get top entropy-reducing items
    top_items_per_cluster = get_top_items_per_cluster(within_cluster_disagreement, top_n=10)
    print("\nTop 10 Most Disagreed Items Within Each K-Modes Cluster (Entropy Reduction):")
    print(top_items_per_cluster)

    # Prepare for word clouds
    df["item_id"] = df["item_id"].astype(str)
    top_items_per_cluster["Item_ID"] = top_items_per_cluster["Item_ID"].astype(str)
    filtered_df = df.merge(top_items_per_cluster[["Item_ID", "Cluster"]],
                           left_on="item_id", right_on="Item_ID", how="inner")

    print("\nTop Disagreed Items with Context and Responses:")
    print(filtered_df[["Cluster", "item_id", "context", "response"]].drop_duplicates())

    # Word Clouds
    print("\nGenerating word clouds for user inputs (context)...")
    generate_wordclouds_per_cluster(filtered_df, entropy_df=within_cluster_disagreement, text_type="context")

    print("\nGenerating word clouds for model responses (response)...")
    generate_wordclouds_per_cluster(filtered_df, entropy_df=within_cluster_disagreement, text_type="response")

    print("\nGenerating word clouds for both context and response...")
    generate_wordclouds_per_cluster(filtered_df, entropy_df=within_cluster_disagreement, text_type="both")

    # Cluster size distribution plot
    clustered_response['Cluster'].value_counts().plot(kind='bar', title='K-Modes Cluster Size Distribution')
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Raters")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


# Hierarchical Clustering (cluster raters based on raters' response)
custom_stopwords = STOPWORDS.union({"USER", "LAMDA"})
def visualize_hierarchical_clusters(df, output_path="hierarchical_cluster_counts.pdf"):
    """Visualizes the count of records in each hierarchical cluster."""
    cluster_counts = df['Cluster'].value_counts()
    # Create a bar plot
    plt.figure(figsize=(4, 3))
    ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, width=0.4, palette="Set2")
    
    # Add labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')
    
    # Add labels and title
    plt.xlabel("Cluster Label", fontsize=7)
    plt.ylabel("Number of Rows", fontsize=7)
    plt.title("Number of Rows in Each Cluster (hierarchical)", fontsize=8)

    ax.tick_params(axis='both', labelsize=6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6) 

    ax.grid(False)
    ax.set_axisbelow(True)
    sns.despine() 
    
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def visualize_hierarchical_demographics(df, demographic_features, output_dir="demographic_plots_hierarchical"):
    """Visualizes how each demographic category is distributed across hierarchical clusters and saves the plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="white")

    for feature in demographic_features:
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(data=df, x=feature, hue='Cluster', palette="Set2")

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 2),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8, fontweight='bold')
                    
        y_max = max([bar.get_height() for container in ax.containers for bar in container] + [1])
        ax.set_ylim(top=y_max * 1.12)

    
        plt.title(f"Distribution of {feature} across Clusters Using hierarchical", fontsize=10)
        plt.xlabel(feature, fontsize=9)
        plt.ylabel("Count", fontsize=9)
        plt.legend(title="Cluster", fontsize=8, title_fontsize=9)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()

        ax.legend(
            title="Cluster", 
            fontsize=8, 
            title_fontsize=9, 
            loc='upper left', 
            bbox_to_anchor=(0, 1),
            borderaxespad=0.1,
            frameon=False)
        # Save as PDF
        plt.savefig(f"{output_dir}/{feature}_hierarchical_distribution.pdf", format="pdf", bbox_inches="tight")
        plt.close()


def generate_wordclouds_per_cluster(df, entropy_df, text_type="context"):
    """
    pass the entropy loss function to the word cloud funcion) to control how big the words 
    """
    os.makedirs("wordclouds-hierarchical", exist_ok=True)

    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]

        if text_type == "context":
            text_data = cluster_df["context"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "response":
            text_data = cluster_df["response"].dropna().astype(str).str.cat(sep=" ")
        elif text_type == "both":
            context = cluster_df["context"].dropna().astype(str)
            response = cluster_df["response"].dropna().astype(str)
            text_data = context.str.cat(sep=" ") + " " + response.str.cat(sep=" ")
        else:
            raise ValueError("Invalid text_type. Choose from 'context', 'response', or 'both'.")

        # Tokenize and lowercase words
        words = text_data.lower().split()

        # Get entropy reduction mapping for this cluster
        cluster_entropy = entropy_df[entropy_df["Cluster"] == cluster].set_index("Item_ID")["Entropy_Reduction"].to_dict()

        word_freq = {}
        for word in words:
            if word in cluster_entropy:
                word_freq[word] = word_freq.get(word, 0) + cluster_entropy[word]
            else:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Generate word cloud from weighted frequencies
        wordcloud = WordCloud(width=800, height=400, background_color="white",
                              color_func=black_color_func, stopwords=custom_stopwords).generate_from_frequencies(word_freq)

        # Save as PDF
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Entropy Reduction Terms in Cluster {cluster} – {text_type.capitalize()}")

        base_filename = f"wordclouds-hierarchical/cluster_{cluster}_{text_type}"
        plt.savefig(f"{base_filename}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        print(f"WordCloud saved: {base_filename}.pdf")


def apply_hierarchical_clustering(df, num_clusters=3):
    """Applies hierarchical clustering using Hamming distance and average linkage."""
    
    distance_matrix = pairwise_distances(df, metric='cosine')
    condensed_distance_matrix = pdist(df, metric='hamming')
    linked = linkage(condensed_distance_matrix, method='average')
    
    # Assign cluster labels
    df_copy = df.copy()
    df_copy['Cluster'] = fcluster(linked, t=5, criterion='maxclust')  

    return df_copy, linked


def plot_dendrogram(linked, labels, output_path="dendrogram_hierarchical.pdf"):
    """Plots the hierarchical clustering dendrogram with correct label mapping."""
    plt.figure(figsize=(12, 6))


    leaf_order = sch.leaves_list(linked)
    ordered_labels = [labels[i] for i in leaf_order]

    # Plot dendrogram
    dendrogram(linked, labels=ordered_labels, leaf_rotation=90, leaf_font_size=8)
    plt.title("Hierarchical Clustering Dendrogram of Raters Based on Response to Questions")
    plt.xlabel("Rater ID")
    plt.ylabel("Hamming Distance")
    # Save as PDF
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()

def main():
    file_path = "Desktop/cs-rit/last_term/capstone/pythonProject/diverse_safety_adversarial_dialog_350.csv"
    demographic_fields = ['rater_gender', 'rater_race', 'rater_age', 'rater_education']
    
    df = pd.read_csv(file_path)  
    reshaped_df = reshape_responses(df) 
    print("Data reshaped. New shape:", reshaped_df.shape)

    numerical_response_df = convert_responses_to_numerical(reshaped_df)
    numerical_response_df = numerical_response_df.iloc[:, 1:]
    print("Numerical responses prepared. Shape:", numerical_response_df.shape)

    # Apply hierarchical clustering
    # Top-K Feature Selection
    top_k = 450
    variances = numerical_response_df.var()
    selected_cols = variances.sort_values(ascending=False).head(top_k).index
    reduced_df = numerical_response_df[selected_cols]
    clustered_response, linked = apply_hierarchical_clustering(reduced_df, num_clusters=3)
    print("Hierarchical Clustering applied.")


    # Plot Dendrogram
    plot_dendrogram(linked, labels=reshaped_df["rater_id"].tolist())

    # Merge Demographics
    clustered_response = merge_demographics(clustered_response, df, demographic_fields)
    print("Demographics merged.")

    visualize_hierarchical_clusters(clustered_response, output_path="hierarchical_cluster_counts.pdf")
    # Visualize hierarchical demographics
    visualize_hierarchical_demographics(clustered_response, demographic_fields,output_dir="demographic_plots_hierarchical")


    # Compute Global Entropy
    global_entropy_df = compute_global_entropy(numerical_response_df)

    # Compute Within-Cluster Disagreement
    within_cluster_disagreement = compute_within_cluster_disagreement(clustered_response, global_entropy_df)

    # Rank Items within Each Cluster based on Entropy Reduction
    top_items_per_cluster = get_top_items_per_cluster(within_cluster_disagreement, top_n=10)

    print("\nTop 10 Most Disagreed Items Within Each Hierarchical Cluster (Entropy Reduction):")
    print(top_items_per_cluster)

    df["item_id"] = df["item_id"].astype(str)
    top_items_per_cluster["Item_ID"] = top_items_per_cluster["Item_ID"].astype(str)

    
    # Extract Conversations for Selected Items
    filtered_df = df.merge(top_items_per_cluster[["Item_ID", "Cluster"]], left_on="item_id", right_on="Item_ID", how="inner")
    
    # Print cluster counts
    print("\nCluster counts:")
    print(clustered_response['Cluster'].value_counts().sort_index())

    
    # Print Conversations
    print("\nTop Disagreed Items with Context and Responses:")
    print(filtered_df[["Cluster", "item_id", "context", "response"]].drop_duplicates())

    # Generate Word Clouds for context, response, and both combined
    print("\nGenerating word clouds for user inputs (context)...")
    generate_wordclouds_per_cluster(filtered_df, entropy_df=within_cluster_disagreement,text_type="context")
    
    print("\nGenerating word clouds for model responses (response)...")
    generate_wordclouds_per_cluster(filtered_df, entropy_df=within_cluster_disagreement,text_type="response")
    
    print("\nGenerating word clouds for both context and response...")
    generate_wordclouds_per_cluster(filtered_df, entropy_df=within_cluster_disagreement,text_type="both")

    # Apply hierarchical clustering
    clustered_response, linked = apply_hierarchical_clustering(numerical_response_df, num_clusters=3)
    print("Hierarchical Clustering applied.")
    
    

if __name__ == "__main__":
    main()





