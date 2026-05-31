from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from src.config import *

# cluster high-dimensionality categorical cols
def fit_category_clusters(df, columns, n_clusters=10, random_state=42):
    """
    Fit CountVectorizer and KMeans clusterer for each categorical text column.

    Args:
        df (pd.DataFrame): Training dataframe.
        columns (list): Columns to cluster.
        n_clusters (int): Number of clusters for KMeans.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary containing fitted vectorizer and KMeans model for each column.
    """
    clusters = {}

    for col in columns:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df[col].astype(str))

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(X)

        clusters[col] = {
            "vectorizer": vectorizer,
            "kmeans": kmeans
        }

    return clusters


def transform_category_clusters(df, clusters):
    """
    Transform categorical text columns into cluster labels using fitted clusters.

    Args:
        df (pd.DataFrame): Dataframe to transform.
        clusters (dict): Fitted vectorizer and KMeans objects for each column.

    Returns:
        pd.DataFrame: Dataframe with clustered columns.
    """
    df = df.copy()

    for col, objects in clusters.items():
        X = objects["vectorizer"].transform(df[col].astype(str))
        df[col] = objects["kmeans"].predict(X)

    return df


# ordinal encoding 
def fit_ordinal_encoders(df, category_orders):
    """
    Fit OrdinalEncoder for each categorical column.
    Args:
        df (pd.DataFrame): Dataframe to fit encoders.
        category_orders (dict): Dictionary where keys are column names and values are ordered categories.
    Returns:
        dict: Dictionary containing fitted OrdinalEncoder for each column.
    """
    encoders = {}

    for col, categories in category_orders.items():
        encoder = OrdinalEncoder(categories=[categories])
        encoder.fit(df[[col]])
        encoders[col] = encoder

    return encoders


def transform_ordinal_columns(df, encoders):
    """
    Transform categorical columns into ordinal values using fitted encoders.
    Args:
        df (pd.DataFrame): Dataframe to transform.
        encoders (dict): Dictionary containing fitted OrdinalEncoder for each column.
    Returns:
        pd.DataFrame: Dataframe with transformed ordinal columns.
    """
    df = df.copy()

    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[[col]]).astype(int)

    return df


def map_column_values(df, column, mapping):
    """
    Map values in one column according to a predefined dictionary.

    Args:
        df (pd.DataFrame): Dataframe to transform.
        column (str): Column to map.
        mapping (dict): Mapping from original values to numeric values.

    Returns:
        pd.DataFrame: Dataframe with mapped column.
    """
    df = df.copy()
    df[column] = df[column].map(mapping)
    return df


# sentiment analysis feature 
def get_sentiment(text, analyzer):
    """
    Convert text sentiment into numeric label:
    negative = 0, neutral = 1, positive = 2.
    """
    scores = analyzer.polarity_scores(str(text))

    if scores["pos"] > scores["neg"]:
        return 2
    elif scores["neg"] > scores["pos"]:
        return 0
    else:
        return 1


def add_sentiment_features(df, columns):
    """
    Add sentiment features for selected text columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): Text columns to analyze.

    Returns:
        pd.DataFrame: Dataframe with new sentiment columns.
    """
    df = df.copy()
    analyzer = SentimentIntensityAnalyzer()

    for col in columns:
        df[f"{col}_sentiment"] = df[col].apply(lambda text: get_sentiment(text, analyzer))

    return df


# word count features
def count_words(text):
    """
    Count words in a text value. Returns 0 for non-string values.
    """
    if isinstance(text, str):
        return len(text.split())
    return 0


def add_word_count_features(df, columns):
    """
    Add word-count features for selected text columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): Text columns to count words for.

    Returns:
        pd.DataFrame: Dataframe with new word-count columns.
    """
    df = df.copy()

    for col in columns:
        df[f"{col}_word_count"] = df[col].apply(count_words)

    return df


# scaling features
def fit_scaler(X_train):
    """
    fit MinMaxScaler on training features.

    Args:
        X_train (pd.DataFrame): training feature dataframe.

    Returns:
        MinMaxScaler: fitted scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(X, scaler):
    """
    scale features using a fitted scaler.

    Args:
        X (pd.DataFrame): feature dataframe to scale.
        scaler (MinMaxScaler): fitted scaler.

    Returns:
        pd.DataFrame: scaled feature dataframe with original columns.
    """
    X_scaled = scaler.transform(X)

    return pd.DataFrame(X_scaled,columns=X.columns,index=X.index)


# feature selection 
def forward_feature_selection(X_train, y_train, random_state=42, cv_splits=10,scoring="roc_auc",
n_features_to_select="auto"):
    """
    perform forward feature selection with logistic regression, AUC-ROC as the evaluation metric.
    uses 10-fold cross-validation. 
    """
    model = LogisticRegression(random_state=random_state, max_iter=1000)

    cv = StratifiedKFold(n_splits=cv_splits,shuffle=True,random_state=random_state)

    selector = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features_to_select,
        direction="forward",
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    selector.fit(X_train, y_train)

    return selector


def select_features(X, selector):
    """
    keep only the features selected by the fitted selector.
    Args:
        X (pd.DataFrame): feature dataframe to select features from.
        selector (SequentialFeatureSelector): fitted selector.
    Returns:
        pd.DataFrame: feature dataframe with only the selected features.
    """
    selected_mask = selector.get_support()
    return X.loc[:, selected_mask]



# dimensionality reduction 
def fit_pca(X_train, n_components=2, random_state=42):
    """
    Fit PCA on training features.

    Args:
        X_train (pd.DataFrame): Training feature dataframe.
        n_components (int): Number of principal components.
        random_state (int): Random seed.

    Returns:
        PCA: Fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train)
    return pca


def transform_with_pca(X, pca):
    """
    Transform features using a fitted PCA object.

    Args:
        X (pd.DataFrame): Feature dataframe.
        pca (PCA): Fitted PCA object.

    Returns:
        pd.DataFrame: Dataframe with principal components.
    """
    X_pca = pca.transform(X)

    columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]

    return pd.DataFrame(
        X_pca,
        columns=columns,
        index=X.index
    )


def get_pca_explained_variance(pca):
    """
    Return explained variance percentage for each principal component.
    """
    return pca.explained_variance_ratio_ * 100


def get_top_pca_loadings(pca, feature_names, component_index=0, top_n=5):
    """
    Return the top contributing original features for a selected principal component.
    """
    loading_scores = pd.Series(
        pca.components_[component_index],
        index=feature_names
    )

    top_features = loading_scores.abs().sort_values(ascending=False).head(top_n).index

    return loading_scores[top_features]


def save_pca_plot_explained_variance(pca):
    """
    save a bar plot of explained variance for each principal component.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    explained_variance = np.round(get_pca_explained_variance(pca), 2)
    labels = [f"PC{i+1}" for i in range(len(explained_variance))]

    plt.figure()
    plt.bar(labels, explained_variance)
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title("PCA Plot of Explained Variance")
    plt.savefig(REPORTS_DIR / "pca_plot_explained_variance.png", bbox_inches="tight")
    plt.close()
    print("PCA plot of explained variance saved to reports directoru in path:", 
            REPORTS_DIR / "pca_plot_explained_variance.png")


def save_single_pca_loading_scores(pca, feature_names, component_index=0, top_n=5, output_dir=REPORTS_DIR):
    """
    Calculate, plot, and save top loading scores for one principal component.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    top_loadings = get_top_pca_loadings(
        pca=pca,
        feature_names=feature_names,
        component_index=component_index,
        top_n=top_n
    )

    pc_number = component_index + 1

    print(f"Top {top_n} loading scores for PC{pc_number}:")
    print(top_loadings)

    plt.figure()
    sns.barplot(x=top_loadings.values, y=top_loadings.index)
    plt.xlabel("Loading Scores")
    plt.ylabel("Features")
    plt.title(f"Top Components - PC{pc_number}")

    output_path = output_dir / f"pca_top_components_pc{pc_number}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"PCA loading plot for PC{pc_number} saved to:", output_path)


def save_pca_plot_loading_scores(pca, feature_names, n_components=None, top_n=5, output_dir=REPORTS_DIR):
    """
    Save PCA loading plots for multiple principal components.
    Args:
        pca (PCA): fitted PCA object.
        feature_names (list): list of feature names.
        n_components (int): number of principal components.
        top_n (int): number of top loading scores to plot.
        output_dir (Path): output directory.
    """
    if n_components is None:
        n_components = pca.n_components_

    for component_index in range(n_components):
        save_single_pca_loading_scores(
            pca=pca,
            feature_names=feature_names,
            component_index=component_index,
            top_n=top_n,
            output_dir=output_dir
        )