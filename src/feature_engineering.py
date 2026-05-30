#TODO: preprocesing data functions 
#TODO: wrapper functions for the feature engineering pipeline


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from nltk.sentiment import SentimentIntensityAnalyzer


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