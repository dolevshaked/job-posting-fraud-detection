#TODO: preprocesing data functions 
#TODO: wrapper functions for the preprocessing pipeline


def preprocess_text(text):
    """
    Preprocesses text by removing non-ASCII characters, tokenizing, filtering stop words, 
    lemmatizing, and joining tokens back into a string.
    Args:
        text (str): The text to preprocess.
    Returns:
        str: The preprocessed text.
    
    """
    # if not str
    if type(text) != str:
        return ''
    # remove non-ASCII and lower text
    ascii_chars = set(string.printable)
    text = ''.join(filter(lambda x: x in ascii_chars, text))
    tokens = word_tokenize(text.lower())
    # remove stop words 
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


#  wrapper  
def preprocess_text_columns(df, columns):
    """
    Preprocesses text columns by applying the preprocess_text function to each column.
    Args:
        df (pd.DataFrame): The dataframe containing the text columns.
        columns (list): The list of text columns to preprocess.
    Returns:
        pd.DataFrame: The dataframe with the preprocessed text columns.
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].apply(preprocess_text)
    return df


# nulls 
# nulls
def get_missing_percentage(df):
    """
    Calculates the percentage of missing values in each column.

    Args:
        df (pd.DataFrame): The dataframe to calculate the missing percentage for.

    Returns:
        pd.Series: A Series where the index contains column names and the values contain
        the percentage of missing values in each column, sorted from highest to lowest.
    """
    df = df.copy()
    return (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)


# drop cols 
def drop_columns(df, columns):
    """
    Drop selected columns from the dataframe.
    Args:
        df (pd.DataFrame): The dataframe to drop columns from.
        columns (list): The list of columns to drop.
    Returns:
        pd.DataFrame: The dataframe with the columns dropped.
    """
    df = df.copy()
    return df.drop(columns=columns, errors="ignore")


# fill nulls 
def fill_null_with_category_distribution(df, column, random_state=42):
    """
    Fill missing values in one categorical column using the existing category distribution.
    Args:
        df (pd.DataFrame): The dataframe to fill the missing values in.
        column (str): The column to fill the missing values in.
    Returns:
        pd.DataFrame: The dataframe with the missing values filled.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    value_counts = df[column].value_counts(normalize=True)
    null_indices = df.index[df[column].isnull()]

    df.loc[null_indices, column] = rng.choice(
        value_counts.index,
        size=len(null_indices),
        p=value_counts.values
    )

    return df


def fill_categorical_nulls(df, columns, random_state=42):
    """
    Fill missing values in categorical columns using each column's category distribution.
    Args:
        df (pd.DataFrame): The dataframe to fill the missing values in.
        columns (list): The list of columns to fill the missing values in.
        random_state (int): The random state to use for the random number generator.
    Returns:
        pd.DataFrame: The dataframe with the missing values filled.
    """
    df = df.copy()

    for col in columns:
        df = fill_null_with_category_distribution(df, col, random_state=random_state)

    return df


def fill_text_nulls(df, columns, fill_value=" "):
    """
    Fill missing values in text columns.
    """
    df = df.copy()

    for col in columns:
        df[col] = df[col].fillna(fill_value)

    return df





