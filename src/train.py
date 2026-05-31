from src.preprocessing import *
from src.feature_engineering import *
from src.config import *
import pandas as pd

# load data 
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
test_labels = pd.read_csv(TEST_LABELS_PATH)

# features, target split 
y_train = train_df["fraudulent"]
X_train = train_df.drop(columns=["fraudulent"])

X_test = test_df.copy()
y_test = test_labels["fraudulent"]

# preprocess text 
df = preprocess_text_columns(df, TEXT_PREPROCESS_COLUMNS)


# drop cols with above 60% missing 
df = drop_columns(df, DROP_COLUMNS)

# fill nulls 
df = fill_categorical_nulls(df, CATEGORICAL_NULL_COLUMNS, random_state=RANDOM_STATE)
df = fill_text_nulls(df, TEXT_NULL_COLUMNS, fill_value=" ")


# cluster high-dimensionality categorical cols
clusterers = fit_category_clusters(train_df,CLUSTER_COLUMNS,n_clusters=10,random_state=RANDOM_STATE)
train_df = transform_category_clusters(train_df, clusterers)
test_df = transform_category_clusters(test_df, clusterers)


# ordinal encoding 
ordinal_encoders = fit_ordinal_encoders(train_df, ORDINAL_CATEGORY_ORDERS)
train_df = transform_ordinal_columns(train_df, ordinal_encoders)
test_df = transform_ordinal_columns(test_df, ordinal_encoders)

# map education col ordinally
train_df = map_column_values(train_df, "required_education", EDUCATION_MAP)
test_df = map_column_values(test_df, "required_education", EDUCATION_MAP)

# sentiment analysis features
train_df = add_sentiment_features(train_df, TEXT_COLUMNS)
test_df = add_sentiment_features(test_df, TEXT_COLUMNS)

# word count features
train_df = add_word_count_features(train_df, TEXT_COLUMNS)
test_df = add_word_count_features(test_df, TEXT_COLUMNS)

# drop text cols
train_df = drop_columns(train_df, TEXT_COLUMNS)
test_df = drop_columns(test_df, TEXT_COLUMNS)

# scale features
scaler = fit_scaler(X_train)
X_train_scaled = transform_with_scaler(X_train, scaler)
X_test_scaled = transform_with_scaler(X_test, scaler)

# feature selection
selector = forward_feature_selection(X_train_scaled, y_train, random_state=RANDOM_STATE)

X_train_selected = select_features(X_train_scaled, selector)
X_test_selected = select_features(X_test_scaled, selector)

# dimensionality reduction 
pca = fit_pca(X_train=X_train_selected, n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca = transform_with_pca(X_train_selected, pca)
X_test_pca = transform_with_pca(X_test_selected, pca)
# save PCA plots
save_pca_plot_explained_variance(pca)
save_pca_plot_loading_scores(pca=pca, feature_names=X_train_selected.columns, top_n=5, output_dir=REPORTS_DIR)