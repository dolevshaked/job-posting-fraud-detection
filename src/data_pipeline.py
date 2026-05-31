from src.preprocessing import *
from src.feature_engineering import *
from src.config import *
import pandas as pd

def prepare_train_test_features(X_train, X_test, y_train):
    """
    Apply preprocessing and feature engineering.
    on both X_train and X_test.
    """

    # preprocessing
    X_train_processed = preprocess_text_columns(X_train, TEXT_PREPROCESS_COLUMNS)
    X_test_processed = preprocess_text_columns(X_test, TEXT_PREPROCESS_COLUMNS)

    X_train_processed = drop_columns(X_train_processed, DROP_COLUMNS)
    X_test_processed = drop_columns(X_test_processed, DROP_COLUMNS)

    X_train_processed = fill_categorical_nulls(
        X_train_processed,
        CATEGORICAL_NULL_COLUMNS,
        random_state=RANDOM_STATE
    )

    X_test_processed = fill_categorical_nulls(
        X_test_processed,
        CATEGORICAL_NULL_COLUMNS,
        random_state=RANDOM_STATE
    )

    X_train_processed = fill_text_nulls(X_train_processed, TEXT_NULL_COLUMNS)
    X_test_processed = fill_text_nulls(X_test_processed, TEXT_NULL_COLUMNS)

    # reduce high-dimensionality categorical cols
    clusterers = fit_category_clusters(
        X_train_processed,
        CLUSTER_COLUMNS,
        n_clusters=10,
        random_state=RANDOM_STATE
    )

    X_train_processed = transform_category_clusters(X_train_processed, clusterers)
    X_test_processed = transform_category_clusters(X_test_processed, clusterers)


    ordinal_encoders = fit_ordinal_encoders(
        X_train_processed,
        ORDINAL_CATEGORY_ORDERS
    )

    X_train_processed = transform_ordinal_columns(X_train_processed, ordinal_encoders)
    X_test_processed = transform_ordinal_columns(X_test_processed, ordinal_encoders)


    X_train_processed = map_column_values(
        X_train_processed,
        "required_education",
        EDUCATION_MAP
    )

    X_test_processed = map_column_values(
        X_test_processed,
        "required_education",
        EDUCATION_MAP
    )

    # add sentiment analysis features
    X_train_processed = add_sentiment_features(X_train_processed, TEXT_COLUMNS)
    X_test_processed = add_sentiment_features(X_test_processed, TEXT_COLUMNS)

    # add word count features
    X_train_processed = add_word_count_features(X_train_processed, TEXT_COLUMNS)
    X_test_processed = add_word_count_features(X_test_processed, TEXT_COLUMNS)

    # drop original text columns
    X_train_processed = drop_columns(X_train_processed, TEXT_COLUMNS)
    X_test_processed = drop_columns(X_test_processed, TEXT_COLUMNS)

    # normalize features
    scaler = fit_scaler(X_train_processed)
    X_train_scaled = transform_with_scaler(X_train_processed, scaler)
    X_test_scaled = transform_with_scaler(X_test_processed, scaler)

    # select features
    selector = forward_feature_selection(
        X_train_scaled,
        y_train,
        random_state=RANDOM_STATE
    )

    X_train_selected = select_features(X_train_scaled, selector)
    X_test_selected = select_features(X_test_scaled, selector)


    pca = fit_pca(
        X_train_selected,
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE
    )

    X_train_pca = transform_with_pca(X_train_selected, pca)
    X_test_pca = transform_with_pca(X_test_selected, pca)

    # save PCA plots
    save_pca_plot_explained_variance(pca)

    save_pca_plot_loading_scores(
        pca=pca,
        feature_names=X_train_selected.columns,
        top_n=5,
        output_dir=REPORTS_DIR
    )

    return X_train_pca, X_test_pca