import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_pipeline import prepare_train_test_features
from src.eval import *
from src.config import *
from src.model_selection import select_best_model

def main():
    # load data
    train_df = pd.read_csv(TRAIN_PATH)

    # split data into train and test
    X = train_df.drop(columns=["fraudulent"])
    y = train_df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # feature engineering pipeline 
    X_train_final, X_test_final = prepare_train_test_features(
        X_train,
        X_test,
        y_train
    )

    # model selection 
    best_model, best_model_name, model_comp_results = select_best_model(
        X_train_final,
        y_train
    )

    # save comparison results to reports and print best model
    save_model_comparison_results(
        model_comp_results=model_comp_results,
        output_path=REPORTS_DIR / "model_comparison_results.txt"
    )

    print(f"model comparison resultssaved to: {REPORTS_DIR / 'model_comparison_results.csv'}")
    print(f"Best model chosen: {best_model_name}, with config: \n{best_model}")

    # evaluate best model  
    evaluate_model(
        best_model,
        X_test_final,
        y_test
    )

    # save results 
    y_pred = best_model.predict(X_test_final)
    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=y_pred,
        output_path=REPORTS_DIR / "confusion_matrix.png"
    )

    save_roc_curve_plot(
        model=best_model,
        X_test=X_test_final,
        y_test=y_test,
        output_path=REPORTS_DIR / "roc_curve_best_model.png",
        model_name=best_model_name
    )



if __name__ == "__main__":
    main()