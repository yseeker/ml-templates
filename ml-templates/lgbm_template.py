from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from dtreeviz.trees import *

def get_boston_dataset():
    boston = load_boston()
    X_array = boston.data
    y_array = boston.target
    df = pd.DataFrame(X_array, columns = boston.feature_names).assign(Price=np.array(y_array))
    return df

def preprocess():
    return None

def calculate_features():
    return None

def visualize_importance(models, features):
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importances_
        _df["column"] = features
        _df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, _df], axis=0, ignore_index=True
        )

    order = (
        feature_importance_df.groupby("column")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index[:50]
    )

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.4), 7))
    sns.boxplot(
        data=feature_importance_df,
        x="column",
        y="feature_importance",
        order=order,
        ax=ax,
        palette="viridis",
    )
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig("figures/features_importance.png")
    plt.close()
    return fig, ax

def split_data(df, target, test_size):
    train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 123)
    x_train = train_set.drop(target, axis = 1)
    y_train = train_set[target]
    x_valid = test_set.drop(target, axis = 1)
    y_valid = test_set[target]
    return x_train, y_train, x_valid, y_valid

def custom_RMSE(y_true, y_pred):
    residual = (y_pred - y_true).astype("float")
    grad = residual
    hess = np.ones(len(y_true))
    return grad, hess

def custom_RMSE2(y_true, y_pred):
    residual = (y_pred - y_true).astype("float")
    grad = np.where(y_pred-y_true > 0, residual*5, residual)
    hess = np.ones(len(y_true))
    return grad, hess

def custom_asymmetric_valid(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual**2)*10.0, residual**2) 
    return "custom_asymmetric_eval", np.mean(loss), False

def train_lgbm(x_train, y_train, x_valid, y_valid, features):
    params = {
        'objective': custom_RMSE2,
        'metric': ['rmse', 'mae'],
    }
    model = LGBMRegressor(**params)
    model.fit(
        x_train[features], y_train,
        sample_weight = x_train["B"],
        eval_set=[(x_valid, y_valid)],
        eval_metric=custom_asymmetric_valid,
        callbacks=[early_stopping(100), log_evaluation(0)]
    )
    y_pred[val_idx] = model.predict(x_valid[val_idx])
    return model

def train_lgbm_cv(X, y, split, feature_list):
    model_list = []
    for fold, (train_idx, val_idx) in enumerate(sk_fold.split(X, y)):
        x_train, y_train = X.iloc[train_idx,:], y[train_idx]
        x_valid, y_valid = X.iloc[val_idx,:], y[val_idx]
        model = train_lgbm()
        y_pred[val_idx] = model.predict(X[val_idx])
    visualize_importance(models, features)
    return model_list

def inference_lgbm(x_test, model_list):
    lgbm_preds = []
    for model in model_list:
        lgbm_preds.append(model.predict(x_test))
    preds_cv_avg = np.mean(lgbm_preds, axis=0)
    return preds_cv_avg

def pipeline():
    df = get_boston_dataset()
    x_train, y_train, x_valid, y_valid = split_data(df, target='Price', test_size=0.2)
    features = x_train.columns.tolist()
    train_lgbm(x_train, y_train, x_valid, y_valid, features)

pipeline()
    

