import inspect

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def serialize_obj(obj):
    if hasattr(obj, "__dict__"):
        return {"class": obj.__class__.__name__, "params": obj.__dict__}
    return str(obj)


class CrossvalConfig:

    # 0. Versioning
    RUN_TYPE = "REAL_FOLDER_NAME"  # SANDBOX_FOLDER_NAME or REAL_FOLDER_NAME
    FEATURE_LIBRARY_VERSION = "feature_library_v10"
    EXPERIMENT_NAME = "13_add_new_models"

    # 1. Modeling
    # 1.1. Feature Selection
    FEATURE_SELECTION_METHOD = ("xgboost", {"k": 20})
    NESTED_FEATURE_SELECTION = True  # If True, feature selection done inside each fold

    # 1.2 Evaluation
    # a) Metrics
    METRIC_LIST = [
        "mse",
        "mae",
    ]

    # b) Splitting
    KFOLD = 5
    MIN_TRAIN_SIZE = 30
    MIN_TEST_SIZE = 30
    RANDOM_STATE_SPLITTING = 42

    # 1.3 Hyperparameter optimization
    KFOLD_HP = 3

    # 1.2. Modelling
    MODEL_LIST = [
        {
            "model_name": "mean_regressor",
            "param_grid": {},
        },
        {
            "model_name": "random_regressor",
            "param_grid": {},
        },
        {
            "model_name": "xgboost_regressor",
            "param_grid": {
                "imputer": [SimpleImputer(strategy="mean"), KNNImputer()],
                "scaler": [StandardScaler(), MinMaxScaler()],
                "regressor__max_depth": [3, 6, 9],
                "regressor__subsample": [0.5, 0.8],
            },
        },
        {
            "model_name": "ridge_regressor",
            "param_grid": {
                "imputer": [SimpleImputer(strategy="mean"), KNNImputer()],
                "scaler": [StandardScaler(), MinMaxScaler()],
                "regressor__alpha": [1.0, 2.0, 3.0, 4.0],
            },
        },
        {
            "model_name": "random_forest_regressor",
            "param_grid": {
                "imputer": [SimpleImputer(strategy="mean"), KNNImputer()],
                "scaler": [StandardScaler()],
                "regressor__n_estimators": [100, 300],
                "regressor__max_depth": [3, 6, None],
                "regressor__max_features": ["sqrt", 0.5],
            },
        },
        {
            "model_name": "linear_regressor",
            "param_grid": {
                "imputer": [SimpleImputer(strategy="mean"), KNNImputer()],
                "scaler": [StandardScaler(), MinMaxScaler()],
            },
        },
        {
            "model_name": "lasso_regressor",
            "param_grid": {
                "imputer": [SimpleImputer(strategy="mean"), KNNImputer()],
                "scaler": [StandardScaler(), MinMaxScaler()],
                "regressor__alpha": [0.1, 0.5, 1.0, 2.0],
            },
        },
        {
            "model_name": "elasticnet_regressor",
            "param_grid": {
                "imputer": [SimpleImputer(strategy="mean"), KNNImputer()],
                "scaler": [StandardScaler(), MinMaxScaler()],
                "regressor__alpha": [0.1, 0.5, 1.0],
                "regressor__l1_ratio": [0.2, 0.5, 0.8],
            },
        },
    ]
    TARGET_NAME = "C.TARGET_SCORE_TOT"  # TARGET_SCORE_TOT, TARGET_POSITIVE_FUNCTIONING, TARGET_EMOTIONAL_HEALTH
    TARGET_BORNE_SUP = (
        70  # 70 for score_tot, 55 for positive_functioning, 15 for emotional_health
    )

    # Display config as a dictionary
    @classmethod
    def to_dict(cls):
        return {
            name: serialize_obj(attr)
            for name, attr in cls.__dict__.items()
            if not inspect.isroutine(attr) and not name.startswith("__")
        }


class CreateSandboxConfig:
    SANDBOX_N_ATTRIBUTES = 20
    SANDBOX_N_SAMPLE = 200
    SANDBOX_RANDOM_STATE = 42
