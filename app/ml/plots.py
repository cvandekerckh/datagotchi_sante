import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from app.ml.constants import Constants as C
from app.ml.loaders import load_targets

# Coefficients from the final model (used in print_feature_weights and print_missing_data_rates)
FEATURE_COEFFICIENTS = {
    "sommeil_1": 19.480257779442653,
    "autogestion_9": 13.739258950508374,
    "act_friends": 10.573981915422074,
    "quartier_domicile_3": 10.37441561209553,
    "act_volunteer": 7.582039522720624,
    "act_nature_1": 6.712984457109272,
    "issue_ai_data_3_4.0": 6.211935973164852,
    "style_2.0": 5.800051764248763,
    "origines_ethniques_2.0": 5.546301967076551,
    "quartier_opportunite": 5.277927206481349,
    "style_8.0": 4.346996663137828,
    "maladies_15": 4.316886975680833,
    "chronotype_3.0": 3.114592617545137,
    "style_3.0": 2.621579066168746,
    "maladies_20": 2.2096988659620567,
    "travail_domaine_10": 1.9941260783687138,
    "travail_domaine_1": 1.611802693180884,
    "travail_domaine_6": 0.7180376658381294,
    "car_model_4.0": -0.09800847663296446,
    "nb_friends_dispo": -0.9103018830122579,
    "chronotype_2.0": -1.0195887582044736,
    "maladies_16": -1.2263633227378077,
    "consult_who_3": -1.8672544591860067,
    "travail_domaine_2": -2.4228695857740035,
    "married_5.0": -2.8769237833374386,
    "LatDec_3": -3.2189642982888675,
    "smoking": -3.3220889376401725,
    "maladies_22": -3.4089094543669365,
    "consult_who_6": -3.692072046631762,
    "origines_ethniques_4.0": -4.535129453843979,
    "travail_domaine_8": -4.744858060178581,
    "maladies_21": -5.675348481322472,
    "consult_who_5": -5.783787669451122,
    "maladies_19": -7.299199694152446,
    "travail_domaine_3": -7.64902421282413,
    "SoutSup_6": -10.924356674397478,
}


def print_feature_contribution_table():
    # === Load data ===
    FEATURE_SELECTION_PATH = (
        C.ML_PATH / "real" / C.FEATURE_SELECTION_FOLDER_NAME / "feature_library_v10"
    )
    FEATURE_LIBARY_PATH = (
        C.ML_PATH / "real" / C.FEATURE_LIBRARIES_FOLDER_NAME / "feature_library_v10"
    )
    feature_selection_filename = C.FEATURE_SELECTION_FILENAME.format("xgboost_k_20")

    feature_scores = pd.read_csv(FEATURE_SELECTION_PATH / feature_selection_filename)
    feature_lookup = pd.read_csv(FEATURE_LIBARY_PATH / C.FEATURE_LOOKUP_FILENAME)

    # === Merge to associate each feature with its corresponding question ID ===
    df = feature_scores.merge(
        feature_lookup[["feature_names", "id"]], on="feature_names", how="left"
    )

    # === Sort by feature importance ===
    df = df.sort_values(by="feature_scores", ascending=False).reset_index(drop=True)

    # === Compute cumulative sum of feature scores ===
    df["cumulative_feature_score"] = df["feature_scores"].cumsum()

    # === Compute cumulative number of unique questions (based on ID) ===
    unique_questions = []
    seen = set()
    for qid in df["id"]:
        seen.add(qid)
        unique_questions.append(len(seen))
    df["num_questions"] = unique_questions

    # === Compute number of features so far ===
    df["num_features"] = df.index + 1

    # === Aggregate to get first occurrence for each question count ===
    agg = df.groupby("num_questions", as_index=False).agg(
        num_features=("num_features", "max"),
        cumulative_feature_score=("cumulative_feature_score", "max"),
    )

    # === Filter for the target question counts ===
    target_questions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 74]
    result = agg[agg["num_questions"].isin(target_questions)].copy()

    # === Compute delta (incremental gain in cumulative feature score) ===
    result["delta_feature_score"] = result["cumulative_feature_score"].diff().fillna(0)

    # === Display final table ===
    print(result)


def print_vif():
    """
    Compute and display Variance Inflation Factors (VIF) for selected features.

    - Reads 'feature_library.csv' (with columns: ResponseID, feature1, feature2, ...)
    - Reads 'feature_selection_xgboost_k_20.csv' (with columns: feature_names, feature_score, feature_selected)
    - Keeps only features with feature_selected == 1
    - Handles NaNs by imputing column means
    - Computes and prints the VIF for each selected feature
    - Plots the VIF distribution with fixed bin width 0.01
    """

    # --- Load the data ---
    FEATURE_SELECTION_PATH = (
        C.ML_PATH / "real" / C.FEATURE_SELECTION_FOLDER_NAME / "feature_library_v10"
    )
    FEATURE_LIBARY_PATH = (
        C.ML_PATH / "real" / C.FEATURE_LIBRARIES_FOLDER_NAME / "feature_library_v10"
    )
    feature_selection_filename = C.FEATURE_SELECTION_FILENAME.format("xgboost_k_20")

    features_df = pd.read_csv(FEATURE_LIBARY_PATH / C.FEATURE_LIBRARY_FILENAME)
    selection_df = pd.read_csv(FEATURE_SELECTION_PATH / feature_selection_filename)

    # --- Filter selected features ---
    selected_features = selection_df.loc[
        selection_df["feature_selected"] == 1, "feature_names"
    ].tolist()

    if not selected_features:
        print("No selected features found (feature_selected = 1).")
        return

    # --- Extract corresponding columns from feature_library ---
    X = features_df[selected_features].copy()

    # --- Handle NaNs (replace with column mean) ---
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)

    # --- Compute VIFs manually using scikit-learn ---
    vif_list = []
    for feature in X.columns:
        y = X[feature].values
        X_other = X.drop(columns=[feature]).values
        model = LinearRegression()
        model.fit(X_other, y)
        r2 = model.score(X_other, y)
        vif = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
        vif_list.append(vif)

    vif_data = pd.DataFrame({"feature": X.columns, "VIF": vif_list})
    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    # --- Print sorted table ---
    print("\n=== Variance Inflation Factor (VIF) ===")
    print(vif_data.to_string(index=False))

    # --- Plot distribution with fixed bin width 0.01 ---
    vifs = vif_data["VIF"].values
    bin_width = 0.01

    # Start at max(1.0, min observed VIF), since VIF is theoretically >= 1
    vmin = max(1.0, vifs.min())
    vmax = vifs.max()

    # If all VIFs are equal, force a small range so that the histogram still displays
    if vmax == vmin:
        vmax = vmin + bin_width

    bins = np.arange(vmin, vmax + bin_width, bin_width)

    plt.figure(figsize=(8, 5))
    plt.hist(vifs, bins=bins, edgecolor="black", alpha=0.7, color="#31b5b7")
    plt.title("Distribution of VIF Values")
    plt.xlabel("VIF")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_feature_weights():
    # Convert to DataFrame and sort
    df = pd.DataFrame(FEATURE_COEFFICIENTS.items(), columns=["Feature", "Coefficient"])
    df_sorted = df.sort_values(by="Coefficient", ascending=True)

    # Plot
    plt.figure(figsize=(10, 8))  # Reduced height from 12 to 8
    colors = df_sorted["Coefficient"].apply(lambda x: "green" if x > 0 else "red")
    plt.barh(df_sorted["Feature"], df_sorted["Coefficient"], color=colors)
    plt.xlabel("Coefficient Value")
    plt.title("Impact of Lifestyle Features on Well-being Score")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("coef_barplot.png", dpi=300)
    plt.show()


def compute_metrics(group):
    y_true = group["y_test"]
    y_pred = group["y_predict"]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return pd.Series(
        {
            "MAE": np.mean(np.abs(y_true - y_pred)),
            "MSE": np.mean((y_true - y_pred) ** 2),
            "R2": r2,
        }
    )


def _load_latest_predictions(experiment_name):
    artifacts_path = C.ML_PATH / "real" / "experiments" / experiment_name / "artifacts"
    artifact_dirs = sorted(artifacts_path.iterdir(), key=lambda p: p.stat().st_mtime)
    return pd.read_csv(artifact_dirs[-1] / "predictions.csv")


def print_boxplots():

    # Load Study 1 predictions (cross-validation) — base experiment
    CHOSEN_EXPERIMENT_FOLDER = "experiments/12_paper_review_round1"
    CHOSEN_ARTIFACT_FOLDER = "artifacts/HumorousShark5621"
    df_study1 = pd.read_csv(
        C.ML_PATH
        / "real"
        / CHOSEN_EXPERIMENT_FOLDER
        / CHOSEN_ARTIFACT_FOLDER
        / "predictions.csv"
    )
    df_study1["study"] = "Study 1"

    # Load additional models from 13_add_new_models experiment
    df_new_models = _load_latest_predictions("13_add_new_models")
    df_new_models["study"] = "Study 1 (new models)"

    # Load Study 2 predictions (out-of-sample)
    df_study2 = pd.read_csv(C.DATA_EXPLAIN_CLEAN_PATH / C.CLEAN_RESULTS_FILENAME)
    df_study2 = df_study2.rename(columns={
        "observed_wellbeing_score": "y_test",
        "predicted_wellbeing_score": "y_predict"
    })
    df_study2["model_name"] = "ridge_regressor"
    df_study2["study"] = "Study 2"

    # --- Compute metrics by study and model ---
    metrics_study1 = df_study1.groupby("model_name").apply(compute_metrics).reset_index()
    metrics_study1["study"] = "Study 1"
    metrics_study1["N"] = len(df_study1) // len(metrics_study1)  # approx N per model

    metrics_new_models = df_new_models.groupby("model_name").apply(compute_metrics).reset_index()
    metrics_new_models["study"] = "Study 1 (new models)"
    metrics_new_models["N"] = len(df_new_models) // len(metrics_new_models)

    metrics_study2 = df_study2.groupby("model_name").apply(compute_metrics).reset_index()
    metrics_study2["study"] = "Study 2"
    metrics_study2["N"] = len(df_study2)

    # Combine metrics tables
    metrics_combined = pd.concat([metrics_study1, metrics_new_models, metrics_study2], ignore_index=True)
    metrics_combined = metrics_combined[["study", "model_name", "N", "MAE", "MSE", "R2"]]

    # --- Print combined metrics table ---
    print("=== Model Performance: Study 1 (CV) vs Study 2 (Out-of-sample) ===\n")
    print(metrics_combined.sort_values(["study", "MAE"]).to_string(
        index=False, float_format="%.3f"
    ))

    # --- Ground truth distribution table ---
    # For Study 1, get unique y_test values (same across models)
    y_test_study1 = df_study1[df_study1["model_name"] == "ridge_regressor"]["y_test"]
    y_test_study2 = df_study2["y_test"]

    ground_truth_stats = pd.DataFrame({
        "study": ["Study 1", "Study 2"],
        "N": [len(y_test_study1), len(y_test_study2)],
        "Mean": [y_test_study1.mean(), y_test_study2.mean()],
        "SD": [y_test_study1.std(), y_test_study2.std()],
        "Min": [y_test_study1.min(), y_test_study2.min()],
        "Max": [y_test_study1.max(), y_test_study2.max()],
    })

    print("\n\n=== Ground Truth (Well-being Score) Distribution ===\n")
    print(ground_truth_stats.to_string(index=False, float_format="%.2f"))

    # --- Filtered analysis (excluding extreme values y_test = 0 and y_test = 100) ---
    df_study1_filtered = df_study1[(df_study1["y_test"] > 0) & (df_study1["y_test"] < 100)]
    df_new_models_filtered = df_new_models[(df_new_models["y_test"] > 0) & (df_new_models["y_test"] < 100)]
    df_study2_filtered = df_study2[(df_study2["y_test"] > 0) & (df_study2["y_test"] < 100)]

    y_test_study1_filt = df_study1_filtered[
        df_study1_filtered["model_name"] == "ridge_regressor"
    ]["y_test"]
    y_test_study2_filt = df_study2_filtered["y_test"]

    ground_truth_stats_filtered = pd.DataFrame({
        "study": ["Study 1", "Study 2"],
        "N": [len(y_test_study1_filt), len(y_test_study2_filt)],
        "Mean": [y_test_study1_filt.mean(), y_test_study2_filt.mean()],
        "SD": [y_test_study1_filt.std(), y_test_study2_filt.std()],
        "Min": [y_test_study1_filt.min(), y_test_study2_filt.min()],
        "Max": [y_test_study1_filt.max(), y_test_study2_filt.max()],
    })

    print("\n\n=== Ground Truth Distribution (excluding y_test = 0 and 100) ===\n")
    print(ground_truth_stats_filtered.to_string(index=False, float_format="%.2f"))

    # Compute metrics for filtered data
    metrics_study1_filt = df_study1_filtered.groupby("model_name").apply(
        compute_metrics
    ).reset_index()
    metrics_study1_filt["study"] = "Study 1"
    metrics_study1_filt["nMAE"] = metrics_study1_filt["MAE"] / y_test_study1_filt.std()

    y_test_new_models_filt = df_new_models_filtered[
        df_new_models_filtered["model_name"] == "ridge_regressor"
    ]["y_test"]
    metrics_new_models_filt = df_new_models_filtered.groupby("model_name").apply(
        compute_metrics
    ).reset_index()
    metrics_new_models_filt["study"] = "Study 1 (new models)"
    metrics_new_models_filt["nMAE"] = metrics_new_models_filt["MAE"] / y_test_new_models_filt.std()

    metrics_study2_filt = df_study2_filtered.groupby("model_name").apply(
        compute_metrics
    ).reset_index()
    metrics_study2_filt["study"] = "Study 2"
    metrics_study2_filt["nMAE"] = metrics_study2_filt["MAE"] / y_test_study2_filt.std()

    metrics_filtered = pd.concat([metrics_study1_filt, metrics_new_models_filt, metrics_study2_filt], ignore_index=True)
    metrics_filtered = metrics_filtered[["study", "model_name", "MAE", "nMAE", "MSE", "R2"]]

    print("\n\n=== Model Performance (excluding y_test = 0 and 100) ===\n")
    print(metrics_filtered.sort_values(["study", "MAE"]).to_string(
        index=False, float_format="%.3f"
    ))

    # --- Print prediction error distribution for Study 2 ---
    prediction_error = df_study2["y_predict"] - df_study2["y_test"]
    print("\n\nStudy 2 prediction error distribution (predicted - observed):")
    print(f"  Mean:   {prediction_error.mean():.2f}")
    print(f"  SD:     {prediction_error.std():.2f}")
    print(f"  Median: {prediction_error.median():.2f}")
    print(f"  Min:    {prediction_error.min():.2f}")
    print(f"  Max:    {prediction_error.max():.2f}")

    # --- Combine data for boxplot ---
    df_study1_ridge = df_study1[df_study1["model_name"] == "ridge_regressor"].copy()
    df_study2_copy = df_study2[["y_test", "y_predict", "model_name", "study"]].copy()
    df_combined = pd.concat([df_study1_ridge, df_study2_copy], ignore_index=True)
    df_combined["abs_error"] = np.abs(df_combined["y_test"] - df_combined["y_predict"])

    # --- Boxplot comparing Study 1 vs Study 2 (ridge) ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="study", y="abs_error", data=df_combined, hue="study", legend=False)
    plt.ylabel("Absolute Error")
    plt.xlabel("")
    plt.title("Prediction Error Distribution: Study 1 (CV) vs Study 2 (Out-of-sample)")
    plt.tight_layout()
    plt.show()

    # --- Side-by-side scatter plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Study 1: Ridge predictions
    sns.scatterplot(
        x="y_test", y="y_predict", data=df_study1_ridge, alpha=0.6, ax=axes[0]
    )
    axes[0].plot([0, 100], [0, 100], "--", color="gray")
    axes[0].set_xlabel("True Well-Being Score")
    axes[0].set_ylabel("Predicted Score")
    axes[0].set_title("Study 1: Ridge (Cross-Validation)")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)

    # Study 2: Ridge predictions (out-of-sample)
    sns.scatterplot(
        x="y_test", y="y_predict", data=df_study2, alpha=0.6, ax=axes[1]
    )
    axes[1].plot([0, 100], [0, 100], "--", color="gray")
    axes[1].set_xlabel("True Well-Being Score")
    axes[1].set_ylabel("Predicted Score")
    axes[1].set_title("Study 2: Ridge (Out-of-sample)")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.show()

def print_post_exclusion_sociodemos():
    # --- 1. Load data ---

    df = pd.read_csv(C.ML_PATH / "real" / C.ATTRIBUTES_FILENAME)
    fields_of_interest = ['genre', 'age', 'education', 'revenu', 'enfants']


    # --- 2. Mappings ---

    gender_map = {
        1: "Male (cisgender male)",
        2: "Female (cisgender female)",
        3: "Male (transgender male)",
        4: "Female (transgender woman)",
        5: "Non-binary",
        6: "Queer",
        7: "Agender",
        8: "Other"
    }

    income_map = {
        1: "No income",
        2: "$1 to $30,000",
        3: "$30,001 to $60,000",
        4: "$60,001 to $90,000",
        5: "$90,001 to $110,000",
        6: "$110,001 to $150,000",
        7: "$150,001 to $200,000",
        8: "More than $200,000"
    }

    education_map = {
        1: "No schooling",
        2: "Elementary school",
        3: "Highchool",
        4: "College / CEGEP / Classical College",
        5: "Bachelor's degree",
        6: "Master's degree",
        7: "PhD"
    }

    children_map = {
        1: "0",
        2: "1",
        3: "2",
        4: "3",
        5: "4",
        6: "5 or more"
    }

    # --- 3. Apply mapping---
    df["genre_cat"] = df["genre"].map(gender_map)
    df["education_cat"] = df["education"].map(education_map)
    df["revenu_cat"] = df["revenu"].map(income_map)
    df["enfants_cat"] = df["enfants"].map(children_map)

    # --- 4. Final number of participants ---
    final_n = len(df)

    # --- 5. Age stats ---
    age_mean = df["age"].mean()
    age_sd = df["age"].std()
    age_median = df["age"].median()
    age_iqr = df["age"].quantile(0.75) - df["age"].quantile(0.25)
    age_min = df["age"].min()
    age_max = df["age"].max()

    # --- 6. Distributions % ---
    gender_dist = df["genre_cat"].value_counts(normalize=True) * 100
    education_dist = df["education_cat"].value_counts(normalize=True) * 100
    income_dist = df["revenu_cat"].value_counts(normalize=True) * 100
    children_dist = df["enfants_cat"].value_counts(normalize=True) * 100

    # --- 7. API friendly ---
    print("\n--- STUDY 1: FINAL SAMPLE CHARACTERISTICS ---")
    print(f"Final N: {final_n}\n")

    print("Gender distribution (%):")
    print(gender_dist.round(1), "\n")

    print("Education distribution (%):")
    print(education_dist.round(1), "\n")

    print("Income distribution (%):")
    print(income_dist.round(1), "\n")

    print("Parental status (%):")
    print(children_dist.round(1), "\n")

    print("Age statistics:")
    print(f" Mean:   {age_mean:.2f}")
    print(f" SD:     {age_sd:.2f}")
    print(f" Median: {age_median:.2f}")
    print(f" IQR:    {age_iqr:.2f}")
    print(f" Range:  {age_min} – {age_max}")


def print_missing_data_rates():
    """
    Compute and display the percentage of missing data for each feature
    in feature_library_v10.
    """
    FEATURE_LIBRARY_PATH = (
        C.ML_PATH / "real" / C.FEATURE_LIBRARIES_FOLDER_NAME / "feature_library_v10"
    )
    features_df = pd.read_csv(FEATURE_LIBRARY_PATH / C.FEATURE_LIBRARY_FILENAME)

    # Exclude ResponseId column if present
    if "ResponseId" in features_df.columns:
        features_df = features_df.drop(columns=["ResponseId"])

    # Compute missing rate per feature
    missing_rates = features_df.isnull().mean() * 100
    missing_rates = missing_rates.sort_values(ascending=False)

    # Print results for all features
    print("\n=== Missing Data Rates per Feature (%) ===\n")
    print(f"Total features: {len(missing_rates)}")
    print(f"Total observations: {len(features_df)}\n")

    for feature, rate in missing_rates.items():
        print(f"{feature}: {rate:.2f}%")

    # Print results for selected features (from model coefficients)
    selected_features = list(FEATURE_COEFFICIENTS.keys())
    selected_missing = missing_rates[missing_rates.index.isin(selected_features)]
    selected_missing = selected_missing.sort_values(ascending=False)

    print("\n\n=== Missing Data Rates for Selected Features (%) ===\n")
    print(f"Selected features: {len(selected_missing)}")
    print(f"Total observations: {len(features_df)}\n")

    for feature, rate in selected_missing.items():
        coef = FEATURE_COEFFICIENTS[feature]
        print(f"{feature}: {rate:.2f}% (coef: {coef:.2f})")


def print_target_distribution():
    """
    Compute and display the distribution (mean, SD, range) of the rescaled
    well-being target (0-100) from feature_library_v10.
    """
    FEATURE_LIBRARY_PATH = (
        C.ML_PATH / "real" / C.FEATURE_LIBRARIES_FOLDER_NAME / "feature_library_v10"
    )
    df_targets = load_targets(FEATURE_LIBRARY_PATH, C.TARGETS_FILENAME)

    target_col = C.TARGET_SCORE_TOT
    raw_values = df_targets[target_col]
    n_total = len(raw_values)
    n_missing = raw_values.isna().sum()

    # Filter out edge cases (0 and 70)
    non_missing = raw_values.dropna()
    n_zero = (non_missing == 0).sum()
    n_seventy = (non_missing == 70).sum()
    filtered_values = non_missing[(non_missing > 0) & (non_missing < 70)]

    # Rescale from 0-70 to 0-100
    target_values = filtered_values * (100 / 70)

    # Compute statistics
    mean_val = target_values.mean()
    sd_val = target_values.std()
    median_val = target_values.median()
    min_val = target_values.min()
    max_val = target_values.max()
    iqr_val = target_values.quantile(0.75) - target_values.quantile(0.25)
    n_obs = len(target_values)

    # Print results
    print("\n=== Well-being Target Distribution (score_tot, 0-100 scale) ===\n")
    print(f"N total: {n_total}")
    print(f"N missing: {n_missing}")
    print(f"N excluded (score=0): {n_zero}")
    print(f"N excluded (score=70): {n_seventy}")
    print(f"N observations: {n_obs}\n")
    print(f"Mean:   {mean_val:.2f}")
    print(f"SD:     {sd_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"IQR:    {iqr_val:.2f}")
    print(f"Range:  {min_val:.2f} - {max_val:.2f}")

