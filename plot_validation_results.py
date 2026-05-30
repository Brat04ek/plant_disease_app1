import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_PATH = "results"
OUTPUT_PATH = os.path.join(RESULTS_PATH, "validation_plots")


def ensure_output_dir():
    os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_required_csv(file_name: str):
    file_path = os.path.join(RESULTS_PATH, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def add_config_label(df: pd.DataFrame):
    df = df.copy()

    df["config_label"] = df.apply(
        lambda row: (
            f"{row['config_name']}\n"
            f"K={int(row['n_clusters'])}, "
            f"trees={int(row['n_estimators'])}, "
            f"C={row['svm_C']}"
        ),
        axis=1
    )

    return df


def plot_cv_macro_f1_summary(summary_df: pd.DataFrame):
    df = add_config_label(summary_df)
    df = df.sort_values("macro_f1_score_mean", ascending=False).reset_index(drop=True)

    x = range(len(df))

    plt.figure(figsize=(12, 7))

    bars = plt.bar(x, df["macro_f1_score_mean"])

    if "macro_f1_score_std" in df.columns:
        plt.errorbar(
            x,
            df["macro_f1_score_mean"],
            yerr=df["macro_f1_score_std"],
            fmt="none",
            capsize=5,
            linewidth=1.2
        )

    plt.title("Порівняння конфігурацій за середнім macro F1-score")
    plt.xlabel("Конфігурація моделі")
    plt.ylabel("Середнє значення macro F1-score")

    min_value = max(0.0, df["macro_f1_score_mean"].min() - 0.03)
    max_value = min(1.0, df["macro_f1_score_mean"].max() + 0.03)
    plt.ylim(min_value, max_value)

    plt.xticks(x, df["config_label"], rotation=20, ha="right")

    for bar, value in zip(bars, df["macro_f1_score_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "01_cv_macro_f1_summary.jpg"), dpi=300)
    plt.close()


def plot_cv_main_metrics_summary(summary_df: pd.DataFrame):
    df = add_config_label(summary_df)
    df = df.sort_values("macro_f1_score_mean", ascending=False).reset_index(drop=True)

    x = range(len(df))

    plt.figure(figsize=(13, 7))

    plt.bar([i - 0.30 for i in x], df["accuracy_mean"], width=0.20, label="Accuracy")
    plt.bar([i - 0.10 for i in x], df["macro_precision_mean"], width=0.20, label="Macro precision")
    plt.bar([i + 0.10 for i in x], df["macro_recall_mean"], width=0.20, label="Macro recall")
    plt.bar([i + 0.30 for i in x], df["macro_f1_score_mean"], width=0.20, label="Macro F1-score")

    plt.title("Порівняння конфігурацій за основними метриками крос-валідації")
    plt.xlabel("Конфігурація моделі")
    plt.ylabel("Середнє значення метрики")

    min_metric = min(
        df["accuracy_mean"].min(),
        df["macro_precision_mean"].min(),
        df["macro_recall_mean"].min(),
        df["macro_f1_score_mean"].min()
    )

    plt.ylim(max(0.0, min_metric - 0.03), 1.0)

    plt.xticks(x, df["config_label"], rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "02_cv_main_metrics_summary.jpg"), dpi=300)
    plt.close()


def plot_cv_fold_stability(cv_df: pd.DataFrame):
    df = cv_df.copy()

    config_order = (
        df.groupby("config_name")["macro_f1_score"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    plt.figure(figsize=(12, 7))

    positions = []
    labels = []
    data = []

    for idx, config_name in enumerate(config_order, start=1):
        subset = df[df["config_name"] == config_name]
        data.append(subset["macro_f1_score"].values)
        positions.append(idx)

        first_row = subset.iloc[0]
        labels.append(
            f"{config_name}\n"
            f"K={int(first_row['n_clusters'])}, "
            f"trees={int(first_row['n_estimators'])}, "
            f"C={first_row['svm_C']}"
        )

    plt.boxplot(data, positions=positions, showmeans=True)

    plt.title("Стабільність macro F1-score за fold крос-валідації")
    plt.xlabel("Конфігурація моделі")
    plt.ylabel("Macro F1-score")

    all_values = df["macro_f1_score"]
    plt.ylim(max(0.0, all_values.min() - 0.03), min(1.0, all_values.max() + 0.03))

    plt.xticks(positions, labels, rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "03_cv_macro_f1_fold_stability.jpg"), dpi=300)
    plt.close()


def plot_cv_fold_lines(cv_df: pd.DataFrame):
    df = cv_df.copy()

    plt.figure(figsize=(12, 7))

    for config_name, subset in df.groupby("config_name"):
        subset = subset.sort_values("fold")

        first_row = subset.iloc[0]
        label = (
            f"{config_name} "
            f"(K={int(first_row['n_clusters'])}, "
            f"trees={int(first_row['n_estimators'])}, "
            f"C={first_row['svm_C']})"
        )

        plt.plot(
            subset["fold"],
            subset["macro_f1_score"],
            marker="o",
            label=label
        )

    plt.title("Значення macro F1-score на окремих fold")
    plt.xlabel("Номер fold")
    plt.ylabel("Macro F1-score")

    min_value = max(0.0, df["macro_f1_score"].min() - 0.03)
    max_value = min(1.0, df["macro_f1_score"].max() + 0.03)
    plt.ylim(min_value, max_value)

    plt.xticks(sorted(df["fold"].unique()))
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "04_cv_macro_f1_by_fold.jpg"), dpi=300)
    plt.close()


def plot_cv_time_comparison(summary_df: pd.DataFrame):
    df = add_config_label(summary_df)
    df = df.sort_values("fold_time_seconds_sum", ascending=True).reset_index(drop=True)

    x = range(len(df))

    plt.figure(figsize=(12, 7))

    bars = plt.bar(x, df["fold_time_seconds_sum"] / 60)

    plt.title("Порівняння сумарного часу крос-валідації")
    plt.xlabel("Конфігурація моделі")
    plt.ylabel("Сумарний час, хв")

    plt.xticks(x, df["config_label"], rotation=20, ha="right")

    for bar, value in zip(bars, df["fold_time_seconds_sum"] / 60):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "05_cv_time_comparison.jpg"), dpi=300)
    plt.close()


def plot_final_metrics(final_metrics_df: pd.DataFrame):
    row = final_metrics_df.iloc[0]

    metrics = {
        "Accuracy": row["accuracy"],
        "Macro precision": row["macro_precision"],
        "Macro recall": row["macro_recall"],
        "Macro F1-score": row["macro_f1_score"],
        "Weighted F1-score": row["weighted_f1_score"]
    }

    labels = list(metrics.keys())
    values = list(metrics.values())

    x = range(len(labels))

    plt.figure(figsize=(10, 6))

    bars = plt.bar(x, values)

    plt.title("Фінальні метрики на відкладеній тестовій вибірці")
    plt.xlabel("Метрика")
    plt.ylabel("Значення")

    min_value = max(0.0, min(values) - 0.03)
    max_value = min(1.0, max(values) + 0.03)
    plt.ylim(min_value, max_value)

    plt.xticks(x, labels, rotation=15, ha="right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "06_final_test_metrics.jpg"), dpi=300)
    plt.close()


def plot_class_f1_scores(class_metrics_df: pd.DataFrame):
    df = class_metrics_df.copy()
    df = df.sort_values("f1_score", ascending=True).reset_index(drop=True)

    y = range(len(df))

    plt.figure(figsize=(11, 8))

    bars = plt.barh(y, df["f1_score"])

    plt.title("F1-score за класами на тестовій вибірці")
    plt.xlabel("F1-score")
    plt.ylabel("Клас")

    plt.xlim(max(0.0, df["f1_score"].min() - 0.05), 1.0)
    plt.yticks(y, df["class_name"], fontsize=8)

    for bar, value in zip(bars, df["f1_score"]):
        plt.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=8
        )

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "07_final_f1_score_by_class.jpg"), dpi=300)
    plt.close()


def plot_class_precision_recall_f1(class_metrics_df: pd.DataFrame):
    df = class_metrics_df.copy()
    df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)

    x = range(len(df))

    plt.figure(figsize=(15, 7))

    plt.bar([i - 0.25 for i in x], df["precision"], width=0.25, label="Precision")
    plt.bar(x, df["recall"], width=0.25, label="Recall")
    plt.bar([i + 0.25 for i in x], df["f1_score"], width=0.25, label="F1-score")

    plt.title("Precision, recall та F1-score за класами")
    plt.xlabel("Клас")
    plt.ylabel("Значення метрики")

    min_metric = min(
        df["precision"].min(),
        df["recall"].min(),
        df["f1_score"].min()
    )

    plt.ylim(max(0.0, min_metric - 0.05), 1.0)

    plt.xticks(x, df["class_name"], rotation=75, ha="right", fontsize=8)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "08_final_precision_recall_f1_by_class.jpg"), dpi=300)
    plt.close()


def plot_lowest_classes(class_metrics_df: pd.DataFrame, n: int = 5):
    df = class_metrics_df.copy()
    df = df.sort_values("f1_score", ascending=True).head(n).reset_index(drop=True)

    x = range(len(df))

    plt.figure(figsize=(10, 6))

    plt.bar([i - 0.25 for i in x], df["precision"], width=0.25, label="Precision")
    plt.bar(x, df["recall"], width=0.25, label="Recall")
    plt.bar([i + 0.25 for i in x], df["f1_score"], width=0.25, label="F1-score")

    plt.title(f"Класи з найнижчим значенням F1-score")
    plt.xlabel("Клас")
    plt.ylabel("Значення метрики")

    min_metric = min(
        df["precision"].min(),
        df["recall"].min(),
        df["f1_score"].min()
    )

    plt.ylim(max(0.0, min_metric - 0.05), 1.0)

    plt.xticks(x, df["class_name"], rotation=25, ha="right", fontsize=9)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "09_lowest_f1_classes.jpg"), dpi=300)
    plt.close()


def plot_confusion_matrix_from_csv():
    cm_path = os.path.join(RESULTS_PATH, "final_confusion_matrix.csv")

    if not os.path.exists(cm_path):
        print("final_confusion_matrix.csv not found, skipping confusion matrix heatmap.")
        return

    cm_df = pd.read_csv(cm_path, index_col=0)

    plt.figure(figsize=(12, 10))

    plt.imshow(cm_df.values, aspect="auto")
    plt.title("Матриця помилок на тестовій вибірці")
    plt.xlabel("Прогнозований клас")
    plt.ylabel("Справжній клас")

    plt.xticks(
        ticks=np.arange(len(cm_df.columns)),
        labels=cm_df.columns,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        ticks=np.arange(len(cm_df.index)),
        labels=cm_df.index,
        fontsize=7
    )

    plt.colorbar(label="Кількість зображень")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "10_final_confusion_matrix_heatmap.jpg"), dpi=300)
    plt.close()


def main():
    ensure_output_dir()

    print("=" * 80)
    print("Building validation plots from saved results")
    print("=" * 80)

    cv_df = load_required_csv("cv_results_all_folds.csv")
    summary_df = load_required_csv("cv_results_summary.csv")
    final_metrics_df = load_required_csv("final_test_metrics_summary.csv")
    class_metrics_df = load_required_csv("final_class_metrics_by_class.csv")

    plot_cv_macro_f1_summary(summary_df)
    plot_cv_main_metrics_summary(summary_df)
    plot_cv_fold_stability(cv_df)
    plot_cv_fold_lines(cv_df)
    plot_cv_time_comparison(summary_df)

    plot_final_metrics(final_metrics_df)
    plot_class_f1_scores(class_metrics_df)
    plot_class_precision_recall_f1(class_metrics_df)
    plot_lowest_classes(class_metrics_df, n=5)
    plot_confusion_matrix_from_csv()

    print("\nPlots saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()