import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera, gaussian_kde

def metrics_plot_multi(vecs, bins=30):
    """
    محاسبه شاخص‌ها و رسم هیستوگرام + Bell Shape
    برای یک ستون یا چند ستون DataFrame/Series.
    """
    # اطمینان از numeric بودن
    if isinstance(vecs, pd.Series):
        vecs = pd.DataFrame(vecs)
    elif isinstance(vecs, np.ndarray):
        if vecs.ndim == 1:
            vecs = pd.DataFrame(vecs, columns=['Data'])
        else:
            vecs = pd.DataFrame(vecs, columns=[f'Col{i}' for i in range(vecs.shape[1])])
    elif not isinstance(vecs, pd.DataFrame):
        raise ValueError("Input must be a pandas Series, DataFrame, or numpy array")

    # تبدیل همه ستون‌ها به numeric و حذف غیر عددی
    vecs = vecs.apply(pd.to_numeric, errors='coerce')
    vecs = vecs.select_dtypes(include=[np.number])

    if vecs.shape[1] == 0:
        raise ValueError("No numeric columns found after coercion.")

    # ترتیب قطعی الفبایی برای ستون‌ها (case-insensitive)
    ordered_cols = sorted(list(vecs.columns), key=lambda c: str(c).lower())

    results = {}

    for col in ordered_cols:
        data = vecs[col].replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)

        if data.size == 0:
            print(f"\n--- {col} ---")
            print("Empty column after dropping NaNs/inf. Skipping.")
            continue

        # محاسبه شاخص‌ها
        sk_val = float(skew(data))
        ku_val = float(kurtosis(data, fisher=True))
        jb_stat, jb_p_val = jarque_bera(data)
        jb_p_val = float(jb_p_val)

        results[col] = (sk_val, ku_val, jb_p_val)

        # چاپ مقادیر
        print(f"\n--- {col} ---")
        print(f"Skewness: {sk_val:.4f}")
        print(f"Kurtosis (Fisher): {ku_val:.4f}")
        print(f"Jarque-Bera p-value: {jb_p_val:.4f}")
        print("Normality:", "Not rejected" if jb_p_val > 0.05 else "Rejected")

        # رسم هیستوگرام + KDE
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        sns.histplot(data, bins=bins, stat="count", color='skyblue', alpha=0.6, edgecolor='black')

        dmin, dmax = data.min(), data.max()
        if dmax > dmin and data.size >= 2:
            kde = gaussian_kde(data)
            x = np.linspace(dmin, dmax, 1000)
            kde_scaled = kde(x) * len(data) * (dmax - dmin) / bins
            plt.plot(x, kde_scaled, color='red', lw=2, label='KDE Curve')
            plt.fill_between(x, 0, kde_scaled, color='red', alpha=0.3)
        else:
            plt.plot([dmin, dmax], [0, 0], color='red', lw=2, label='KDE Curve (degenerate)')

        plt.title(f'Distribution of {col} (Count + Bell Shape)', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend()
        plt.show()

    return results


def metrics_plot(vecs, column_name=None, bins=30):
    """
    Wrapper امن برای Series یا DataFrame چندستونه
    """
    if isinstance(vecs, pd.Series) and column_name:
        vecs = vecs.rename(column_name)
    return metrics_plot_multi(vecs, bins=bins)