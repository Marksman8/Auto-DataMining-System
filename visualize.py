import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_visuals(df):
    os.makedirs('static/plots', exist_ok=True)

    image_paths = {}

    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    heatmap_path = "static/plots/heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    image_paths["heatmap"] = heatmap_path

    # Scatter Plot (first 2 numeric columns)
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) >= 2:
        plt.figure()
        sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1])
        scatter_path = "static/plots/scatter.png"
        plt.savefig(scatter_path)
        plt.close()
        image_paths["scatter"] = scatter_path

    # Box Plot
    if len(num_cols) >= 1:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[num_cols])
        box_path = "static/plots/boxplot.png"
        plt.savefig(box_path)
        plt.close()
        image_paths["boxplot"] = box_path

    # Line Plot (if index is numeric or range)
    plt.figure()
    df[num_cols].plot(kind='line', figsize=(10, 5))
    plt.title("Line Plot")
    line_path = "static/plots/lineplot.png"
    plt.savefig(line_path)
    plt.close()
    image_paths["lineplot"] = line_path

    # Pie Chart (Top 1 categorical column with few unique values)
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].nunique() <= 6:
            plt.figure()
            df[col].value_counts().plot.pie(autopct='%1.1f%%')
            plt.title(f"Pie Chart - {col}")
            pie_path = f"static/plots/pie_{col}.png"
            plt.savefig(pie_path)
            plt.close()
            image_paths["piechart"] = pie_path
            break

    return image_paths
