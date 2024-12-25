import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_community_detection_results(csv_file):
    """
    Reads a CSV file with columns:
        elapsed_time, num_nodes, num_edges, modularity, nmi_louvain, nmi_leiden
    and produces a single figure with three subplots:

    1) A plot showing modularity vs NMI (Louvain and Leiden) with confidence intervals.
    2) A plot showing elapsed_time vs (num_nodes/num_edges) with confidence intervals.
    3) A plot showing num_nodes, num_edges, and modularity relationships.
    """

    # Read data from CSV
    df = pd.read_csv(csv_file)

    # ---------------------------------------------------------------------------
    # 1) Plot: Modularity vs NMI (Louvain / Leiden) with confidence intervals
    # ---------------------------------------------------------------------------
    # For confidence intervals, we'll assume multiple runs produce repeated values
    # for each unique modularity or grouping. We can group by 'modularity' if repeating.
    # If no repeated data for the same modularity values, confidence intervals will not
    # be meaningful and will appear as single points.
    # Here, we demonstrate an approach grouping by 'modularity' and calculating mean+std.
    grouped_mod = df.groupby("modularity", as_index=False).agg(
        {"nmi_louvain": ["mean", "std"], "nmi_leiden": ["mean", "std"]}
    )
    # Flatten columns
    grouped_mod.columns = [
        "modularity",
        "nmi_louvain_mean",
        "nmi_louvain_std",
        "nmi_leiden_mean",
        "nmi_leiden_std",
    ]

    # ---------------------------------------------------------------------------
    # 2) Plot: Elapsed time vs ratio (num_nodes / num_edges) with confidence intervals
    # ---------------------------------------------------------------------------
    df["nodes_edges_ratio"] = df["num_nodes"] / df["num_edges"]
    grouped_ratio = df.groupby("nodes_edges_ratio", as_index=False).agg(
        {"elapsed_time": ["mean", "std"]}
    )
    grouped_ratio.columns = [
        "nodes_edges_ratio",
        "elapsed_time_mean",
        "elapsed_time_std",
    ]

    # ---------------------------------------------------------------------------
    # 3) Plot: num_nodes, num_edges, and modularity
    # ---------------------------------------------------------------------------
    # We changed to a 2D scatter instead of a 3D plot, coloring points by modularity.
    # ---------------------------------------------------------------------------
    # Create the figure and subplots
    fig = plt.figure(figsize=(15, 5))

    # Subplot 1: Modularity vs NMI (Louvain / Leiden)
    ax1 = fig.add_subplot(1, 3, 1)
    sns.lineplot(
        data=grouped_mod,
        x="modularity",
        y="nmi_louvain_mean",
        label="NMI Louvain",
        ax=ax1,
    )
    ax1.fill_between(
        grouped_mod["modularity"],
        grouped_mod["nmi_louvain_mean"] - grouped_mod["nmi_louvain_std"],
        grouped_mod["nmi_louvain_mean"] + grouped_mod["nmi_louvain_std"],
        alpha=0.2,
    )

    sns.lineplot(
        data=grouped_mod,
        x="modularity",
        y="nmi_leiden_mean",
        label="NMI Leiden",
        ax=ax1,
    )
    ax1.fill_between(
        grouped_mod["modularity"],
        grouped_mod["nmi_leiden_mean"] - grouped_mod["nmi_leiden_std"],
        grouped_mod["nmi_leiden_mean"] + grouped_mod["nmi_leiden_std"],
        alpha=0.2,
    )
    ax1.set_xlabel("Modularity")
    ax1.set_ylabel("NMI")
    ax1.set_title("Modularity vs NMI (with CI)")
    ax1.legend()

    # Subplot 2: Elapsed time vs ratio (num_nodes/num_edges)
    ax2 = fig.add_subplot(1, 3, 2)
    sns.lineplot(
        data=grouped_ratio, x="nodes_edges_ratio", y="elapsed_time_mean", ax=ax2
    )
    ax2.fill_between(
        grouped_ratio["nodes_edges_ratio"],
        grouped_ratio["elapsed_time_mean"] - grouped_ratio["elapsed_time_std"],
        grouped_ratio["elapsed_time_mean"] + grouped_ratio["elapsed_time_std"],
        alpha=0.2,
    )
    ax2.set_xlabel("Average Degree")  # Changed label to 'Average Degree'
    ax2.set_ylabel("Elapsed Time (s)")
    ax2.set_title("Elapsed Time vs Graph Degree")  # Changed title

    # Subplot 3: Relationship among num_nodes, num_edges, and modularity
    ax3 = fig.add_subplot(1, 3, 3)
    scatter_plot = sns.scatterplot(
        data=df,
        x="num_nodes",
        y="num_edges",
        hue="modularity",
        palette="viridis",
        ax=ax3,
    )
    ax3.set_title("Modularity by Graph Density")
    ax3.set_xlabel("Nodes")
    ax3.set_ylabel("Edges")
    # Create a colorbar for modularity
    norm = plt.Normalize(df["modularity"].min(), df["modularity"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, pad=0.02)
    cbar.set_label("Modularity")

    # Adjust spacing and show
    plt.tight_layout()
    plt.savefig("400g_100_p.png", dpi=300)
    plt.show()


import seaborn as sns


def visualize_community_detection_with_mu(csv_file):
    """
    Reads a CSV file with columns:
        elapsed_time, num_nodes, num_edges, modularity, nmi_louvain, nmi_leiden

    Assumes:
        - For each mu value (starting at 0.1 up to 0.9, increment 0.1), there are 10 consecutive rows.
        - The first 10 lines correspond to mu=0.1, the next 10 lines to mu=0.2, and so on.

    Produces two subplots that separately compare Modularity vs NMI for:
        - Louvain across mu values (with confidence intervals).
        - Leiden across mu values (with confidence intervals).
    """

    # Read data from CSV
    df = pd.read_csv(csv_file)

    # Assign mu based on row index (10 rows per mu block)
    df["mu_block"] = df.index // 10
    df["mu"] = 0.1 + 0.1 * df["mu_block"]
    df["mu"] = df["mu"].round(1)  # Avoid floating-point precision issues

    # Group by (mu, modularity) to calculate means and std for both Louvain and Leiden
    grouped = df.groupby(["mu", "modularity"], as_index=False).agg(
        {"nmi_louvain": ["mean", "std"], "nmi_leiden": ["mean", "std"]}
    )
    grouped.columns = [
        "mu",
        "modularity",
        "nmi_louvain_mean",
        "nmi_louvain_std",
        "nmi_leiden_mean",
        "nmi_leiden_std",
    ]

    # Create a figure with two subplots side by side
    fig, (ax_louvain, ax_leiden) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    # -----------------------------------------------------------------------
    # Subplot 1: Modularity vs NMI (Louvain), hue by mu, with confidence intervals
    # -----------------------------------------------------------------------
    sns.lineplot(
        ax=ax_louvain,
        data=grouped,
        x="modularity",
        y="nmi_louvain_mean",
        hue="mu",
        marker="o",
        palette="viridis",
        legend="full",
    )

    # Fill confidence intervals for Louvain
    for mu_val in grouped["mu"].unique():
        subset = grouped[grouped["mu"] == mu_val]
        ax_louvain.fill_between(
            subset["modularity"],
            subset["nmi_louvain_mean"] - subset["nmi_louvain_std"],
            subset["nmi_louvain_mean"] + subset["nmi_louvain_std"],
            alpha=0.2,
        )

    ax_louvain.set_title("Modularity vs NMI (Louvain)", fontsize=12)
    ax_louvain.set_xlabel("Modularity", fontsize=10)
    ax_louvain.set_ylabel("NMI (Louvain)", fontsize=10)
    ax_louvain.legend(title="mu")

    # -----------------------------------------------------------------------
    # Subplot 2: Modularity vs NMI (Leiden), hue by mu, with confidence intervals
    # -----------------------------------------------------------------------
    sns.lineplot(
        ax=ax_leiden,
        data=grouped,
        x="modularity",
        y="nmi_leiden_mean",
        hue="mu",
        marker="s",
        palette="viridis",
        legend=None,  # Hide the extra legend since it's the same mu values
    )

    # Fill confidence intervals for Leiden
    for mu_val in grouped["mu"].unique():
        subset = grouped[grouped["mu"] == mu_val]
        ax_leiden.fill_between(
            subset["modularity"],
            subset["nmi_leiden_mean"] - subset["nmi_leiden_std"],
            subset["nmi_leiden_mean"] + subset["nmi_leiden_std"],
            alpha=0.2,
        )

    ax_leiden.set_title("Modularity vs NMI (Leiden)", fontsize=12)
    ax_leiden.set_xlabel("Modularity", fontsize=10)
    ax_leiden.set_ylabel("NMI (Leiden)", fontsize=10)

    plt.tight_layout()
    plt.savefig("modularity_vs_nmi_louvain_leiden_separate_ci.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # Replace 'results.csv' with the actual CSV file generated by your shell script.
    file = "res/mocd_output.csv"
    visualize_community_detection_results(file)
    visualize_community_detection_with_mu(file)
