import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

import matplotlib.lines as mlines
import os

sns.set_theme(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

# Group by run_id to apply rescaling per cross-validation run
def rescale_group(group, rescale=True):

    group = group.copy()

    allowed_reductions = np.linspace(0, 100, num=25)

    # Get the baseline value for this run_id (alpha == -1)
    value_baseline = group.loc[group['alpha'] == -1, 'H(S,X)'].values[0]
    
    # Drop the row where alpha == value_baseline (originally incorrect condition)
    group = group[group.alpha != value_baseline]
    
    # Remap alpha values based on the baseline
    group.loc[group['alpha'] == -1, 'alpha'] = value_baseline
    group.loc[:, 'reduction'] = 100 * (1 - (group['alpha'] / value_baseline))
    
    # Map to closest allowed reduction
    group.loc[:, 'alpha'] = group['reduction'].apply(
        lambda x: allowed_reductions[np.argmin(np.abs(allowed_reductions - x))]
    )
    
    if rescale:
        max_harm = group['H(S,X)'].max()
        min_harm = group['H(S,X)'].min()
        group.loc[group['alpha'] == 0, 'H(S,X)'] = max_harm
        group['H(S,X)'] = ((group['H(S,X)']-min_harm)/(max_harm-min_harm))*value_baseline
    
    # Compute empirical harmfulness
    group.loc[:, 'empr_harmfulness'] = 100 * (1 - (group['H(S,X)'] / value_baseline))

    return group

def load_csv_files_to_dataframe(directory, rescale=True):
    # List to store dataframes
    dataframes = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            # Extract model name
            model_name = os.path.basename(filepath).split("_")[3]
            users_targeted = os.path.basename(filepath).split("_")[-1].replace(".csv", "")
            print(f"Loading file: {filepath}")
            # Read the CSV file and append to the list
            tmp = pd.read_csv(filepath, float_precision='high')
            tmp["score_method"] = model_name
            tmp["users"] = users_targeted.capitalize()

            tmp = tmp.groupby('run_id', group_keys=False).apply(rescale_group, rescale=rescale, include_groups=True)

            dataframes.append(tmp)


    # Concatenate all dataframes into one
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    return combined_dataframe

def filter_name(row):
    if row.conformal_score == "Naive" or row.conformal_score == "Global Harm":
        return row.score_method
    else:
        return r"\textsc{"+f"{row.score_method.capitalize()}"+r"}"

def improve_legend(ax, custom_handles=[], additional_labels_to_remove=[], save_legend=False):
    # Get the current legend and remove it
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

    additional_labels_to_remove += ["Risk Control", "Strategy", "epoch"]

    # Keep only unique legend items (avoid redundant hue/style elements)
    new_labels = [label for label in labels if label not in additional_labels_to_remove]  # Remove duplicates while preserving order
    new_handles = [handles[labels.index(label)] for label in new_labels]

    if len(custom_handles) > 0:
        for cstmlabel, cstmhandle in custom_handles:
            new_labels += [cstmlabel]
            new_handles += [cstmhandle]
    
    # Save legend separately to disk if neede
    if save_legend:
        legend_fig = plt.figure(figsize=(len(new_labels) * 1, 0.4))  # Width based on number of labels
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")
        _ = legend_ax.legend(
            new_handles,
            new_labels,
            loc="center",
            ncol=len(labels),  # Put all items in one row
            frameon=True,
        )
        legend_fig.savefig("legend_only.pdf", bbox_inches='tight', pad_inches=0, format="pdf")
    
    ax.grid(axis='y')

X_AXIS_TEXT = r"Desired reduction in unwanted content (\%)"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    df = load_csv_files_to_dataframe(args.file, rescale=True)

    df["conformal_score"] = df["conformal_score"].fillna("None")
    df["conformal_score"] = df.conformal_score.apply(lambda x : "Similarity" if x=="weights" else x)
    df["conformal_method"] = df["conformal_method"].fillna("None")

    df["Method"] = df["Method"].apply(lambda x: "None" if x == "Classic" else x)
    df["Method"] = df["Method"].apply(lambda x: "Pre-train" if x == "Classic (masked)" else x)
    df["Strategy"] = df.apply(lambda x: f"{x.conformal_method.capitalize()}" if x.Method == "Conformal" else x.Method, axis=1)
    df["Strategy"] = df["Strategy"].apply(lambda x: r"\textsc{Remove}" if x=="Remove" else r"\textsc{Replace} (Ours)")
    df["Risk Control"] = df.apply(lambda x: filter_name(x) if x.Method == "Conformal" else "None", axis=1)

    df = df[df["Risk Control"] != r'\textsc{Ncf}']

    order_methods = [r'\textsc{Lightgcl}', r'\textsc{Gformer}', r'\textsc{Siren}', r'\textsc{Sigformer}']

    max_alpha = df["alpha"].max()

    df.rename(
        columns={"alpha": X_AXIS_TEXT,
                "|S|": r"$|S_\lambda(U)|$",
                "random_items": r"\# of replaced items"},
        inplace=True
    )

    FIGURE_SIZE = (3.2,2)

    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
    g = sns.lineplot(
        data=df[(df.Strategy != "None") & (df["Risk Control"] != "None") & (df["Risk Control"] != r"\textsc{Harm}")],
        x=X_AXIS_TEXT,
        y="nDCG @ k",
        style="Strategy",
        hue="Risk Control",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax = ax,
        hue_order=order_methods
    )
    improve_legend(ax, save_legend=True)
    ax.set_ylabel(r"nDCG @ 20", fontsize="small")
    ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
    fig.savefig("00_ndcg.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
    g = sns.lineplot(
        data=df[(df.Strategy != "None") & (df["Risk Control"] != "None") & (df["Risk Control"] != r"\textsc{Harm}")],
        x=X_AXIS_TEXT,
        y="Recall @ k",
        style="Strategy",
        hue="Risk Control",
        markers=True,
        legend=True,
        dashes=False,
        errorbar="sd",
        ax=ax, hue_order=order_methods
    )
    improve_legend(ax)
    ax.set_ylabel(r"Recall @ 20", fontsize="small")
    ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
    plt.savefig("00_recall.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
    g = sns.lineplot(
        data=df[(df.Strategy != "None") & (df["Risk Control"] != "None") & (df["Risk Control"] != r"\textsc{Harm}")],
        x=X_AXIS_TEXT,
        y = "empr_harmfulness",
        style="Strategy",
        hue="Risk Control",
        markers=True,
        legend=True,
        dashes=False,
        errorbar="sd",
        ax=ax, hue_order=order_methods
    )

    # Add optimal line
    ax.plot([0, 100], [0, 100], linestyle=':', color='black')       
    handle = mlines.Line2D(
                [], [], color="black", linestyle=":"
            )
    improve_legend(ax, custom_handles=[
        ("Optim.", handle)
    ])

    ax.set_ylabel(r"Empirical reduction (\%)", fontsize="small")
    ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
    ax.set(ylim=(105, -5.0))
    plt.savefig("00_harm.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
    g = sns.lineplot(
        data=df[(df.Strategy != "None") & (df["Risk Control"] != "None") & (df["Risk Control"] != r"\textsc{Harm}")],
        x=X_AXIS_TEXT,
        y=r"$|S_\lambda(U)|$",
        style="Strategy",
        hue="Risk Control",
        markers=True,
        legend=True,
        errorbar="sd",
        ax=ax, hue_order=order_methods
    )
    improve_legend(ax)
    ax.set_ylabel(r"$|S_\lambda(U)|$", fontsize="small")
    ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
    plt.savefig("00_size.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=(4,2))
    tmp = df[(df.Strategy == r"\textsc{Replace} (Ours)") & (df["Risk Control"] != "None") & (df["Risk Control"] != r"\textsc{Harm}")]
    tmp = tmp[tmp[X_AXIS_TEXT].isin([12.5, 25, 50, 75, 100])]
    tmp["fraction"] = 100*tmp[r"\# of replaced items"]/tmp[r"$|S_\lambda(U)|$"]
    g = sns.barplot(
        data=tmp,
        x=X_AXIS_TEXT,
        y="fraction",
        hue="Risk Control",
        legend=True,
        errorbar="sd",
        err_kws={"linewidth": 1.2},
        capsize=.4,
        ax=ax, hue_order=order_methods
    )
    improve_legend(ax, save_legend=False)
    ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
    ax.set_ylabel(r"\% of previously seen items", fontsize="small")
    fig.savefig("00_replacement.pdf", format="pdf", bbox_inches='tight')
    plt.clf()