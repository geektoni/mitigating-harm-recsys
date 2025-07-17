import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

import os

from analytics.plot_figure_3_and_4 import load_csv_files_to_dataframe, filter_name, X_AXIS_TEXT

sns.set_theme(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    }, palette="brg")

def improve_legend(ax, custom_handles=[]):
    # Get the current legend and remove it
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

    # Keep only unique legend items (avoid redundant hue/style elements)
    new_labels = [label for label in labels if label != "users" and label != "Strategy"]  # Remove duplicates while preserving order
    new_handles = [handles[labels.index(label)] for label in new_labels]

    if len(custom_handles) > 0:
        for cstmlabel, cstmhandle in custom_handles:
            new_labels += [cstmlabel]
            new_handles += [cstmhandle]

    # Add a clean legend
    ax.legend(new_handles, new_labels, fontsize='xx-small')

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)
args = parser.parse_args()

df = load_csv_files_to_dataframe(args.file)

df["beta"] = df.beta.apply(lambda x : f"$W_\%>{str(round(x*100))}\%$" if x != -1 else "None")
df["Strategy"] = df.apply(lambda x: f"{x.conformal_method.capitalize()}" if x.Method == "Conformal" else x.Method, axis=1)
df["Risk Control"] = df.apply(lambda x: r"\textsc{"+f"{x.conformal_score}"+r"}" if x.Method == "Conformal" else "None", axis=1)
df["Risk Control"] = df.apply(lambda x: filter_name(x) if x.Method == "Conformal" else "None", axis=1)
df["Strategy"] = df["Strategy"].apply(lambda x: r"\textsc{Remove}" if x=="Remove" else r"\textsc{Replace} (Ours)")
df["users"] = df["users"].apply(lambda x: f"Low-reporting" if x=="Easy" else "High-reporting")


df.rename(
    columns={"alpha": r"\% of undersired items",
             "|S|": r"$|S(X)|$",
             "random_items": r"\# of replaced items"},
    inplace=True
)

# Figure size
FIGURE_SIZE = (3.2,2)

# Consider only the replace strategy
df = df[(df["Risk Control"] == r"\textsc{Lightgcl}")]

fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
g = sns.lineplot(
    data=df,
    x=r"\% of undersired items",
    y="nDCG @ k",
    hue="users",
    style="Strategy",
    markers=True,
    errorbar="sd",
    ax = ax,
)
improve_legend(ax)
ax.set_ylabel(r"nDCG @ 20", fontsize="small")
ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
ax.grid(axis="y")

plt.savefig("00_users_ndcg.pdf", format="pdf", bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
g = sns.lineplot(
    data=df,
    x=r"\% of undersired items",
    y="empr_harmfulness",
    style="Strategy",
    hue="users",
    markers=True,
    legend=True,
    errorbar="sd",
    ax=ax
)
improve_legend(ax)
ax.plot([0, 100], [0, 100], linestyle=':', color='black', label="Optim.")
ax.set(ylim=(105, -5.0))
ax.set_ylabel(r"Empirical reduction (\%)", fontsize="small")
ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
ax.grid(axis="y")

plt.savefig("00_users_harm.pdf", format="pdf", bbox_inches='tight')
plt.clf()