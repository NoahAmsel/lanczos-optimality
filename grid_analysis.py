import glob

import pandas as pd
import seaborn as sns


def read_all(out_folder="output/grid_search/"):
    # files = glob.glob(out_folder + "/*.tsv")
    files = [
        out_folder + "2023-02-11_21:31.tsv",
        out_folder + "2023-02-12_00:18.tsv",
        out_folder + "2023-02-12_01:58.tsv",
        out_folder + "2023-02-12_23:00.tsv",
        out_folder + "2023-02-13_16:28.tsv",
        out_folder + "2023-02-13_16:36.tsv",
    ]
    dfs = [pd.read_csv(file, sep="\t") for file in files]
    return pd.concat(dfs)


# combo = read_all()
# grid = combo[(combo.width == combo.width.unique()[14]) & (combo.kappa == 1e4)].groupby(["b0", "q"]).ratio_argmax.max()

from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams.update({'figure.autolayout': True})
# df1 = pd.read_csv("output/grid_search/2023-02-11_21:31.tsv", sep="\t")

# for ix, width in enumerate([df1.width.unique()[4], df1.width.unique()[14]]):
#     print(width)
#     fig = plt.figure()
#     sns.heatmap(df1[df1.width == width].pivot(index="b0", columns="q", values="ratio_max"))
#     fig.savefig(f"output/heat{ix}.png")

dfnew = pd.read_csv("output/grid_search/2023-02-14_20:55.tsv", sep="\t")
sns.heatmap(dfnew.pivot(index="b0", columns="q", values="ratio_max")).get_figure().savefig("output/heat_narrow.png")