from paper_plots import *

output_folder = "output/paper_output"
use_cache = True
flamp.set_dps(300)  # compute with this many decimal digits precision
print(f"Using {flamp.get_dps()} digits of precision")

# sns.set(font_scale=2)
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

plotter = GeneralPerformancePlotter(output_folder)
with open(plotter.data_path(), "rb") as f:
    data = pkl.load(f)

data2 = {
    k: v.drop("Spectrum Optimal", axis=1).rename(
        {"FOV Optimal": "Uniform Approximation Bound"}, axis=1
    )
    for k, v in data.items()
}
data1 = {k: v.drop("Instance Optimal", axis=1) for k, v in data2.items()}
style_df = plotter.master_style_df()
style_df["Uniform Approximation Bound"] = style_df["FOV Optimal"]
plotter.convergence_plot(data1, (12, 4.5), False, style_df).savefig("chris1.svg")
plotter.convergence_plot(data2, (12, 4.5), False, style_df).savefig("chris2.svg")
