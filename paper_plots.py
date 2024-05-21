from abc import ABC, abstractmethod
from pathlib import Path

import baryrat
import flamp
from fire import Fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.io as sio
import seaborn as sns
from tqdm import tqdm
from tqdm.contrib.itertools import product

import experiments
import matrix_functions as mf


class PaperPlotter(ABC):
    def __init__(self, output_folder):
        self.output_folder = output_folder
        Path(f"{self.output_folder}/data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_folder}/plots").mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def plot_data(self, _):
        pass

    def data_path(self):
        return f"{self.output_folder}/data/{self.name()}.pkl"

    def plot_path(self):
        return f"{self.output_folder}/plots/{self.name()}.svg"

    def plot(self, use_saved_data=False):
        print(self.name())
        if use_saved_data:
            with open(self.data_path(), "rb") as f:
                data = pkl.load(f)
        else:
            data = self.generate_data()
            with open(self.data_path(), "wb") as f:
                pkl.dump(data, f)
        fig = self.plot_data(data)
        fig.savefig(self.plot_path())
        return fig


class ConvergencePlotter(PaperPlotter):
    def convergence_plot(self, data, figsize, plot_optimality_ratio, style_df, already_long_fmt=False, legend_ix=0):
        fig, axs = plt.subplots(
            (2 if plot_optimality_ratio else 1),
            len(data),
            squeeze=False,
            sharex=True,
            height_ratios=[1, 0.25] if plot_optimality_ratio else [1],
            figsize=figsize,
        )
        for i, (label, relative_error_df) in enumerate(data.items()):
            experiments.plot_convergence_curves(
                relative_error_df,
                relative_error=True,
                already_long_fmt=already_long_fmt,
                ax=axs[0, i],
                title=label,
                **style_df.transpose().to_dict(),
            )

            if plot_optimality_ratio:
                optimality_ratios = (
                    relative_error_df["Lanczos-FA"]
                    / relative_error_df["Instance Optimal"]
                )
                sns.lineplot(
                    data=optimality_ratios.astype(float),
                    ax=axs[1, i],
                    lw=1.5,
                    color="k",
                ).set(ylabel="Optimality Ratio" if (i == 0) else None)
            axs[0, i].set(xlabel=None)
            if i != 0:
                axs[0, i].set(ylabel=None)
            if i != legend_ix:
                axs[0, i].legend([], [], frameon=False)

        fig.supxlabel("Number of iterations ($k$)")
        fig.tight_layout()
        return fig

    @staticmethod
    def master_style_df():
        fov_optimal_style = [(3, 1), 1.5, sns.color_palette("rocket", 4)[1]]
        our_bound_style = [(3, 1, 1, 1), 1.5, sns.color_palette("rocket", 4)[3]]
        return pd.DataFrame(
            {
                "FOV Optimal": fov_optimal_style,
                "Fact 1": fov_optimal_style,
                "Spectrum Optimal": [
                    (2, 1, 1, 1, 1, 1),
                    1.5,
                    sns.color_palette("husl", 8)[1],
                ],
                "Theorem 2.1": our_bound_style,
                "Theorem 3.1": our_bound_style,
                "Theorem 3.2": our_bound_style,
                "Lanczos-FA": [(1, 1), 3, sns.color_palette("rocket", 4)[2]],
                "Instance Optimal": [(1, 0), 1, sns.color_palette("rocket", 4)[0]],
            },
            index=["dashes", "sizes", "palette"],
        )


class Sec4Plotter(ConvergencePlotter):
    def name(self):
        return "sec4"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.0)
        lambda_min = flamp.gmpy2.mpfr(1.0)
        b = flamp.to_mp(mf.geometric_spectrum(dim, 1e4, 1000))

        def inv_sqrt(x):
            return 1 / flamp.sqrt(x)

        ks = list(range(1, 61))

        data = {}

        a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-3, lambda_1=lambda_min)
        inv_sqrt_problem = mf.DiagonalFAProblem(
            inv_sqrt, a_diag_geom, b, cache_k=max(ks)
        )
        data[r"$\mathbf A^{-1/2}\mathbf b$"] = pd.DataFrame(
            index=ks,
            data={
                "Fact 1": [
                    experiments.fact1(
                        inv_sqrt_problem, k, max_iter=100, n_grid=1000, tol=1e-14
                    )
                    for k in tqdm(ks)
                ],
                "Theorem 3.1": [
                    experiments.thm2(inv_sqrt_problem, k, max_iter=100, tol=1e-14)
                    for k in tqdm(ks)
                ],
                "Lanczos-FA": [inv_sqrt_problem.lanczos_error(k) for k in tqdm(ks)],
                "Instance Optimal": [
                    inv_sqrt_problem.instance_optimal_error(k) for k in tqdm(ks)
                ],
            },
        ) / mf.norm(inv_sqrt_problem.ground_truth())

        a_diag_cluster = mf.two_cluster_spectrum(
            dim, kappa, low_cluster_size=10, lambda_1=lambda_min
        )
        sqrt_problem = mf.DiagonalFAProblem(
            flamp.sqrt, a_diag_cluster, b, cache_k=max(ks)
        )
        data[r"$\mathbf A^{1/2}\mathbf b$"] = pd.DataFrame(
            index=ks,
            data={
                "FOV Optimal": [
                    experiments.fact1(
                        sqrt_problem, k, max_iter=100, n_grid=1000, tol=1e-14
                    )
                    for k in tqdm(ks)
                ],
                "Theorem 3.2": [
                    experiments.thm2(sqrt_problem, k, max_iter=100, tol=1e-14)
                    for k in tqdm(ks)
                ],
                "Lanczos-FA": [sqrt_problem.lanczos_error(k) for k in tqdm(ks)],
                "Instance Optimal": [
                    sqrt_problem.instance_optimal_error(k) for k in tqdm(ks)
                ],
            },
        ) / mf.norm(sqrt_problem.ground_truth())

        return data

    def plot_data(self, data):
        fig = self.convergence_plot(data, (8, 4), False, self.master_style_df())
        # add back the legend
        handles, labels = fig.axes[0].get_legend_handles_labels()
        labels[2] = "Theorem 3.1 (left)\nTheorem 3.2 (right)"
        fig.axes[0].legend(reversed(handles), reversed(labels))
        return fig


class GeneralPerformancePlotter(ConvergencePlotter):
    def name(self):
        return "general_performance"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.0)
        lambda_min = flamp.gmpy2.mpfr(1.0)
        a_diag_unif = flamp.linspace(lambda_min, kappa * lambda_min, dim)
        a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
        a_diag_two_cluster = mf.two_cluster_spectrum(
            dim, kappa, low_cluster_size=10, lambda_1=lambda_min
        )
        geom_b = flamp.to_mp(mf.geometric_spectrum(dim, 1e4, 1000))
        ks = list(range(1, 61))
        problems = {
            r"$\mathbf A^{\!-2}\,\mathbf b$": mf.DiagonalFAProblem(
                experiments.InverseMonomial(2), a_diag_unif, geom_b, cache_k=max(ks)
            ),
            r"$\exp(-\mathbf A / 10)\mathbf b$": mf.DiagonalFAProblem(
                lambda x: flamp.exp(-x / 10), a_diag_geom, geom_b, cache_k=max(ks)
            ),
            r"$\log(\mathbf A)\mathbf b$": mf.DiagonalFAProblem(
                flamp.log, a_diag_two_cluster, geom_b, cache_k=max(ks)
            ),
        }
        return {
            label: pd.DataFrame(
                index=ks,
                data={
                    "Fact 1": [
                        p.fov_optimal_error_remez(
                            k, max_iter=100, n_grid=1000, tol=1e-14
                        )
                        for k in tqdm(ks)
                    ],
                    # "Spectrum Optimal": [
                    #     p.spectrum_optimal_error(k, max_iter=100, tol=1e-14)
                    #     for k in tqdm(ks)
                    # ],
                    "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
                    "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)],
                },
            )
            / mf.norm(p.ground_truth())
            for label, p in problems.items()
        }

    def plot_data(self, data):
        return self.convergence_plot(data, (8, 3.5), True, self.master_style_df(), legend_ix=2)


class CIQPlotter(ConvergencePlotter):
    def __init__(self, dim, q, output_folder):
        super().__init__(output_folder)
        self.dim = dim
        self.q = q

    def name(self):
        return "CIQ"

    def generate_data(self):
        t = np.array(list(range(1, self.dim + 1)))
        a_diag_sqrt = 1/np.sqrt(t)
        a_diag_square = 1/(t**2)
        # a_diag_exp = np.exp(-t) + 1e-10
        b = np.random.randn(self.dim)
        ks = list(range(1, 200))
        problems = {
            r"$\lambda_t = 1/\sqrt{t}$": mf.DiagonalSqrtAProblem(
                a_diag_sqrt, b, cache_k=max(ks)
            ),
            r"$\lambda_t = 1/t^2$": mf.DiagonalSqrtAProblem(
                a_diag_square, b, cache_k=max(ks)
            ),
            # r"$\lambda_t = e^{-t} + 10^{-10}$": mf.DiagonalSqrtAProblem(
            #     a_diag_exp, b, cache_k=max(ks)
            # ),
        }
        return {
            label: pd.DataFrame(
                index=ks,
                data={
                    f"CIQ {self.q}": [p.ciq_error(self.q, k) for k in tqdm(ks)],
                    f"Zolotarev-CG {self.q}": [p.zolotarev_lanczos_error(self.q, k) for k in tqdm(ks)],
                    f"Zolotarev {self.q}": [p.zolotarev_error(self.q)] * len(ks),
                    "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
                },
            )
            / mf.norm(p.ground_truth())
            for label, p in problems.items()
        }

    def plot_data(self, data):
        df = self.master_style_df()
        df[f"CIQ {self.q}"] = df["Instance Optimal"]
        df[f"Zolotarev-CG {self.q}"] = df["FOV Optimal"]
        df[f"Zolotarev {self.q}"] = df["Theorem 2.1"]
        return self.convergence_plot(data, (8, 4.75), False, df)


class OurBoundPlotter(ConvergencePlotter):
    def name(self):
        return "our_bound"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.0)
        lambda_min = flamp.gmpy2.mpfr(1.0)
        a_diag_unif = flamp.linspace(lambda_min, kappa * lambda_min, dim)
        a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
        a_diag_two_cluster = mf.two_cluster_spectrum(
            dim, kappa, low_cluster_size=10, lambda_1=lambda_min
        )
        b = flamp.to_mp(mf.geometric_spectrum(dim, 1e4, 1000))
        ks = list(range(1, 61))
        problems = {
            r"$\mathbf A^{\!-2}\,\mathbf b$": mf.DiagonalFAProblem(
                experiments.InverseMonomial(2), a_diag_unif, b, cache_k=max(ks)
            ),
            r"$r(\mathbf A)\mathbf b \approx \exp(-\mathbf A / 10)\mathbf b$ (deg=5)": mf.DiagonalFAProblem(
                experiments.ExpRationalApprox(
                    a_diag_geom.min(), a_diag_geom.max(), -1 / 10, 5
                ),
                a_diag_geom,
                b,
                cache_k=max(ks),
            ),
            r"$r(\mathbf A)\mathbf b \approx \log(\mathbf A)\mathbf b$ (deg=10)": mf.DiagonalFAProblem(
                baryrat.brasil(
                    flamp.log,
                    (flamp.gmpy2.mpfr(1), flamp.gmpy2.mpfr(kappa)),
                    10,
                    info=False,
                ),
                a_diag_two_cluster,
                b,
                cache_k=max(ks),
            ),
        }
        relative_error_dfs = {
            label: pd.DataFrame(
                index=ks,
                data={
                    "Fact 1": [
                        experiments.fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14)
                        for k in tqdm(ks)
                    ],
                    "Theorem 2.1": [experiments.thm1(p, k) for k in tqdm(ks)],
                    "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
                    "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)],
                },
            )
            / mf.norm(p.ground_truth())
            for label, p in problems.items()
        }
        return relative_error_dfs

    def plot_data(self, data):
        return self.convergence_plot(data, (8, 2.75), False, self.master_style_df(), legend_ix=2)


class SqrtVsRationalPlotter(ConvergencePlotter):
    def name(self):
        return "sqrt_vs_rat"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.0)
        lambda_min = flamp.gmpy2.mpfr(1.0)
        a_diag = mf.two_cluster_spectrum(
            dim, kappa, low_cluster_size=10, lambda_1=lambda_min
        )
        b = flamp.ones(dim)
        ks = list(range(1, 41))

        def f(x):
            return x ** (-0.4)

        ground_truth_problem = mf.DiagonalFAProblem(f, a_diag, b, cache_k=max(ks))

        # this is a separate function to ensure that `rational_approx` is calculated only once per deg
        # and reused for each k
        def rational_approx_convergence(deg):
            rational_approx = baryrat.brasil(f, (a_diag.min(), a_diag.max()), deg)
            return [
                ground_truth_problem.lanczos_on_approximant_error(k, rational_approx)
                for k in tqdm(ks)
            ]

        df_cols = {
            f"deg={deg}": rational_approx_convergence(deg) for deg in [5, 10, 15, 20]
        }
        df_cols["Lanczos-FA"] = [
            ground_truth_problem.lanczos_error(k) for k in tqdm(ks)
        ]
        return pd.DataFrame(index=ks, data={**df_cols}) / mf.norm(
            ground_truth_problem.ground_truth()
        )

    def plot_data(self, data):
        title = ""
        sns.set_palette(sns.color_palette("rocket", 5))
        return self.convergence_plot({title: data}, (5.4, 2.5), False, pd.DataFrame())


class IndefinitePlotter(ConvergencePlotter):
    def name(self):
        return "indefinite"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.0)
        lambda_min = flamp.gmpy2.mpfr(1.0)
        geom_spectrum = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
        a_diag = np.hstack([-geom_spectrum, geom_spectrum])
        b = flamp.ones(2 * dim)
        ks = list(range(1, 61))
        problems = {
            r"$\mathrm{sign}(\mathbf A)\mathbf b$": mf.DiagonalFAProblem(
                np.sign, a_diag, b, cache_k=max(ks)
            ),
            r"$(5 - \mathbf A^2)^{-1} \mathbf b$": mf.DiagonalFAProblem(
                experiments.InversePolynomial(np.polynomial.Polynomial([5, 0, -1])),
                a_diag,
                b,
                cache_k=max(ks),
            ),
            r"$(5 + \mathbf A^2)^{-1} \mathbf b$": mf.DiagonalFAProblem(
                experiments.InversePolynomial(np.polynomial.Polynomial([5, 0, 1])),
                a_diag,
                b,
                cache_k=max(ks),
            ),
        }
        relative_error_dfs = {
            label: pd.DataFrame(
                index=ks,
                data={
                    "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
                    "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)],
                },
            )
            / mf.norm(p.ground_truth())
            for label, p in problems.items()
        }
        return relative_error_dfs

    def plot_data(self, data):
        style_df = self.master_style_df().drop("sizes")
        return self.convergence_plot(data, (8, 4.75), True, style_df)


class GenericOptLowerBoundPlotter(PaperPlotter):
    @staticmethod
    @abstractmethod
    def build_norm_matrix(a_diag, q):
        pass

    def generate_data(self):
        dim = 100
        # For speed, we restrict k = 7 here,
        # but we find the the results are unchanged if larger ks are considered as well.
        # The biggest ratios always seem to occur in the first few iterations
        ks = list(range(1, 7))
        results = []
        for kappa, q, high_cluster_width in tqdm(
            product(
                [10**3, 10**4, 10**5, 10**6], [2, 4, 8, 16, 32, 64], [0.5e-5]
            )
        ):
            kappa = flamp.gmpy2.mpfr(kappa)
            a_diag = mf.two_cluster_spectrum(
                dim, kappa, low_cluster_size=1, high_cluster_width=high_cluster_width
            )
            opt_b0, opt_ratio = experiments.worst_b0(
                experiments.InverseMonomial(q),
                a_diag,
                ks,
                (1e-8, 1),
                norm_matrix_sqrt=self.build_norm_matrix(a_diag, q),
                xatol=1e-10,
            )
            results.append(
                dict(
                    kappa=kappa,
                    q=q,
                    high_cluster_width=high_cluster_width,
                    dimension=dim,
                    b0=opt_b0,
                    ratio=opt_ratio,
                )
            )
        return pd.DataFrame(results)


class OptLowerBoundPlotter(GenericOptLowerBoundPlotter):
    def name(self):
        return "opt_lower_bound"

    @staticmethod
    def build_norm_matrix(a_diag, q):
        return None

    def plot_data(self, data):
        data = data.astype(float)
        data = data.groupby(["kappa", "q"])["ratio"].max().reset_index()
        data["log_kappa"] = np.log10(data["kappa"])
        fig, axs = plt.subplots(1, 2, figsize=(8, 2.75))
        palette = sns.color_palette("rocket", data["kappa"].nunique())
        sns.scatterplot(
            x="q",
            y="ratio",
            hue="log_kappa",
            data=data,
            legend=False,
            palette=palette,
            ax=axs[0],
            s=60,
        )
        axs[0].plot(
            data.q.unique(),
            np.sqrt(data.q.unique() * data.kappa.max()),
            lw=1.5,
            ls=":",
            color="k",
        )
        axs[0].set(
            xscale="log",
            yscale="log",
            xlabel=r"$q$",
            ylabel=r"Max Optimality Ratio ($C$)",
        )
        axs[0].set_xscale("log", base=2)

        sns.scatterplot(
            x="kappa",
            y="ratio",
            hue="log_kappa",
            data=data,
            legend=False,
            palette=palette,
            ax=axs[1],
            s=60,
        )
        axs[1].plot(
            data.kappa.unique(),
            np.sqrt(data.kappa.unique() * data.q.max()),
            lw=1.5,
            ls=":",
            color="k",
        )
        axs[1].set(xscale="log", yscale="log", xlabel=r"$\kappa$", ylabel="")
        fig.tight_layout()
        return fig


class LanczosORLowerPlotter(GenericOptLowerBoundPlotter):
    def name(self):
        return "lanczos_OR_lower"

    @staticmethod
    def build_norm_matrix(a_diag, q):
        return mf.DiagonalMatrix(flamp.sqrt(a_diag**q))

    def plot_data(self, data):
        data = data.astype(float)
        data = data.groupby(["kappa", "q"])["ratio"].max().reset_index()
        data["log_kappa"] = np.log10(data["kappa"])
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        palette = sns.color_palette("rocket", data["kappa"].nunique())
        sns.scatterplot(
            x="q",
            y="ratio",
            hue="log_kappa",
            data=data,
            legend=False,
            palette=palette,
            ax=axs[0],
            s=60,
        )
        axs[0].plot(
            np.geomspace(data.q.min(), data.q.max()),
            (data.kappa.max() ** (np.geomspace(data.q.min(), data.q.max()) / 2)),
            lw=1.5,
            ls=":",
            color="k",
        )
        axs[0].set(
            xscale="log",
            yscale="log",
            xlabel=r"$q$",
            ylabel=r"Max Optimality Ratio ($C$)",
        )
        axs[0].set_xscale("log", base=2)

        sns.scatterplot(
            x="kappa",
            y="ratio",
            hue="log_kappa",
            data=data,
            legend=False,
            palette=palette,
            ax=axs[1],
            s=60,
        )
        axs[1].plot(
            data.kappa.unique(),
            (data.kappa.unique() ** (data.q.max() / 2)),
            lw=1.5,
            ls=":",
            color="k",
        )
        axs[1].set(xscale="log", yscale="log", xlabel=r"$\kappa$", ylabel="")
        fig.tight_layout()
        return fig


class JinSidfordPlotter(ConvergencePlotter):
    def name(self):
        return "jin_sidford"

    @staticmethod
    def load_matlab_data(path):
        state = sio.loadmat(path)
        df_rat = pd.DataFrame({
            "Number of iterations ($k$)": state["rational_count"][:, 0],
            "Relative Error": state["rational_error"][:, 0],
            "Line": pd.Series(state["rational_deg_list"][:, 0]).apply(lambda x: f"``rational'' deg={x}"),
        }).sort_values("Line", ascending=False)
        df_slanczos = pd.DataFrame({
            "Number of iterations ($k$)": state["slanczos_count"][:, 0],
            "Relative Error": state["slanczos_error"][:, 0],
            "Line": pd.Series(state["slanczos_deg_list"][:, 0]).apply(lambda x: f"``slanczos'' deg={x}"),
        }).sort_values("Line", ascending=False)
        df_lan = pd.DataFrame({
            "Number of iterations ($k$)": state["real_lanczos_count"][:, 0],
            "Relative Error": state["real_lanczos_error"][:, 0]
        })
        df_lan["Line"] = "Lanczos-FA"
        df_all = pd.concat([df_lan, df_slanczos, df_rat], axis=0)
        df_all["Number of iterations ($k$)"] = np.floor(df_all["Number of iterations ($k$)"]).astype(int)
        df_all = df_all[df_all["Number of iterations ($k$)"] <= 250]
        return df_all

    def generate_data(self):
        return {
            "Eigengap Uniform": self.load_matlab_data("noah_eigengap_unif.mat"),
            "Eigengap Skewed": self.load_matlab_data("noah_eigengap_skewed.mat"),
            "No Eigengap Skewed": self.load_matlab_data("noah_no_eigengap_skewed.mat")
        }

    def plot_data(self, data):
        sns.set_palette(sns.color_palette("rocket", 5))
        fig = self.convergence_plot(data, (8, 2.7), False, pd.DataFrame(), already_long_fmt=True)
        fig.subplots_adjust(bottom=0.37)
        fig.axes[0].legend(loc='upper center', bbox_to_anchor=(1.7, -0.15), ncol=3)
        fig.supxlabel("Number of matvecs (or equivalent in vector-vector products)")
        return fig


def main(output_folder, use_cache=False):
    flamp.set_dps(300)  # compute with this many decimal digits precision
    print(f"Using {flamp.get_dps()} digits of precision")

    # sns.set(font_scale=2)
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r'\usepackage{newtxtext,newtxmath}', "font.family": "serif"})

    GeneralPerformancePlotter(output_folder).plot(use_cache)
    OurBoundPlotter(output_folder).plot(use_cache)
    SqrtVsRationalPlotter(output_folder).plot(use_cache)
    Sec4Plotter(output_folder).plot(use_cache)
    IndefinitePlotter(output_folder).plot(use_cache)
    OptLowerBoundPlotter(output_folder).plot(use_cache)
    LanczosORLowerPlotter(output_folder).plot(use_cache)
    JinSidfordPlotter(output_folder).plot()

    # WARNING: On the 1/t^2 spectrum, Zolotarev approx should be getting < 10^-6 according to Pleiss!
    # TODO: add exponentially decaying spectrum
    # CIQPlotter(2500, 8, "output/paper_output").plot(False)


if __name__ == "__main__":
    Fire(main)
