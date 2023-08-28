from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path

import baryrat
import flamp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
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
    def name(self): pass
    @abstractmethod
    def generate_data(self): pass
    @abstractmethod
    def plot_data(self, _): pass

    def data_path(self): return f"{self.output_folder}/data/{self.name()}.pkl"
    def plot_path(self): return f"{self.output_folder}/plots/{self.name()}.svg"

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
    def convergence_plot(self, data, figsize, plot_optimality_ratio, style_df):
        fig, axs = plt.subplots(
            (2 if plot_optimality_ratio else 1), len(data),
            squeeze=False,
            sharex=True,
            height_ratios=[1, 0.25] if plot_optimality_ratio else [1],
            figsize=figsize,
        )
        for i, (label, relative_error_df) in enumerate(data.items()):
            experiments.plot_convergence_curves(
                relative_error_df,
                relative_error=True,
                ax=axs[0, i],
                title=label,
                **style_df.transpose().to_dict()
            )

            if plot_optimality_ratio:
                optimality_ratios = relative_error_df["Lanczos-FA"] / relative_error_df["Instance Optimal"]
                sns.lineplot(data=optimality_ratios, ax=axs[1, i], lw=1.5).set(
                    ylabel="Optimality Ratio" if (i == 0) else None
                )
            axs[0, i].set(xlabel=None)
            if i > 0:
                axs[0, i].set(ylabel=None)
                axs[0, i].legend([], [], frameon=False)

        fig.supxlabel("Number of iterations ($k$)")
        fig.tight_layout()
        return fig

    @staticmethod
    def master_style_df():
        our_bound_style = [(3, 1, 1, 1), 1.5, sns.color_palette("rocket", 4)[3]]
        return pd.DataFrame({
            "FOV Optimal": [(3, 1), 1.5, sns.color_palette("rocket", 4)[1]],
            "Spectrum Optimal": [(2, 1, 1, 1, 1, 1), 1.5, sns.color_palette("tab10")[-1]],
            "Theorem 1": our_bound_style,
            "Theorem 2": our_bound_style,
            "Theorem 3": our_bound_style,
            "Lanczos-FA": [(1, 1), 3, sns.color_palette("rocket", 4)[2]],
            "Instance Optimal": [(1, 0), 1, sns.color_palette("rocket", 4)[0]],
        }, index=["dashes", "sizes", "palette"])


class Sec4Plotter(ConvergencePlotter):
    def name(self):
        return "sec4"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.)
        lambda_min = flamp.gmpy2.mpfr(1.)
        b = flamp.ones(dim)
        def inv_sqrt(x): return 1 / flamp.sqrt(x)
        ks = list(range(1, 61))

        data = {}

        a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-3, lambda_1=lambda_min)
        inv_sqrt_problem = mf.DiagonalFAProblem(inv_sqrt, a_diag_geom, b, cache_k=max(ks))
        data[r"$\mathbf A^{-1/2}\mathbf b$"] = pd.DataFrame(index=ks, data={
            "FOV Optimal": [experiments.fact1(inv_sqrt_problem, k, max_iter=100, n_grid=1000, tol=1e-14) for k in tqdm(ks)],
            "Theorem 2": [experiments.thm2(inv_sqrt_problem, k, max_iter=100, tol=1e-14) for k in tqdm(ks)],
            "Lanczos-FA": [inv_sqrt_problem.lanczos_error(k) for k in tqdm(ks)],
            "Instance Optimal": [inv_sqrt_problem.instance_optimal_error(k) for k in tqdm(ks)]
        }) / mf.norm(inv_sqrt_problem.ground_truth())

        a_diag_cluster = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
        sqrt_problem = mf.DiagonalFAProblem(flamp.sqrt, a_diag_cluster, b, cache_k=max(ks))
        data[r"$\mathbf A^{1/2}\mathbf b$"] = pd.DataFrame(index=ks, data={
            "FOV Optimal": [experiments.fact1(sqrt_problem, k, max_iter=100, n_grid=1000, tol=1e-14) for k in tqdm(ks)],
            "Theorem 3": [experiments.thm2(sqrt_problem, k, max_iter=100, tol=1e-14) for k in tqdm(ks)],
            "Lanczos-FA": [sqrt_problem.lanczos_error(k) for k in tqdm(ks)],
            "Instance Optimal": [sqrt_problem.instance_optimal_error(k) for k in tqdm(ks)]
        }) / mf.norm(sqrt_problem.ground_truth())

        return data

    def plot_data(self, data):
        fig = self.convergence_plot(data, (8, 4), False, self.master_style_df())
        # add back the legend
        handles, labels = fig.axes[0].get_legend_handles_labels()
        labels[2] = "Theorem 2 (left)\nTheorem 3 (right)"
        fig.axes[0].legend(reversed(handles), reversed(labels))
        return fig


class GeneralPerformancePlotter(ConvergencePlotter):
    def name(self):
        return "general_performance"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.)
        lambda_min = flamp.gmpy2.mpfr(1.)
        a_diag_unif = flamp.linspace(lambda_min, kappa*lambda_min, dim)
        a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
        a_diag_two_cluster = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
        geom_b = flamp.to_mp(mf.geometric_spectrum(dim, 1e4, 1000))
        ks = list(range(1, 61))
        problems = {
            r"$\mathbf A^{-2}\mathbf b$": mf.DiagonalFAProblem(experiments.InverseMonomial(2), a_diag_unif, geom_b, cache_k=max(ks)),
            r"$\exp(\mathbf A)\mathbf b$": mf.DiagonalFAProblem(flamp.exp, a_diag_geom, geom_b, cache_k=max(ks)),
            r"$\log(\mathbf A)\mathbf b$": mf.DiagonalFAProblem(flamp.log, a_diag_two_cluster, geom_b, cache_k=max(ks)),
        }
        return {
            label: pd.DataFrame(index=ks, data={
                "FOV Optimal": [experiments.fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in tqdm(ks)],
                "Spectrum Optimal": [p.spectrum_optimal_error(k, max_iter=100, tol=1e-14) for k in tqdm(ks)],
                "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
                "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)]
            }) / mf.norm(p.ground_truth()) for label, p in problems.items()
        }

    def plot_data(self, data):
        return self.convergence_plot(data, (8, 4.75), True, self.master_style_df())


class OurBoundPlotter(ConvergencePlotter):
    def name(self):
        return "our_bound"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.)
        lambda_min = flamp.gmpy2.mpfr(1.)
        a_diag_unif = flamp.linspace(lambda_min, kappa*lambda_min, dim)
        a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
        a_diag_two_cluster = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
        b = flamp.ones(dim)
        ks = list(range(1, 61))
        problems = {
            r"$\mathbf A^{-2}\mathbf b$": mf.DiagonalFAProblem(experiments.InverseMonomial(2), a_diag_unif, b, cache_k=max(ks)),
            r"$r(\mathbf A)\mathbf b \approx \mathbf A^{.9} \mathbf b$ (deg=5)": mf.DiagonalFAProblem(baryrat.brasil(lambda x: x**(.9), (flamp.gmpy2.mpfr(1), flamp.gmpy2.mpfr(kappa)), 5, info=False), a_diag_geom, b, cache_k=max(ks)),
            r"$r(\mathbf A)\mathbf b \approx \log(\mathbf A)\mathbf b$ (deg=10)": mf.DiagonalFAProblem(baryrat.brasil(flamp.log, (flamp.gmpy2.mpfr(1), flamp.gmpy2.mpfr(kappa)), 10, info=False), a_diag_two_cluster, b, cache_k=max(ks)),
        }
        relative_error_dfs = {
            label: pd.DataFrame(index=ks, data={
                "FOV Optimal": [experiments.fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in tqdm(ks)],
                "Theorem 1": [experiments.thm1(p, k) for k in tqdm(ks)],
                "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
                "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)]
            }) / mf.norm(p.ground_truth()) for label, p in problems.items()
        }
        return relative_error_dfs

    def plot_data(self, data):
        return self.convergence_plot(data, (8, 3.75), False, self.master_style_df())


class SqrtVsRationalPlotter(ConvergencePlotter):
    def name(self):
        return "sqrt_vs_rat"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.)
        lambda_min = flamp.gmpy2.mpfr(1.)
        a_diag = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
        b = flamp.ones(dim)
        ks = list(range(1, 41))

        ground_truth_problem = mf.DiagonalFAProblem(flamp.sqrt, a_diag, b, cache_k=max(ks))
        df_cols = {
            f"deg={deg}": [
                ground_truth_problem.lanczos_on_approximant_error(
                    k, baryrat.brasil(flamp.sqrt, (a_diag.min(), a_diag.max()), deg)
                )
                for k in tqdm(ks)
            ]
            for deg in [5, 10, 15, 20]
        }
        df_cols["Square root"] = [ground_truth_problem.lanczos_error(k) for k in tqdm(ks)]
        return pd.DataFrame(index=ks, data={
            **df_cols, "Square root": [ground_truth_problem.lanczos_error(k) for k in tqdm(ks)]
        }) / mf.norm(ground_truth_problem.ground_truth())

    def plot_data(self, data):
        title = ""
        sns.set_palette(sns.color_palette("rocket", 5))
        return self.convergence_plot({title: data}, (5.4, 3.75), False, pd.DataFrame())


class IndefinitePlotter(ConvergencePlotter):
    def name(self):
        return "indefinite"

    def generate_data(self):
        dim = 100
        kappa = flamp.gmpy2.mpfr(100.)
        lambda_min = flamp.gmpy2.mpfr(1.)
        geom_spectrum = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
        a_diag = np.hstack([-geom_spectrum, geom_spectrum])
        b = flamp.ones(2 * dim)
        ks = list(range(1, 61))
        problems = {
            r"$\mathrm{sign}(\mathbf A)\mathbf b$": mf.DiagonalFAProblem(np.sign, a_diag, b, cache_k=max(ks)),
            r"$(5 - \mathbf A^2)^{-1} \mathbf b$": mf.DiagonalFAProblem(experiments.InversePolynomial(np.polynomial.Polynomial([5, 0, -1])), a_diag, b, cache_k=max(ks)),
            r"$(5 + \mathbf A^2)^{-1} \mathbf b$": mf.DiagonalFAProblem(experiments.InversePolynomial(np.polynomial.Polynomial([5, 0, 1])), a_diag, b, cache_k=max(ks))
        }
        relative_error_dfs = {
            label: pd.DataFrame(index=ks, data={
                "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
                "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)]
            }) / mf.norm(p.ground_truth()) for label, p in problems.items()
        }
        return relative_error_dfs

    def plot_data(self, data):
        style_df = self.master_style_df().drop("sizes")
        return self.convergence_plot(data, (8, 4.75), True, style_df)


class GenericOptLowerBoundPlotter(PaperPlotter):
    @abstractstaticmethod
    def build_norm_matrix(a_diag, q):
        pass

    def generate_data(self):
        dim = 100
        # For speed, we restrict k = 7 here,
        # but we find the the results are unchanged if larger ks are considered as well.
        # The biggest ratios always seem to occur in the first few iterations
        ks = list(range(1, 7))
        results = []
        for kappa, q, high_cluster_width in tqdm(product(
            [10**2, 10**3, 10**4, 10**5, 10**6],
            [2, 4, 8, 16, 32, 64],
            np.geomspace(0.5e-5, 0.5e0, num=20)
        )):
            kappa = flamp.gmpy2.mpfr(kappa)
            a_diag = mf.two_cluster_spectrum(
                dim, kappa, low_cluster_size=1, high_cluster_width=high_cluster_width
            )
            opt_b0, opt_ratio = experiments.worst_b0(
                experiments.InverseMonomial(q),
                a_diag, ks, (1e-8, 1),
                norm_matrix_sqrt=self.build_norm_matrix(a_diag, q),
                xatol=1e-10
            )
            results.append(dict(
                kappa=kappa,
                q=q,
                high_cluster_width=high_cluster_width,
                dimension=dim,
                b0=opt_b0,
                ratio=opt_ratio,
            ))
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
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        palette = sns.color_palette("rocket", 5)
        sns.scatterplot(x='q', y='ratio', hue='log_kappa', data=data, legend=False, palette=palette, ax=axs[0], s=60)
        sns.lineplot(x=data.q.unique(), y=np.sqrt(data.q.unique() * data.kappa.max()), ax=axs[0], lw=1.5, ls=':')
        axs[0].set(xscale='log', yscale='log', xlabel=r"$q$", ylabel=r'Max Optimality Ratio ($C$)')
        axs[0].set_xscale('log', base=2)

        sns.scatterplot(x='kappa', y='ratio', hue='log_kappa', data=data, legend=False, palette=palette, ax=axs[1], s=60)
        sns.lineplot(x=data.kappa.unique(), y=np.sqrt(data.kappa.unique() * data.q.max()), ax=axs[1], lw=1.5, ls=':')
        axs[1].set(xscale='log', yscale='log', xlabel=r'$\kappa$', ylabel='')
        fig.tight_layout()
        return fig


class LanczosORLowerPlotter(GenericOptLowerBoundPlotter):
    def name(self):
        return "lanczos_OR_lower"

    @staticmethod
    def build_norm_matrix(a_diag, q):
        return mf.DiagonalMatrix(flamp.sqrt(a_diag ** q))

    def plot_data(self, data):
        data = data.astype(float)
        data = data.groupby(["kappa", "q"])["ratio"].max().reset_index()
        data["log_kappa"] = np.log10(data["kappa"])
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        palette = sns.color_palette("rocket", 5)
        sns.scatterplot(x='q', y='ratio', hue='log_kappa', data=data, legend=False, palette=palette, ax=axs[0], s=60)
        sns.lineplot(x=np.geomspace(data.q.min(), data.q.max()), y=(data.kappa.max() ** (np.geomspace(data.q.min(), data.q.max()) / 2)), ax=axs[0], lw=1.5, ls=':')
        axs[0].set(xscale='log', yscale='log', xlabel=r"$q$", ylabel=r'Max Optimality Ratio ($C$)')
        axs[0].set_xscale('log', base=2)

        sns.scatterplot(x='kappa', y='ratio', hue='log_kappa', data=data, legend=False, palette=palette, ax=axs[1], s=60)
        sns.lineplot(x=data.kappa.unique(), y=(data.kappa.unique() ** (data.q.max() / 2)), ax=axs[1], lw=1.5, ls=':')
        axs[1].set(xscale='log', yscale='log', xlabel=r'$\kappa$', ylabel='')
        fig.tight_layout()
        return fig


if __name__ == "__main__":
    flamp.set_dps(300)  # compute with this many decimal digits precision
    print(f"Using {flamp.get_dps()} digits of precision")

    # sns.set(font_scale=2)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    output_folder = "output/paper_output"
    use_cache = True

    Sec4Plotter(output_folder).plot(use_cache)
    GeneralPerformancePlotter(output_folder).plot(use_cache)
    OurBoundPlotter(output_folder).plot(use_cache)
    SqrtVsRationalPlotter(output_folder).plot(use_cache)
    IndefinitePlotter(output_folder).plot(use_cache)
    OptLowerBoundPlotter(output_folder).plot(use_cache)
    LanczosORLowerPlotter(output_folder).plot(use_cache)
