from __future__ import annotations

import abc
import os.path
from collections import Counter
from collections.abc import Callable
from enum import Enum
from multiprocessing import Pool

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


class Validator(abc.ABC):
    def __init__(
        self,
        parameters_list,
        model_name,
        model_short_name,
        use_theoretical,
        sampler_func=None,
        sample_cast_func=None,
        constant_parameters=None,
        faceted_parameters=None,
    ):
        if not isinstance(parameters_list, list):
            parameters_list = [parameters_list]
        self.parameters_list: list[dict] = parameters_list
        self.sampler_func: Callable = sampler_func
        self.sample_cast_func: Callable = sample_cast_func
        self.use_theoretical = use_theoretical
        self.model_name = model_name
        self.model_short_name = model_short_name
        if (
            constant_parameters is not None
            and not isinstance(constant_parameters, list)
            and not isinstance(constant_parameters, tuple)
        ):
            constant_parameters = [constant_parameters]
        self.constant_parameters = constant_parameters
        if not isinstance(faceted_parameters, list) and not isinstance(
            faceted_parameters, tuple
        ):
            faceted_parameters = [faceted_parameters]
        self.faceted_parameters = faceted_parameters

    def sampler(self, sampler_parameters):
        """
        The sampler, returns observations
        """
        if self.sampler_func is None:
            raise ValueError(
                "You need to either provide a `sampler_func` argument or override the "
                "`sampler` method"
            )
        return self.sampler_func(**sampler_parameters)

    def sample_cast(self, sample):
        """
        Cast the samples returned by the sampler to the correct type (tuple for ordinal samplers).
        """
        if self.sampler_func is None:
            raise ValueError(
                "You need to either provide a `sample_cast_func` argument or override "
                "the `sampler_cast` method"
            )
        return self.sample_cast_func(sample)

    def collect_samples(self, num_samples, num_workers=None) -> list[Counter]:
        """
        Collect the suitable number of samples for all set of sampler parameters.
        """
        print(f"Collecting samples for the {self.model_name} model...")
        res = []
        for sampler_parameters in self.parameters_list:
            print(f"\t{sampler_parameters}")
            count = Counter()
            with Pool(processes=num_workers) as pool:
                for sample in pool.imap_unordered(
                    self.sampler, [sampler_parameters] * num_samples
                ):
                    sample = self.sample_cast(sample)
                    count[sample] += 1
            for c in count:
                count[c] /= num_samples
            res.append(count)
        print("\t... collected!")
        return res

    def all_outcomes(self, sampler_parameters):
        """
        Return all possible outcomes for a given configuration of the sampler.
        """
        raise NotImplemented("You need to override the `all_outcomes` method")

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        """
        Returns the theoretical distribution of the outcomes.
        """
        raise NotImplemented(
            "You need to override the `theoretical_distribution` method"
        )

    def all_theoretical_distributions(self) -> list[dict]:
        res = []
        for parameters in self.parameters_list:
            all_outcomes = self.all_outcomes(parameters)
            res.append(self.theoretical_distribution(parameters, all_outcomes))
        return res

    def write_csv(self, num_samples, dir_path, theoretical=None, num_workers=None):
        def parameter_formatting(v):
            if isinstance(v, Enum):
                return v.name
            return str(v)

        if theoretical is None:
            theoretical = self.use_theoretical

        distributions = None
        if theoretical:
            distributions = self.all_theoretical_distributions()

        sample_collections = self.collect_samples(num_samples, num_workers=num_workers)
        param_names = list(self.parameters_list[0].keys())

        print(f"Writing csv the {self.model_name} model...")
        with open(os.path.join(dir_path, f"{self.model_short_name}.csv"), "w") as f:
            f.write("num_samples;" + ";".join(param_names) + ";outcome;observed_freq")
            if theoretical:
                f.write(";theoretical_freq")
            f.write("\n")
            for i, parameters in enumerate(self.parameters_list):
                samples = sample_collections[i]
                if distributions:
                    distribution = distributions[i]
                    main_source = distribution
                else:
                    distribution = None
                    main_source = samples
                for outcome in main_source:
                    f.write(
                        f"{num_samples};"
                        f"{';'.join(parameter_formatting(parameters[h]) for h in param_names)};"
                        f"{outcome};{samples[outcome]}"
                    )
                    if distribution:
                        f.write(f";{distribution[outcome]}")
                    f.write("\n")
        print("\t...written!")

    def plot_frequencies(
        self, csv_dir_path, plot_dir_path, ordering=None, theoretical=None
    ):
        print(f"Plotting frequencies for the {self.model_name} model...")

        if theoretical is None:
            theoretical = self.use_theoretical

        df = pd.read_csv(
            os.path.join(csv_dir_path, f"{self.model_short_name}.csv"), delimiter=";"
        )

        plt.close("all")
        sns.set_context("paper")

        if ordering == "theoretical-observed" and theoretical:
            df = df.sort_values(["theoretical_freq", "observed_freq"], ascending=False)
        elif ordering == "observed-theoretical" and theoretical:
            df = df.sort_values(["observed_freq", "theoretical_freq"], ascending=False)
        elif ordering == "theoretical" and theoretical:
            df = df.sort_values("theoretical_freq", ascending=False)
        else:
            df = df.sort_values("observed_freq", ascending=False)

        if theoretical:
            df = df.melt(
                id_vars=[
                    c
                    for c in df.columns
                    if c not in ("theoretical_freq", "observed_freq")
                ],
                var_name="freq_type",
                value_name="frequency",
            )
        else:
            df.rename(columns={"observed_freq": "frequency"}, inplace=True)

        if not self.faceted_parameters:
            faceted_parameters = [None, None]
        elif len(self.faceted_parameters) == 1:
            faceted_parameters = list(self.faceted_parameters)
            faceted_parameters.append(None)
        else:
            faceted_parameters = self.faceted_parameters
        g = sns.catplot(
            data=df,
            x="outcome",
            y="frequency",
            hue="freq_type" if theoretical else None,
            col=faceted_parameters[1],
            row=faceted_parameters[0],
            kind="bar",
            sharex=False if faceted_parameters[0] else True,
            sharey="row" if faceted_parameters[0] else True,
            legend="full",
        )

        if faceted_parameters[0] is not None:
            if faceted_parameters[1] is None:
                g.set_titles("{row_var} = {row_name}")
            else:
                g.set_titles("{row_var} = {row_name} | {col_var} = {col_name}")
            g.set_xticklabels()
        g.set(xticklabels=[])
        g.set_axis_labels("Outcome identifier", "Frequency")
        title = f"{self.model_name} Model\nnum_samples = {df['num_samples'][0]}"
        constant_params = None
        if self.constant_parameters:
            constant_params = [
                f"{k} = {self.parameters_list[0][k]}" for k in self.constant_parameters
            ]
        if constant_params:
            title += " | " + " | ".join(constant_params)
        title += "\n"
        plt.suptitle(title, fontsize=14, fontweight="bold")
        g.figure.tight_layout()
        if g._legend:
            g._legend.set_title("")
            new_labels = ["Observed", "Theoretical"]
            for t, l in zip(g._legend.texts, new_labels):
                t.set_text(l)
            sns.move_legend(g, "upper right", bbox_to_anchor=(1, 0.8), frameon=True)

        plt.savefig(
            os.path.join(plot_dir_path, f"{self.model_short_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"\t...plotted!")
