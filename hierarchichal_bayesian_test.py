# hierarchical_posterior_rank_real.py

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from scipy.special import expit


METHODS = [
    # "Baseline",
    "GAUFS",
    "ELSA-mean",
    "FSSEM-k",
    "LS",
    "SPEC",
    "INFFS20",
    "INFFS",
    "FMIUFS",
    "RNE",
]


def build_real_world_dataframe() -> pd.DataFrame:
    """
    AMI values copied from the LaTeX table provided by the user.
    Only AMI is used for the hierarchical posterior rank analysis.
    """
    data = [
        ("01", 0.1039, 0.2509, 0.2686, 0.1039, 0.1039, 0.1039, 0.0286, 0.1039, 0.1039),
        ("02", 0.7792, 0.6984, 0.4697, 0.2518, 0.0157, 0.6129, 0.6940, 0.2594, 0.2960),
        ("03", 0.5236, 0.6689, 0.4562, 0.6410, 0.5437, 0.3504, 0.1717, 0.6797, 0.4531),
        ("04", 0.4898, 0.4431, 0.6206, 0.6380, 0.6542, 0.1026, 0.1026, 0.4269, 0.4269),
        ("05", 0.2337, 0.2552, 0.2233, 0.2339, 0.2337, 0.2337, 0.1230, 0.3437, 0.3437),
        ("06", 0.5810, 0.3007, 0.1310, 0.4029, 0.0163, 0.1907, 0.1439, 0.5810, 0.4215),
        ("07", 0.3279, 0.0999, 0.1207, 0.2029, 0.2420, 0.6389, 0.6389, 0.4184, 0.3549),
        ("08", 0.3325, 0.2566, 0.2671, 0.1106, 0.1671, 0.2451, 0.2451, 0.1309, 0.2451),
        ("09", 0.7316, 0.5975, 0.6314, 0.7316, 0.7316, 0.6900, 0.6900, 0.7316, 0.7316),
        ("10", 0.4234, 0.4574, 0.3358, 0.4169, 0.5036, 0.3088, 0.0820, 0.4658, 0.3844),
        ("11", 0.5288, 0.3598, 0.2577, 0.4437, 0.4748, 0.3765, 0.1758, 0.2506, 0.2420),
        ("12", 0.3355, 0.5487, 0.3468, 0.2445, 0.3004, 0.3359, 0.2201, 0.3295, 0.6590),
        ("13", 0.2124, 0.4953, 0.2872, 0.4666, 0.2050, 0.0851, 0.0851, 0.2434, 0.1770),
        ("14", 0.5855, 0.3819, 0.2703, 0.5291, 0.5471, 0.5362, 0.5452, 0.5373, 0.5334),
        ("15", 0.8583, 0.5496, 0.3668, 0.0997, 0.0997, 0.8342, 0.8199, 0.0997, 0.8080),
        ("16", 0.5928, 0.3861, 0.5193, 0.2916, 0.5928, 0.2396, 0.2396, 0.3448, 0.5928),
        ("17", 0.5407, 0.3377, 0.1949, 0.5407, 0.0115, 0.1225, 0.0186, 0.2152, 0.1636),
        ("18", 0.7363, 0.5383, 0.6486, 0.5330, 0.4340, 0.6491, 0.6221, 0.3707, 0.5967),
        ("19", 0.0084, 0.1120, 0.1659, 0.1241, 0.1051, 0.1596, 0.1596, 0.1924, 0.0834),
        ("20", 0.5025, 0.4627, 0.4971, 0.4449, 0.4614, 0.2266, 0.2293, 0.3688, 0.3160),
        ("21", 0.6066, 0.2882, 0.6050, 0.5628, 0.5845, 0.2869, 0.0040, 0.2880, 0.2869),
        ("22", 0.5691, 0.4674, 0.5926, 0.5012, 0.6163, 0.6110, 0.6026, 0.5904, 0.3897),
        ("23", 0.3649, 0.3031, 0.1895, 0.3119, 0.2564, 0.0815, 0.0884, 0.3392, 0.1397),
        ("24", 0.3600, 0.5937, 0.9201, 0.2910, 0.3205, 0.3600, 0.2833, 0.2767, 0.3074),
        ("25", 0.3365, 0.2714, 0.0591, 0.3321, 0.2950, 0.3492, 0.3296, 0.3760, 0.3365),
        ("26", 0.6935, 0.4646, 0.0000, 0.6715, 0.6975, 0.6893, 0.6893, 0.6715, 0.6715),
        ("27", 0.5119, 0.1633, 0.5920, 0.0073, 0.0000, 0.0421, 0.0421, 0.5119, 0.0170),
        ("28", 0.4640, 0.2685, 0.3013, 0.7141, 0.4671, 0.0387, 0.0387, 0.5250, 0.3929),
        ("29", 0.4481, 0.5949, 0.7075, 0.4191, 0.8264, 0.5125, 0.5125, 0.3551, 0.7998),
        ("30", 0.1125, 0.3419, 0.1738, 0.0839, 0.1825, 0.0387, 0.0107, 0.1741, 0.1308),
    ]

    # data = [
    #     ("01", 0.1039, 0.2686, 0.1039, 0.1039, 0.1039, 0.0286, 0.1039, 0.1039),
    #     ("02", 0.7792, 0.4697, 0.2518, 0.0157, 0.6129, 0.6940, 0.2594, 0.2960),
    #     ("03", 0.5236, 0.4562, 0.6410, 0.5437, 0.3504, 0.1717, 0.6797, 0.4531),
    #     ("04", 0.4898, 0.6206, 0.6380, 0.6542, 0.1026, 0.1026, 0.4269, 0.4269),
    #     ("05", 0.2337, 0.2233, 0.2339, 0.2337, 0.2337, 0.1230, 0.3437, 0.3437),
    #     ("06", 0.5810, 0.1310, 0.4029, 0.0163, 0.1907, 0.1439, 0.5810, 0.4215),
    #     ("07", 0.3279, 0.1207, 0.2029, 0.2420, 0.6389, 0.6389, 0.4184, 0.3549),
    #     ("08", 0.3325, 0.2671, 0.1106, 0.1671, 0.2451, 0.2451, 0.1309, 0.2451),
    #     ("09", 0.7316, 0.6314, 0.7316, 0.7316, 0.6900, 0.6900, 0.7316, 0.7316),
    #     ("10", 0.4234, 0.3358, 0.4169, 0.5036, 0.3088, 0.0820, 0.4658, 0.3844),
    #     ("11", 0.5288, 0.2577, 0.4437, 0.4748, 0.3765, 0.1758, 0.2506, 0.2420),
    #     ("12", 0.3355, 0.3468, 0.2445, 0.3004, 0.3359, 0.2201, 0.3295, 0.6590),
    #     ("13", 0.2124, 0.2872, 0.4666, 0.2050, 0.0851, 0.0851, 0.2434, 0.1770),
    #     ("14", 0.5855, 0.2703, 0.5291, 0.5471, 0.5362, 0.5452, 0.5373, 0.5334),
    #     ("15", 0.8583, 0.3668, 0.0997, 0.0997, 0.8342, 0.8199, 0.0997, 0.8080),
    #     ("16", 0.5928, 0.5193, 0.2916, 0.5928, 0.2396, 0.2396, 0.3448, 0.5928),
    #     ("17", 0.5407, 0.1949, 0.5407, 0.0115, 0.1225, 0.0186, 0.2152, 0.1636),
    #     ("18", 0.7363, 0.6486, 0.5330, 0.4340, 0.6491, 0.6221, 0.3707, 0.5967),
    #     ("19", 0.0084, 0.1659, 0.1241, 0.1051, 0.1596, 0.1596, 0.1924, 0.0834),
    #     ("20", 0.5025, 0.4971, 0.4449, 0.4614, 0.2266, 0.2293, 0.3688, 0.3160),
    #     ("21", 0.6066, 0.6050, 0.5628, 0.5845, 0.2869, 0.0040, 0.2880, 0.2869),
    #     ("22", 0.5691, 0.5926, 0.5012, 0.6163, 0.6110, 0.6026, 0.5904, 0.3897),
    #     ("23", 0.3649, 0.1895, 0.3119, 0.2564, 0.0815, 0.0884, 0.3392, 0.1397),
    #     ("24", 0.3600, 0.9201, 0.2910, 0.3205, 0.3600, 0.2833, 0.2767, 0.3074),
    #     ("25", 0.3365, 0.0591, 0.3321, 0.2950, 0.3492, 0.3296, 0.3760, 0.3365),
    #     ("26", 0.6935, 0.0000, 0.6715, 0.6975, 0.6893, 0.6893, 0.6715, 0.6715),
    #     ("27", 0.5119, 0.5920, 0.0073, 0.0000, 0.0421, 0.0421, 0.5119, 0.0170),
    #     ("28", 0.4640, 0.3013, 0.7141, 0.4671, 0.0387, 0.0387, 0.5250, 0.3929),
    #     ("29", 0.4481, 0.7075, 0.4191, 0.8264, 0.5125, 0.5125, 0.3551, 0.7998),
    #     ("30", 0.1125, 0.1738, 0.0839, 0.1825, 0.0387, 0.0107, 0.1741, 0.1308),
    # ]

    columns = ["dataset"] + METHODS
    return pd.DataFrame(data, columns=columns)


def to_long_form(df_wide: pd.DataFrame) -> pd.DataFrame:
    df_long = df_wide.melt(
        id_vars="dataset",
        var_name="method",
        value_name="ami",
    ).copy()

    df_long["dataset_code"] = pd.Categorical(
        df_long["dataset"],
        categories=df_wide["dataset"].tolist(),
        ordered=True,
    ).codes

    df_long["method_code"] = pd.Categorical(
        df_long["method"],
        categories=METHODS,
        ordered=True,
    ).codes

    return df_long


def rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Convert a matrix of scores into ranks, row by row.
    Higher score = better rank.
    Rank 1 = best.
    """
    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order)
    row_idx = np.arange(values.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, values.shape[1] + 1)
    return ranks


def fit_hierarchical_model(
    df_long: pd.DataFrame,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
):
    """
    Hierarchical Beta model:
        AMI_{dataset,method} ~ Beta(mu * phi, (1-mu) * phi)
        logit(mu) = intercept + dataset_offset[dataset] + method_effect[method]

    dataset_offset accounts for difficulty differences between datasets.
    method_effect captures the global performance of each method.
    """
    y = df_long["ami"].to_numpy(dtype=float)

    # Beta likelihood cannot take exact 0 or 1.
    eps = 1e-6
    y = np.clip(y, eps, 1.0 - eps)

    dataset_idx = df_long["dataset_code"].to_numpy()
    method_idx = df_long["method_code"].to_numpy()

    coords = {
        "dataset": sorted(df_long["dataset"].unique(), key=lambda x: int(x)),
        "method": METHODS,
        "obs_id": np.arange(len(df_long)),
    }

    with pm.Model(coords=coords) as model:
        ds_idx = pm.Data("dataset_idx", dataset_idx, dims="obs_id")
        mt_idx = pm.Data("method_idx", method_idx, dims="obs_id")
        y_obs = pm.Data("y_obs", y, dims="obs_id")

        intercept = pm.Normal("intercept", mu=0.0, sigma=1.5)

        sigma_dataset = pm.HalfNormal("sigma_dataset", sigma=1.0)
        sigma_method = pm.HalfNormal("sigma_method", sigma=1.0)

        dataset_offset = pm.Normal(
            "dataset_offset",
            mu=0.0,
            sigma=sigma_dataset,
            dims="dataset",
        )

        method_raw = pm.Normal(
            "method_raw",
            mu=0.0,
            sigma=1.0,
            dims="method",
        )

        # Center method effects so they are identifiable.
        method_effect_unc = method_raw * sigma_method
        method_effect = pm.Deterministic(
            "method_effect",
            method_effect_unc - pt.mean(method_effect_unc),
            dims="method",
        )

        eta = intercept + dataset_offset[ds_idx] + method_effect[mt_idx]
        mu = pm.Deterministic("mu", pm.math.sigmoid(eta), dims="obs_id")

        phi = pm.Gamma("phi", alpha=2.0, beta=0.1)

        alpha_param = mu * phi
        beta_param = (1.0 - mu) * phi

        pm.Beta(
            "likelihood",
            alpha=alpha_param,
            beta=beta_param,
            observed=y_obs,
            dims="obs_id",
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return idata


def posterior_rank_summary(idata) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build posterior rankings from the fitted method effects.

    We compute posterior expected AMI for an "average dataset":
        mu_avg = sigmoid(intercept + method_effect)

    Since sigmoid is monotonic, this produces the same ordering as the
    method effects, but is easier to interpret.
    """
    posterior = idata.posterior

    intercept = posterior["intercept"].stack(sample=("chain", "draw")).values
    method_effect = (
        posterior["method_effect"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "method")
        .values
    )

    # Posterior expected AMI on an average dataset
    mu_avg = expit(intercept[:, None] + method_effect)

    ranks = rank_desc(mu_avg)

    summary_rows = []
    for j, method in enumerate(METHODS):
        summary_rows.append(
            {
                "method": method,
                "posterior_mean_rank": ranks[:, j].mean(),
                "rank_q025": np.quantile(ranks[:, j], 0.025),
                "rank_q975": np.quantile(ranks[:, j], 0.975),
                "p_best": np.mean(ranks[:, j] == 1),
                "p_top3": np.mean(ranks[:, j] <= 3),
                "posterior_mean_ami": mu_avg[:, j].mean(),
                "ami_q025": np.quantile(mu_avg[:, j], 0.025),
                "ami_q975": np.quantile(mu_avg[:, j], 0.975),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["posterior_mean_rank", "posterior_mean_ami"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return summary_df, ranks, mu_avg


def posterior_pairwise_vs_reference(
    mu_avg: np.ndarray,
    reference_method: str = "GAUFS",
) -> pd.DataFrame:
    """
    Optional: pairwise posterior probabilities vs a reference method.
    """
    ref_idx = METHODS.index(reference_method)

    rows = []
    for j, method in enumerate(METHODS):
        if method == reference_method:
            continue

        diff = mu_avg[:, ref_idx] - mu_avg[:, j]

        rows.append(
            {
                "comparison": f"{reference_method} vs {method}",
                "p_ref_better": np.mean(diff > 0.0),
                "p_ref_worse": np.mean(diff < 0.0),
                "median_diff": np.median(diff),
                "diff_q025": np.quantile(diff, 0.025),
                "diff_q975": np.quantile(diff, 0.975),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="p_ref_better",
        ascending=False,
    ).reset_index(drop=True)


def plot_posterior_rank_distributions(
    ranks: np.ndarray,
    out_path: str = "posterior_rank_distributions_real.png",
) -> None:
    plt.figure(figsize=(11, 5.5))
    plt.violinplot(
        [ranks[:, j] for j in range(ranks.shape[1])],
        showmeans=True,
        showextrema=False,
    )
    plt.xticks(np.arange(1, len(METHODS) + 1), METHODS, rotation=30, ha="right")
    plt.ylabel("Posterior rank (1 = best)")
    plt.title("Hierarchical posterior rank distributions - real-world datasets")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df_wide = build_real_world_dataframe()
    df_long = to_long_form(df_wide)

    print("Wide data shape:", df_wide.shape)
    print("Long data shape:", df_long.shape)

    idata = fit_hierarchical_model(
        df_long=df_long,
        draws=500, #2000
        tune=500, #2000
        chains=2, #4
        target_accept=0.9, #0.95
        random_seed=42,
    )

    summary_df, ranks, mu_avg = posterior_rank_summary(idata)
    pairwise_df = posterior_pairwise_vs_reference(mu_avg, reference_method="GAUFS")

    print("\nPosterior rank summary:")
    print(summary_df.to_string(index=False))

    print("\nPosterior pairwise probabilities vs GAUFS:")
    print(pairwise_df.to_string(index=False))

    summary_df.to_csv("posterior_rank_summary_real.csv", index=False)
    pairwise_df.to_csv("posterior_pairwise_vs_gaufs_real.csv", index=False)
    az.to_netcdf(idata, "hierarchical_posterior_rank_real.nc")

    # plot_posterior_rank_distributions(
    #     ranks,
    #     out_path="posterior_rank_distributions_real.png",
    # )

    print("\nSaved files:")
    print("- posterior_rank_summary_real.csv")
    print("- posterior_pairwise_vs_gaufs_real.csv")
    print("- hierarchical_posterior_rank_real.nc")
    # print("- posterior_rank_distributions_real.png")


if __name__ == "__main__":
    main()