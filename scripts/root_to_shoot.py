from glob import glob

from pprint import pprint
import pandas as pd
import wandb
from scipy.stats import ttest_ind, ttest_rel


def main():
    separate_counts_dir = "/home/janneke/Documents/Master/Natural Computing/NaCO_project/analysis_results/nutrients"

    root_counts, leaf_counts = get_root_to_shoot_ratio(separate_counts_dir)

    run = wandb.init(project="naco_statistics", job_type="general", name=f"root-to-shoot")
    run.define_metric("Root-to-leaves Ratio")
    run.define_metric("Month")

    for month, month_data_root, month_data_leaf in zip(range(1,13), root_counts.values(), leaf_counts.values()):
        normal_ratio = sum(month_data_root["normal"])/12 - sum(month_data_leaf["normal"])/12
        warm_ratio = sum(month_data_root["warm"])/12 - sum(month_data_leaf["warm"])/12

        wandb.log({
            "Month": month,
            "Normal": normal_ratio,
            "Warm": warm_ratio
        })

    run.finish()


def get_root_to_shoot_ratio(dir: str):
    sim_files_normal = glob(f"{dir}/basic_seasons/*")
    sim_files_warm = glob(f"{dir}/warm_winter_month/*")

    root_counts = {}
    leaf_counts = {}

    for file_month_normal, file_month_warm in zip(sim_files_normal, sim_files_warm):
        normal_count = pd.read_csv(file_month_normal)
        normal_root_count = list(normal_count[normal_count["Agent Type"] == "root agent count"]["Agent Type Count"])
        normal_leaf_count = list(normal_count[normal_count["Agent Type"] == "leaf agent count"]["Agent Type Count"])

        warm_count = pd.read_csv(file_month_warm)
        warm_root_count = list(warm_count[warm_count["Agent Type"] == "root agent count"]["Agent Type Count"])
        warm_leaf_count = list(warm_count[warm_count["Agent Type"] == "leaf agent count"]["Agent Type Count"])

        for month in range(1, 13):
            if month not in root_counts:
                root_counts[month] = {"normal": [], "warm": []}
                leaf_counts[month] = {"normal": [], "warm": []}
            
            root_counts[month]["normal"].append(normal_root_count[month - 1])
            leaf_counts[month]["normal"].append(normal_leaf_count[month - 1])

            root_counts[month]["warm"].append(warm_root_count[month - 1])
            leaf_counts[month]["warm"].append(warm_leaf_count[month - 1])
        
    return root_counts, leaf_counts


if __name__ == "__main__":
    main()