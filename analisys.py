from glob import glob

from pprint import pprint
import pandas as pd
import wandb
from scipy.stats import ttest_ind, ttest_rel


def main():
    total_counts_dir = "/home/janneke/Documents/Master/Natural Computing/NaCO_project/analysis_results/general"
    separate_counts_dir = "/home/janneke/Documents/Master/Natural Computing/NaCO_project/analysis_results/nutrients"

    total_counts = get_total_agent_counts(total_counts_dir)
    flower_counts, root_counts, leaf_counts = get_separate_agent_counts(separate_counts_dir)

    counts_dict = {
        "total": total_counts, 
        "root": root_counts, 
        "leaf": leaf_counts, 
        "flower": flower_counts, 
    }

    pprint(counts_dict)

    run = wandb.init(project="environment_simulation_data_advanced", job_type="general", name="t_testing")
    run.define_metric("P-value")
    run.define_metric("T-statistic")
    run.define_metric("Month")

    for agent_type, year_data in counts_dict.items():
        for month, month_data in year_data.items():
            normal_counts = month_data["normal"]
            warm_counts = month_data["warm"]

            result = ttest_rel(normal_counts, warm_counts)
            wandb.log({
                "Month": month,
                f"P-value {agent_type}": result["pvalue"],
                f"T-statistic {agent_type}": result["statistic"],
            })

    run.finish()
    

def get_total_agent_counts(dir: str):
    sim_files_normal = glob(f"{dir}/basic_seasons/*")
    sim_files_warm = glob(f"{dir}/warm_winter_month/*")

    counts = {}

    for file_month_normal, file_month_warm in zip(sim_files_normal, sim_files_warm):
        normal_count = list(pd.read_csv(file_month_normal)["Total in Month"])
        warm_count = list(pd.read_csv(file_month_warm)["Total in Month"])

        for month in range(1, 13):
            if month not in counts:
                counts[month] = {"normal": [], "warm": []}
            counts[month]["normal"].append(normal_count[month - 1])
            counts[month]["warm"].append(warm_count[month - 1])

    return counts


def get_separate_agent_counts(dir: str):
    sim_files_normal = glob(f"{dir}/basic_seasons/*")
    sim_files_warm = glob(f"{dir}/warm_winter_month/*")

    flower_counts = {}
    root_counts = {}
    leaf_counts = {}

    for file_month_normal, file_month_warm in zip(sim_files_normal, sim_files_warm):
        normal_count = pd.read_csv(file_month_normal)
        normal_flower_count = list(normal_count[normal_count["Agent Type"] == "flower agent count"]["Agent Type Count"])
        normal_root_count = list(normal_count[normal_count["Agent Type"] == "root agent count"]["Agent Type Count"])
        normal_leaf_count = list(normal_count[normal_count["Agent Type"] == "leaf agent count"]["Agent Type Count"])

        warm_count = pd.read_csv(file_month_warm)
        warm_flower_count = list(warm_count[warm_count["Agent Type"] == "flower agent count"]["Agent Type Count"])
        warm_root_count = list(warm_count[warm_count["Agent Type"] == "root agent count"]["Agent Type Count"])
        warm_leaf_count = list(warm_count[warm_count["Agent Type"] == "leaf agent count"]["Agent Type Count"])

        for month in range(1, 13):
            if month not in flower_counts:
                flower_counts[month] = {"normal": [], "warm": []}
                root_counts[month] = {"normal": [], "warm": []}
                leaf_counts[month] = {"normal": [], "warm": []}
            
            flower_counts[month]["normal"].append(normal_flower_count[month - 1])
            root_counts[month]["normal"].append(normal_root_count[month - 1])
            leaf_counts[month]["normal"].append(normal_leaf_count[month - 1])

            flower_counts[month]["warm"].append(warm_flower_count[month - 1])
            root_counts[month]["warm"].append(warm_root_count[month - 1])
            leaf_counts[month]["warm"].append(warm_leaf_count[month - 1])

    return flower_counts, root_counts, leaf_counts


if __name__ == "__main__":
    main()