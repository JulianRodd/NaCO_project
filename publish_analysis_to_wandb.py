import os
import pandas as pd
import wandb


# Constants
GENERAL_PATH = "analysis_results/general"
NUTRIENTS_PATH = "analysis_results/nutrients"
GROUPS = ["basic_seasons", "warm_winter_month"]

def get_season(month):
  if month in [12, 1, 2]:
    return 1
  elif month in [3, 4, 5]:
    return 2
  elif month in [6, 7, 8]:
    return 3
  else:
    return 4

def log_general_csv(file_path, group):
    """Logs a general CSV file to Weights & Biases as a separate run."""
    data = pd.read_csv(file_path)
    file_name = os.path.basename(file_path).split('.')[0]
    run = wandb.init(project="environment_simulation_data_advanced", group=group, job_type="general", name=file_name)
    run.define_metric("Month")
    run.define_metric("Season")
    month_number = 1
    for _, row in data.iterrows():
        # Log a plot with the x-axis as the month and three plots for count of agents, average agent age, and average agent SI
        run.log(
            {
                "Month": month_number,
                "Season": get_season(month_number),
                "Agents in Month": row["Total in Month"],
                "Avg Agent Age": row[" Avg Agent Age"],
                "Avg Agent SI": row[" Avg Agent SI"],
            }
        )
        month_number += 1

    run.finish()

def log_nutrient_csv(file_path, group):
    """Logs a nutrient CSV file to Weights & Biases as a separate run."""
    data = pd.read_csv(file_path)
    file_name = os.path.basename(file_path).split('.')[0]

    total_agents_by_month = {month: 0 for month in data["Season"].unique()}
    avg_air_nutrients_by_month = {month: 0 for month in data["Season"].unique()}
    avg_soil_nutrients_by_month =   {month: 0 for month in data["Season"].unique()}
    for agent_type in data["Agent Type"].unique():
      agent_type_name = agent_type.split(" ")[0]
      run = wandb.init(project="environment_simulation_data_advanced", group=group, job_type="nutrients", name=f"{file_name}_{agent_type_name}", tags=[agent_type_name])
      data_by_agent_type = data[data["Agent Type"] == agent_type]
      wandb.define_metric("Month")
      wandb.define_metric("Season")
      month_number = 1
      for _, row in data_by_agent_type.iterrows():
          month = row["Season"]
          avg_air_nutrients = row[" Avg Air Nutrients"]
          avg_soil_nutrients = row[" Avg Soil Nutrients"]
          agent_count = row["Agent Type Count"]
          # Log individual agent type data
          run.log(

              {
                  "Month": month_number,
                  "Season": get_season(month_number),
                  f"Agent Type Count {agent_type_name}": agent_count,
                  f"Avg Air Nutrients {agent_type_name}": avg_air_nutrients,
                  f"Avg Soil Nutrients {agent_type_name}": avg_soil_nutrients,
              }
          )
          month_number += 1

          total_agents_by_month[month] += agent_count
          avg_air_nutrients_by_month[month] += avg_air_nutrients
          avg_soil_nutrients_by_month[month] += avg_soil_nutrients
      run.finish()

    run = wandb.init(project="environment_simulation_data_advanced", group=group, job_type="nutrients", name=f"{file_name}_aggregated")
    run.define_metric("Month")
    run.define_metric("Season")
    month_number = 1
    # Log aggregated data
    for month in total_agents_by_month.keys():
        run.log(
            {
                "Month": month_number,
                "Season": get_season(month_number),
                "Total Agent Count": total_agents_by_month[month],
                "Avg Air Nutrients (Sum)": avg_air_nutrients_by_month[month] / data["Agent Type"].nunique() / 5,
                "Avg Soil Nutrients (Sum)": avg_soil_nutrients_by_month[month] / data["Agent Type"].nunique() / 5,
            }
        )
        month_number += 1
    run.finish()

def process_directory(base_path, data_type):
    """Processes a directory and logs files to Weights & Biases."""
    for group in GROUPS:
        group_path = os.path.join(base_path, group)
        for file_name in os.listdir(group_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(group_path, file_name)
                if data_type == "general":
                    log_general_csv(file_path, group)
                elif data_type == "nutrients":
                    log_nutrient_csv(file_path, group)

def main():
    # Process and log general data files
    process_directory(GENERAL_PATH, "general")

    # Process and log nutrients data files
    process_directory(NUTRIENTS_PATH, "nutrients")

if __name__ == "__main__":
    main()
