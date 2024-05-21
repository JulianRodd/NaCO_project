import os

import wandb
from utils.constants import logger
from utils.count_utils import (
    average_agent_age,
    average_agent_structural_integrity,
    count_agent_types,
    count_agents,
    count_plants,
    nutrient_avgs,
    nutrient_counts,
)
from utils.general_utils import month_to_number
from utils.plotting_utils import filter_and_plot_histogram


class EnvironmentHistory:
    def __init__(
        self, base_config, days_since_start=0, folder="", sim=0, use_wandb=False
    ):
        self.history = []
        self.seasons = []
        self.months = []
        self.years = []
        self.days_since_start = days_since_start
        self.base_config = base_config
        self.sim = sim
        self.cache = {}

        if not base_config:
            logger.error("No base config provided.")
            return

        folder_path = f"/{folder}" if folder else ""
        self.image_dir = f"images{folder_path}/years_{base_config.years}-days_{base_config.days_in_year}-start_{days_since_start}"
        os.makedirs(self.image_dir, exist_ok=True)
        self.use_wandb = use_wandb

        run_name = f"sim_{sim}_{base_config.name}"
        self.run = wandb.init(
            mode="disabled" if not use_wandb else "online",
            project="naco_simulations",
            settings=wandb.Settings(start_method="fork"),
            name=run_name,
            group=base_config.name,
            tags=[base_config.name, str(sim)],
            config={
                "days_since_start": days_since_start,
                "years": base_config.years,
                "days_in_year": base_config.days_in_year,
                "folder": folder,
                "sim": sim,
            },
        )

    def _cache_result(self, func, env):
        key = (func.__name__, id(env))
        if key not in self.cache:
            self.cache[key] = func(env)
        return self.cache[key]

    def add(self, environment, season, month, year):
        if not season:
            logger.warning("No season provided for environment.")
            return
        if environment:
            self.history.append(environment)
            self.seasons.append(season)
            self.months.append(month)
            self.years.append(year)
            logger.info(
                f"Environment added to history. Current history length: {len(self.history)}"
            )
            plant_count = self._cache_result(count_plants, self.history[-1])
            logger.info(f"Plant count in last environment: {plant_count}\n")

            wandb.log(
                {
                    "month": month_to_number(month),
                    "year": year + 1,
                    "plant_count": plant_count,
                    "total_agents": self._cache_result(count_agents, environment),
                    "average_agent_age": self._cache_result(
                        average_agent_age, environment
                    ),
                    "average_agent_structural_integrity": self._cache_result(
                        average_agent_structural_integrity, environment
                    ),
                    **self._cache_result(nutrient_avgs, environment),
                    **self._cache_result(nutrient_counts, environment),
                }
            )
        else:
            logger.warning("Attempted to add an empty environment to history.")

    def add_all(self, environments, season, month, year):
        if not season:
            logger.warning("No season provided for environments.")
            return
        if environments:
            self.history.extend(environments)
            self.seasons.extend([season] * len(environments))
            self.months.extend([month] * len(environments))
            self.years.extend([year] * len(environments))
            logger.info(
                f"{len(environments)} environments added to history. Current history length: {len(self.history)}"
            )
            plant_count = self._cache_result(count_plants, self.history[-1])
            logger.info(f"Plant count in last environment: {plant_count}\n")

            for env in environments:

                wandb.log(
                    {
                        "month": month_to_number(month),
                        "year": year + 1,
                        "plant_count": self._cache_result(count_plants, env),
                        "total_agents": self._cache_result(count_agents, env),
                        "average_agent_age": self._cache_result(average_agent_age, env),
                        "average_agent_structural_integrity": self._cache_result(
                            average_agent_structural_integrity, env
                        ),
                        **self._cache_result(nutrient_avgs, env),
                        **self._cache_result(nutrient_counts, env),
                    }
                )
        else:
            logger.warning("Attempted to add empty environments to history.")

    def get(self, index):
        try:
            return self.history[index]
        except IndexError:
            logger.error(f"Index {index} out of bounds.")
            return None

    def get_all(self):
        return self.history

    def plot_histogram(
        self,
        data,
        title,
        x_label,
        y_label,
        legend_title,
        file_name,
        filter_keys=None,
        y_axis_limits=None,
    ):
        if not self.history:
            logger.warning(f"No history available to plot {title.lower()}.")
            return
        file_path = f"{self.image_dir}/{file_name}.png".replace(" ", "_")
        filter_and_plot_histogram(
            data,
            self.seasons,
            title=title,
            x_label=x_label,
            y_label=y_label,
            legend_title=legend_title,
            file_name=file_path,
            filter_keys=filter_keys,
            days_since_start=self.days_since_start,
            y_axis_lower_limit=y_axis_limits[0] if y_axis_limits else None,
            y_axis_upper_limit=y_axis_limits[1] if y_axis_limits else None,
        )

    def plot_agent_type_hist(self, filter_keys=None, y_axis_limits=(0, 500)):
        agent_type_hist = [
            self._cache_result(count_agent_types, env) for env in self.history
        ]
        self.plot_histogram(
            agent_type_hist,
            "History of Agent Type Counts",
            "Day",
            "Count",
            "Agent Types",
            "agent_type_hist",
            filter_keys,
            y_axis_limits,
        )

    def plot_air_soil_nutrient_hist(self, filter_keys=None):
        nutrient_hist = [
            self._cache_result(nutrient_counts, env) for env in self.history
        ]
        self.plot_histogram(
            nutrient_hist,
            "History of Nutrient Counts of Soil and Air",
            "Day",
            "Nutrient Count",
            "Nutrients",
            "nutrient_hist_air_soil",
            filter_keys,
        )

    def plot_nutrient_hist(self, filter_keys=None, y_axis_limits=(0, 5)):
        nutrient_hist = [self._cache_result(nutrient_avgs, env) for env in self.history]
        self.plot_histogram(
            nutrient_hist,
            "History of Average Nutrient Counts Per Agent",
            "Day",
            "Average Nutrient Count",
            "Nutrients",
            "nutrient_hist",
            filter_keys,
            y_axis_limits,
        )

    def plot_plant_hist(self):
        plant_hist = [
            {"Plant Count": self._cache_result(count_plants, env)}
            for env in self.history
        ]
        self.plot_histogram(
            plant_hist, "History of Plant Counts", "Day", "Count", "", "plant_count"
        )

    def plot_agent_count_hist(self, y_axis_limits=(200, 1000)):
        agent_hist = [
            {"Agent Count": self._cache_result(count_agents, env)}
            for env in self.history
        ]
        self.plot_histogram(
            agent_hist,
            "History of Agent Counts",
            "Day",
            "Count",
            "",
            "agent_count_hist",
            y_axis_limits=y_axis_limits,
        )

    def plot_avg_agent_age(self):
        agent_age_hist = [
            {"Average Agent Age": self._cache_result(average_agent_age, env)}
            for env in self.history
        ]
        self.plot_histogram(
            agent_age_hist,
            "History of Average Agent Age",
            "Day",
            "Age",
            "",
            "avg_agent_age_hist",
        )

    def plot_avg_agent_structural_integrity(self):
        agent_integrity_hist = [
            {
                "Average Agent SI": self._cache_result(
                    average_agent_structural_integrity, env
                )
            }
            for env in self.history
        ]
        self.plot_histogram(
            agent_integrity_hist,
            "History of Average Agent Structural Integrity",
            "Day",
            "Structural Integrity",
            "",
            "avg_agent_structural_integrity_hist",
        )

    def return_agent_count_of_last_env(self):
        return self._cache_result(count_agents, self.history[-1])

    def save_results(self, result_type: str, result_number=0):
        self.finish()
        self.months = [month.lower() for month in self.months]
        result_type = result_type.lower().replace(" ", "_")
        nutrient_avg_per_agent = {
            month: {
                nutrient: 0
                for nutrient in [
                    "Avg Air Nutrients in Leafs",
                    "Avg Air Nutrients in Roots",
                    "Avg Air Nutrients in Flowers",
                    "Avg Air Nutrients in Unspecializeds",
                    "Avg Soil Nutrients in Leafs",
                    "Avg Soil Nutrients in Roots",
                    "Avg Soil Nutrients in Flowers",
                    "Avg Soil Nutrients in Unspecializeds",
                ]
            }
            for month in self.months
        }
        agent_type_count_per_month = {
            month: {
                "leaf agent count": 0,
                "root agent count": 0,
                "flower agent count": 0,
                "unspecialized agent count": 0,
            }
            for month in self.months
        }
        total_agent_count_per_month = {month: 0 for month in self.months}
        avg_agent_age_per_month = {month: 0 for month in self.months}
        avg_structural_integrity_per_month = {month: 0 for month in self.months}
        month_occurrences = {
            "january": 0,
            "february": 0,
            "march": 0,
            "april": 0,
            "may": 0,
            "june": 0,
            "july": 0,
            "august": 0,
            "september": 0,
            "october": 0,
            "november": 0,
            "december": 0,
        }

        for i, env in enumerate(self.history):
            month = self.months[i].lower()
            month_occurrences[month] += 1

            nutrient_counts = self._cache_result(nutrient_avgs, env)
            agent_type_counts = self._cache_result(count_agent_types, env)
            total_agent_count = self._cache_result(count_agents, env)

            for nutrient, count in nutrient_counts.items():
                nutrient_avg_per_agent[month][nutrient] += count

            for agent_type, count in agent_type_counts.items():
                if agent_type.lower() in ["leaf", "root", "flower", "unspecialized"]:
                    agent_key = f"{agent_type.lower()} agent count"
                    agent_type_count_per_month[month][agent_key] += count

            avg_agent_age_per_month[month] += self._cache_result(average_agent_age, env)
            avg_structural_integrity_per_month[month] += self._cache_result(
                average_agent_structural_integrity, env
            )
            total_agent_count_per_month[month] += total_agent_count

        for month in nutrient_avg_per_agent:
            for nutrient in nutrient_avg_per_agent[month]:
                nutrient_avg_per_agent[month][nutrient] /= month_occurrences[month]

            for agent in agent_type_count_per_month[month]:
                agent_type_count_per_month[month][agent] /= month_occurrences[month]

            total_agent_count_per_month[month] /= month_occurrences[month]
            avg_agent_age_per_month[month] /= month_occurrences[month]
            avg_structural_integrity_per_month[month] /= month_occurrences[month]

        os.makedirs(f"analysis_results/nutrients/{result_type}", exist_ok=True)
        os.makedirs(f"analysis_results/general/{result_type}", exist_ok=True)

        with open(
            f"analysis_results/nutrients/{result_type}/sim_{result_number}.csv", "w"
        ) as result_file:
            result_file.write(
                "Season,Agent Type,Agent Type Count, Avg Air Nutrients, Avg Soil Nutrients\n"
            )
            for month, nutrient_data in nutrient_avg_per_agent.items():
                for agent, count in agent_type_count_per_month[month].items():
                    air_nutrient = nutrient_data[
                        f"Avg Air Nutrients in {agent.split()[0].title()}s"
                    ]
                    soil_nutrient = nutrient_data[
                        f"Avg Soil Nutrients in {agent.split()[0].title()}s"
                    ]
                    result_file.write(
                        f"{month},{agent},{int(count)},{air_nutrient},{soil_nutrient}\n"
                    )

        with open(
            f"analysis_results/general/{result_type}/sim_{result_number}.csv", "w"
        ) as result_file:
            result_file.write("Season,Total in Month, Avg Agent Age, Avg Agent SI\n")
            for month in nutrient_avg_per_agent:
                result_file.write(
                    f"{month}, {int(total_agent_count_per_month[month])},{avg_agent_age_per_month[month]},{avg_structural_integrity_per_month[month]}\n"
                )

    def __len__(self):
        return len(self.history)

    def __iter__(self):
        return iter(self.history)

    def finish(self):
        if self.use_wandb and self.run:
            self.run.finish()

    def __del__(self):
        self.finish()
