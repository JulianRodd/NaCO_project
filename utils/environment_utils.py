import os

from utils.constants import logger
from utils.count_utils import (
    average_agent_age,
    average_agent_structural_integrity,
    count_agent_types,
    count_agents,
    count_plants,
    nutrient_avgs,
)
from utils.plotting_utils import filter_and_plot_histogram


class EnvironmentHistory:
    def __init__(self, base_config, days_since_start=0, folder=""):
        self.history = []
        self.seasons = []
        self.days_since_start = days_since_start
        self.agent_count_hist = None
        if base_config is None:
            logger.error("No base config provided.")
        self.base_config = base_config
        folder = f"/{folder}" if folder else ""
        self.image_dir = f"images{folder}/years_{base_config.years}-days_{base_config.days_in_year}-start_{days_since_start}"
        os.makedirs(self.image_dir, exist_ok=True)

    def add(self, environment, season):
        if season is None:
            logger.warning("No season provided for environment.")
            return
        if environment:
            self.history.append(environment)
            self.seasons.append(season)
            logger.info("Environment added to history.")
            logger.info(f"Current history length: {len(self.history)}")
            plant_count_last = count_plants(self.history[-1])
            logger.info(f"Plant count in last environment: {plant_count_last}\n\n")
        else:
            logger.warning("Attempted to add an empty environment to history.")

    def add_all(self, environments, season):
        if season is None:
            logger.warning("No season provided for environments.")
            return
        if environments:
            self.history.extend(environments)
            self.seasons.extend([season] * len(environments))
            logger.info(f"{len(environments)} environments added to history.")
            logger.info(f"Current history length: {len(self.history)}")
            plant_count_last = count_plants(self.history[-1])
            logger.info(f"Plant count in last environment: {plant_count_last}\n\n")

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

    def plot_agent_type_hist(
        self, filter_keys=None, y_axis_lower_limit=0, y_axis_upper_limit=500
    ):
        if not self.history:
            logger.warning("No history available to plot agent type counts.")
            return
        agent_type_hist = [count_agent_types(env) for env in self.history]
        season_hist = self.seasons
        file_name = f"{self.image_dir}/agent_type_hist_{'_'.join(filter_keys) if filter_keys else 'all'}.png".replace(
            " ", "_"
        )
        filter_and_plot_histogram(
            agent_type_hist,
            season_hist,
            title="History of Agent Type Counts",
            x_label="Day",
            y_label="Count",
            legend_title="Agent Types",
            file_name=file_name,
            filter_keys=filter_keys,
            days_since_start=self.days_since_start,
            y_axis_lower_limit=y_axis_lower_limit,
            y_axis_upper_limit=y_axis_upper_limit,
        )

    def plot_nutrient_hist(
        self, filter_keys=None, y_axis_lower_limit=0, y_axis_upper_limit=5
    ):
        if not self.history:
            logger.warning("No history available to plot nutrient averages.")
            return

        nutrient_hist = []
        num_environments = len(self.history)

        for i in range(num_environments):
            nutrient_count = nutrient_avgs(self.history[i])

            nutrient_hist.append(nutrient_count)

        file_name = f"{self.image_dir}/nutrient_hist_{'_'.join(filter_keys) if filter_keys else 'all'}.png".replace(
            " ", "_"
        )
        filter_and_plot_histogram(
            nutrient_hist,
            self.seasons,
            title="History of Average Nutrient Counts Per Agent",
            x_label="Day",
            y_label="Average Nutrient Count",
            legend_title="Nutrients",
            file_name=file_name,
            filter_keys=filter_keys,
            days_since_start=self.days_since_start,
            y_axis_lower_limit=y_axis_lower_limit,
            y_axis_upper_limit=y_axis_upper_limit,
        )

    def plot_plant_hist(self):

        if not self.history:
            logger.warning("No history available to plot plant counts.")
            return

        plant_hist = []

        for env in self.history:
            plant_count = count_plants(env)

            plant_hist.append({"Plant Count": plant_count})

        season_hist = self.seasons[: len(plant_hist)]
        file_name = f"{self.image_dir}/plant_count.png"
        filter_and_plot_histogram(
            plant_hist,
            season_hist,
            title="History of Plant Counts",
            x_label="Day",
            y_label="Count",
            legend_title="",
            file_name=file_name,
            days_since_start=self.days_since_start,
        )

    def plot_agent_count_hist(self, y_axis_lower_limit=200, y_axis_upper_limit=1000):
        if not self.history:
            logger.warning("No history available to plot agent counts.")
            return

        agent_hist = []

        for env in self.history:
            agent_count = count_agents(env)
            agent_hist.append({"Agent Count": agent_count})

        season_hist = self.seasons[: len(agent_hist)]
        file_name = f"{self.image_dir}/agent_count_hist.png"
        filter_and_plot_histogram(
            agent_hist,
            season_hist,
            title="History of Agent Counts",
            x_label="Day",
            y_label="Count",
            legend_title="",
            file_name=file_name,
            days_since_start=self.days_since_start,
            y_axis_lower_limit=y_axis_lower_limit,
            y_axis_upper_limit=y_axis_upper_limit,
        )

    def plot_avg_agent_age(self):
        if not self.history:
            logger.warning("No history available to plot average agent age.")
            return

        agent_age_hist = []
        num_environments = len(self.history)

        for env in self.history:
            avg_agent_age = average_agent_age(env)
            agent_age_hist.append({"Average Agent Age": avg_agent_age})

        season_hist = self.seasons[: len(agent_age_hist)]
        file_name = f"{self.image_dir}/avg_agent_age_hist.png"
        filter_and_plot_histogram(
            agent_age_hist,
            season_hist,
            title="History of Average Agent Age",
            x_label="Day",
            y_label="Age",
            legend_title="",
            file_name=file_name,
            days_since_start=self.days_since_start,
        )

    def plot_avg_agent_structural_integrity(self):
        if not self.history:
            logger.warning(
                "No history available to plot average agent structural integrity."
            )
            return

        agent_structural_integrity_hist = []

        for env in self.history:
            avg_agent_structural_integrity = average_agent_structural_integrity(env)
            agent_structural_integrity_hist.append(
                {"Average Agent SI": avg_agent_structural_integrity}
            )

        season_hist = self.seasons[: len(agent_structural_integrity_hist)]
        file_name = f"{self.image_dir}/avg_agent_structural_integrity_hist.png"
        filter_and_plot_histogram(
            agent_structural_integrity_hist,
            season_hist,
            title="History of Average Agent Structural Integrity",
            x_label="Day",
            y_label="Structural Integrity",
            legend_title="",
            file_name=file_name,
            days_since_start=self.days_since_start,
        )

    def save_results(self, result_number=0):
        nutrient_per_agent_dict = {
            "january": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "february": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "march": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "april": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "may": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "june": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "july": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "august": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "october": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "november": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
            "december": {
                "avg air nutrients in leaf agent": 0,
                "avg air nutrients in root agent": 0,
                "avg air nutrients in flower agent": 0,
                "avg soil nutrients in leaf agent": 0,
                "avg soil nutrients in root agent": 0,
                "avg soil nutrients in flower agent": 0,
            },
        }

    nutrient

    def __len__(self):
        return len(self.history)

    def __iter__(self):
        return iter(self.history)
