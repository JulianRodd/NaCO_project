from utils.constants import logger
from utils.count_utils import count_agent_types, count_nutrients, count_plants
from utils.plotting_utils import filter_and_plot_histogram


class EnvironmentHistory:
    def __init__(self, base_config):
        self.history = []
        self.seasons = []
        if base_config is None:
            logger.error("No base config provided.")
        self.base_config = base_config
        self.image_dir = f"images/years_{base_config.years}-days_{base_config.days_in_year}"
    def add(self, environment, season):
        if season is None:
            logger.warning("No season provided for environment.")
            return
        if environment:
            self.history.append(environment)
            self.seasons.append(season)
            logger.info("Environment added to history.")
        else:
            logger.warning("Attempted to add an empty environment to history.")

    def add_all(self, environments, season):
        if season is None:
            logger.warning("No season provided for environments.")
            return
        if environments:
            self.history.extend(environments)
            self.seasons.extend([season] * len(environments))
            logger.info("Multiple environments added to history.")
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

    def plot_agent_type_hist(self, filter_keys=None):
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
            x_label="Time Point",
            y_label="Count",
            legend_title="Agent Types",
            file_name=file_name,
            filter_keys=filter_keys,
        )

    def plot_nutrient_hist(self, filter_keys=None):
        if not self.history:
            logger.warning("No history available to plot nutrient counts.")
            return

        nutrient_hist = []
        num_environments = len(self.history)

        for i in range(num_environments):
            nutrient_count = count_nutrients(self.history[i])
            nutrient_hist.append(nutrient_count)

        file_name = f"{self.image_dir}/nutrient_hist_{'_'.join(filter_keys) if filter_keys else 'all'}.png".replace(
            " ", "_"
        )
        filter_and_plot_histogram(
            nutrient_hist,
            self.seasons,
            title="History of Nutrient Counts",
            x_label="Time Point",
            y_label="Count",
            legend_title="Nutrients",
            file_name=file_name,
            filter_keys=filter_keys,
        )

    def plot_plant_hist(self, batch_size=30):
        if not self.history:
            logger.warning("No history available to plot plant counts.")
            return

        plant_hist = []
        num_environments = len(self.history)

        for i in range(0, num_environments, batch_size):

            batch_envs = self.history[i : i + batch_size]

            if batch_envs:
                plant_count = count_plants(batch_envs[0])
                batch_data = {"Plant Count": plant_count}

                plant_hist.extend([batch_data] * min(batch_size, len(batch_envs)))

        season_hist = self.seasons[: len(plant_hist)]
        file_name = f"{self.image_dir}/plant_count_{str(batch_size)}.png"
        filter_and_plot_histogram(
            plant_hist,
            season_hist,
            title="History of Plant Counts",
            x_label="Time Point",
            y_label="Count",
            legend_title="",
            file_name=file_name,
        )

    def plot_agent_count_hist(self, batch_size=1):
        if not self.history:
            logger.warning("No history available to plot agent counts.")
            return

        agent_hist = []
        num_environments = len(self.history)

        for i in range(0, num_environments, 1):
            batch_envs = self.history[i : i + batch_size]

            if batch_envs:
                plant_count = count_plants(batch_envs[0])
                batch_data = {"Plant Count": plant_count}

                agent_hist.extend([batch_data] * min(batch_size, len(batch_envs)))

        season_hist = self.seasons[: len(agent_hist)]
        file_name = f"{self.image_dir}/agent_hist_{str(batch_size)}.png"
        filter_and_plot_histogram(
            agent_hist,
            season_hist,
            title="History of Agent Counts",
            x_label="Time Point",
            y_label="Count",
            legend_title="",
            file_name=file_name,
        )

    def __len__(self):
        return len(self.history)

    def __iter__(self):
        return iter(self.history)
