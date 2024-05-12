import logging
from collections import Counter

import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
SEASON_COLORS = {
    "Spring": "palegreen",
    "Summer": "lightcoral",
    "Autumn": "navajowhite",
    "Winter": "lightskyblue",
}

MATERIAL_COLORS = {
    "Void": "black",
    "Air": "lightblue",
    "Soil": "saddlebrown",
    "Immovable": "black",
    "Sun": "orange",
    "Out of Bounds": "red",
    "Unassigned": "grey",
    "Root": "chocolate",
    "Leaf": "lime",
    "Flower": "deeppink",
    "Plant Count": "red",
    "Air Nutrients": "deepskyblue",
    "Soil Nutrients": "saddlebrown",
    "Root Air Nutrients": "chocolate",
    "Root Soil Nutrients": "saddlebrown",
    "Leaf Air Nutrients": "lime",
    "Leaf Soil Nutrients": "green",
    "Flower Air Nutrients": "deeppink",
    "Flower Soil Nutrients": "darkmagenta",
}


class AgentTypeDef:
    class types:
        VOID = 0
        AIR = 1
        EARTH = 2
        IMMOVABLE = 3
        SUN = 4
        OUT_OF_BOUNDS = 5
        AGENT_UNSPECIALIZED = 6
        AGENT_ROOT = 7
        AGENT_LEAF = 8
        AGENT_FLOWER = 9

    def __init__(self):
        self.type_names = {
            self.types.VOID: "Void",
            self.types.AIR: "Air",
            self.types.EARTH: "Soil",
            self.types.IMMOVABLE: "Immovable",
            self.types.SUN: "Sun",
            self.types.OUT_OF_BOUNDS: "Out of Bounds",
            self.types.AGENT_UNSPECIALIZED: "Unassigned",
            self.types.AGENT_ROOT: "Root",
            self.types.AGENT_LEAF: "Leaf",
            self.types.AGENT_FLOWER: "Flower",
        }


agent_type_def = AgentTypeDef()

# grabbed these from environment.py
STR_IDX = 0
AGE_IDX = 1
EN_ST = 2
A_INT_STATE_ST = 4
EARTH_NUTRIENT_RPOS = 0
AIR_NUTRIENT_RPOS = 1


def count_nutrients(env, agent_type_def=agent_type_def):
    try:
        air_indices = (env.type_grid == agent_type_def.types.AIR) | (
            env.type_grid == agent_type_def.types.SUN
        )
        soil_indices = (env.type_grid == agent_type_def.types.EARTH) | (
            env.type_grid == agent_type_def.types.IMMOVABLE
        )
        root_indices = (env.state_grid[:, :, 0] == agent_type_def.types.AGENT_ROOT)
        leaf_indices  = (env.state_grid[:, :, 0] == agent_type_def.types.AGENT_LEAF)
        flower_indices  = (env.state_grid[:, :, 0] == agent_type_def.types.AGENT_FLOWER)

        # pick the dimension of 150 first and then 266 and make a tuple of those

        air_nutrient_col = EN_ST + AIR_NUTRIENT_RPOS
        soil_nutrient_col = EN_ST + EARTH_NUTRIENT_RPOS
        agent_air_nutrient_col = EN_ST
        agent_nutrient_col = EN_ST + 1

        air_nutrients = env.state_grid[:, :, air_nutrient_col][air_indices].sum()
        soil_nutrients = env.state_grid[:, :, soil_nutrient_col][soil_indices].sum()
        root_air_nutrients = env.state_grid[:, :, agent_air_nutrient_col][
            root_indices
        ].sum()
        leaf_air_nutrients = env.state_grid[:, :, agent_air_nutrient_col][
            leaf_indices
        ].sum()
        flower_air_nutrients = env.state_grid[:, :, agent_air_nutrient_col][
            flower_indices
        ].sum()
        root_soil_nutrients = env.state_grid[:, :, agent_nutrient_col][
            root_indices
        ].sum()
        leaf_soil_nutrients = env.state_grid[:, :, agent_nutrient_col][
            leaf_indices
        ].sum()
        flower_soil_nutrients = env.state_grid[:, :, agent_nutrient_col][
            flower_indices
        ].sum()

        nutrient_counts = {
            "Air Nutrients": float(air_nutrients),
            "Soil Nutrients": float(soil_nutrients),
            "Root Air Nutrients": float(root_air_nutrients),
            "Leaf Air Nutrients": float(leaf_air_nutrients),
            "Flower Air Nutrients": float(flower_air_nutrients),
            "Root Soil Nutrients": float(root_soil_nutrients),
            "Leaf Soil Nutrients": float(leaf_soil_nutrients),
            "Flower Soil Nutrients": float(flower_soil_nutrients),
        }

        logger.debug(f"Nutrient counts: {nutrient_counts}")
        return nutrient_counts
    except Exception as e:
        logger.error(f"Error counting nutrients: {e}")
        return {}


def count_agent_types(env, agent_type_def=agent_type_def):
    try:
        flattened_ids = env.state_grid.flatten()
        flattened_ids = jp.array(flattened_ids, dtype=jp.int32)
        id_counts = jp.bincount(flattened_ids, length=len(agent_type_def.type_names))
        readable_counts = Counter(
            {
                agent_type_def.type_names.get(i, f"Unknown Agent {i}"): int(count)
                for i, count in enumerate(id_counts)
                if count > 0
            }
        )
        logger.debug(f"Agent type counts: {readable_counts}")
        return readable_counts
    except Exception as e:
        logger.error(f"Error counting agent types: {e}")
        return Counter()


def count_plants(env, agent_type_def=agent_type_def):
    try:

        type_grid = env.type_grid
        agent_id_grid = env.agent_id_grid

        plant_types = {
            agent_type_def.types.AGENT_ROOT,
            agent_type_def.types.AGENT_LEAF,
            agent_type_def.types.AGENT_FLOWER,
        }

        plant_mask = jp.isin(
            type_grid, jp.array(list(plant_types), dtype=type_grid.dtype)
        )

        plant_agent_ids = jp.where(
            plant_mask, agent_id_grid, jp.zeros_like(agent_id_grid)
        )

        def flood_fill(x, y):
            if (
                x < 0
                or x >= plant_agent_ids.shape[0]
                or y < 0
                or y >= plant_agent_ids.shape[1]
            ):
                return
            if not plant_agent_ids[x, y] or visited[x, y]:
                return

            visited[x, y] = True

            # All 8 possible neighboring directions (horizontal, vertical, and diagonal).
            directions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
            for dx, dy in directions:
                flood_fill(x + dx, y + dy)

        visited = np.zeros_like(plant_agent_ids, dtype=bool)
        num_unique_plants = 0

        for i in range(plant_agent_ids.shape[0]):
            for j in range(plant_agent_ids.shape[1]):
                if plant_agent_ids[i, j] and not visited[i, j]:
                    num_unique_plants += 1
                    flood_fill(i, j)

        logger.debug(f"Total number of plants: {num_unique_plants}")
        return num_unique_plants
    except Exception as e:
        logger.error(f"Error counting plants: {e}")
        return 0


def plot_histogram(data, season_hist, title, x_label, y_label, legend_title, file_name):
    try:
        sorted_keys = sorted(data.keys())
        time_points = range(len(season_hist))

        fig, ax = plt.subplots(figsize=(10, 6))

        lines = []
        for key in sorted_keys:

            if key not in MATERIAL_COLORS:
                color = plt.cm.tab20(sorted_keys.index(key) % 20)
            else:
                color = MATERIAL_COLORS.get(key, "black")
            (line,) = ax.plot(time_points, data[key], label=key, color=color)
            lines.append(line)

        current_season_base = season_hist[0].split()[0]
        start_index = 0
        for i, season in enumerate(season_hist):
            base_season = season.split()[0]  # Extract base season name
            if base_season != current_season_base or i == len(season_hist) - 1:
                end_index = i if i < len(season_hist) - 1 else i + 1
                ax.axvspan(
                    start_index,
                    end_index,
                    color=SEASON_COLORS.get(current_season_base, "grey"),
                    alpha=0.2,
                )
                current_season_base = base_season
                start_index = i

        data_legend = ax.legend(
            lines,
            [l.get_label() for l in lines],
            loc="upper left",
            title=legend_title,
            bbox_to_anchor=(1.05, 1),
        )
        ax.add_artist(data_legend)

        season_patches = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=color,
                markersize=10,
            )
            for season, color in SEASON_COLORS.items()
        ]
        season_legend = ax.legend(
            season_patches,
            SEASON_COLORS.keys(),
            title="Seasons",
            loc="upper left",
            bbox_to_anchor=(1.05, 0.6),
        )

        ax.add_artist(season_legend)

        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.subplots_adjust(right=0.7)
        plt.savefig(file_name)
        logger.info(f"Saved plot to {file_name}")
    except Exception as e:
        logger.error(f"Error plotting histogram: {e}")


def filter_and_plot_histogram(
    hist_data,
    season_hist,
    title,
    x_label,
    y_label,
    legend_title,
    file_name,
    filter_keys=None,
):
    if filter_keys is None:
        filter_keys = set(hist_data[0].keys())

    data = {key: [] for key in filter_keys}
    for counter in hist_data:
        for key in data:
            data[key].append(counter.get(key, 0))

    plot_histogram(data, season_hist, title, x_label, y_label, legend_title, file_name)


def plot_agent_type_hist(agent_type_hist, season_hist, filter_keys=None):
    filter_and_plot_histogram(
        agent_type_hist,
        season_hist,
        title="History of Agent Type Counts",
        x_label="Time Point",
        y_label="Count",
        legend_title="Agent Types",
        file_name="images/agent_type_hist.png",
        filter_keys=filter_keys,
    )


class EnvironmentHistory:
    def __init__(self):
        self.history = []
        self.seasons = []

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
        plot_agent_type_hist(agent_type_hist, self.seasons, filter_keys)

    def plot_nutrient_hist(self, filter_keys=None):
        if not self.history:
            logger.warning("No history available to plot nutrient counts.")
            return

        nutrient_hist = []
        num_environments = len(self.history)

        for i in range(num_environments):
            nutrient_count = count_nutrients(self.history[i])
            nutrient_hist.append(nutrient_count)

        filter_and_plot_histogram(
            nutrient_hist,
            self.seasons,
            title="History of Nutrient Counts",
            x_label="Time Point",
            y_label="Count",
            legend_title="Nutrients",
            file_name="images/nutrient_hist.png",
            filter_keys=filter_keys,
        )

    def plot_plant_hist(self, batch_size=30):
        if not self.history:
            logger.warning("No history available to plot plant counts.")
            return

        plant_hist = []
        num_environments = len(self.history)

        for i in range(0, num_environments, batch_size):
            # Get the environments for the current batch
            batch_envs = self.history[i : i + batch_size]

            # Count plants in the first environment of the batch
            if batch_envs:
                plant_count = count_plants(batch_envs[0])
                batch_data = {"Plant Count": plant_count}

                # Repeat this count for all environments in the batch
                plant_hist.extend([batch_data] * min(batch_size, len(batch_envs)))

        # Ensure the histogram data aligns with the seasons history
        season_hist = self.seasons[: len(plant_hist)]

        filter_and_plot_histogram(
            plant_hist,
            season_hist,
            title="History of Plant Counts",
            x_label="Time Point",
            y_label="Count",
            legend_title="",
            file_name="images/plant_hist.png",
        )

    def __len__(self):
        return len(self.history)

    def __iter__(self):
        return iter(self.history)
