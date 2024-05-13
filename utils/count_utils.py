from collections import Counter

import jax.numpy as jp
import numpy as np

from utils.constants import (
    AGENT_TYPE_DEF,
    AIR_NUTRIENT_RPOS,
    EARTH_NUTRIENT_RPOS,
    EN_ST,
    logger,
)


def count_agents(env):
    try:
        agent_id_grid = env.agent_id_grid
        agent_indices = agent_id_grid.flatten()
        num_agents = len(np.unique(agent_indices))
        logger.debug(f"Total number of agents: {num_agents}")
        return num_agents
    except Exception as e:
        logger.error(f"Error counting agents: {e}")
        return Counter()


def count_nutrients(env, agent_type_def=AGENT_TYPE_DEF):
    try:
        air_indices = env.type_grid == agent_type_def.types.AIR
        soil_indices = env.type_grid == agent_type_def.types.EARTH
        root_indices = env.type_grid[:, :] == agent_type_def.types.AGENT_ROOT
        leaf_indices = env.type_grid[:, :] == agent_type_def.types.AGENT_LEAF
        flower_indices = env.type_grid[:, :] == agent_type_def.types.AGENT_FLOWER

        air_nutrient_col = EN_ST + AIR_NUTRIENT_RPOS
        soil_nutrient_col = EN_ST + EARTH_NUTRIENT_RPOS

        air_air_nutrients = env.state_grid[:, :, air_nutrient_col][air_indices].sum()

        soil_soil_nutrients = env.state_grid[:, :, soil_nutrient_col][
            soil_indices
        ].sum()
        root_air_nutrients = env.state_grid[:, :, air_nutrient_col][root_indices].sum()
        leaf_air_nutrients = env.state_grid[:, :, air_nutrient_col][leaf_indices].sum()
        flower_air_nutrients = env.state_grid[:, :, air_nutrient_col][
            flower_indices
        ].sum()
        root_soil_nutrients = env.state_grid[:, :, soil_nutrient_col][
            root_indices
        ].sum()
        leaf_soil_nutrients = env.state_grid[:, :, soil_nutrient_col][
            leaf_indices
        ].sum()
        flower_soil_nutrients = env.state_grid[:, :, soil_nutrient_col][
            flower_indices
        ].sum()

        nutrient_counts = {
            "Air Nutrients in Air": air_air_nutrients,
            "Soil Nutrients in Soil": soil_soil_nutrients,
            "Air Nutrients in Roots": root_air_nutrients,
            "Air Nutrients in Leafs": leaf_air_nutrients,
            "Air Nutrients in Flowers": flower_air_nutrients,
            "Soil Nutrients in Roots": root_soil_nutrients,
            "Soil Nutrients in Leafs": leaf_soil_nutrients,
            "Soil Nutrients in Flowers": flower_soil_nutrients,
        }

        logger.debug(f"Nutrient counts: {nutrient_counts}")
        return nutrient_counts
    except Exception as e:
        logger.error(f"Error counting nutrients: {e}")
        return {}


def count_agent_types(env, agent_type_def=AGENT_TYPE_DEF):
    try:
        flattened_ids = env.type_grid.flatten()
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


def count_plants(env, agent_type_def=AGENT_TYPE_DEF):
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
