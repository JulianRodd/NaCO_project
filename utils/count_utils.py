from collections import Counter

import jax.numpy as jp
import numpy as np

from utils.constants import (
    AGE_IDX,
    AGENT_TYPE_DEF,
    AIR_NUTRIENT_RPOS,
    EARTH_NUTRIENT_RPOS,
    EN_ST,
    STR_IDX,
    logger,
)


def count_agents(env):
    try:
        agent_id_grid = env.agent_id_grid
        num_agents = np.count_nonzero(agent_id_grid)
        logger.debug(f"Total number of agents: {num_agents}")
        return num_agents
    except Exception as e:
        logger.error(f"Error counting agents: {e}")
        return 0


def average_agent_age(env, num_agents=None):
    try:
        agent_age_grid = env.state_grid[:, :, AGE_IDX]
        if num_agents is None:
            num_agents = count_agents(env)
        if num_agents == 0:
            return 0
        total_age = agent_age_grid.sum()
        average_age = total_age / num_agents if num_agents > 0 else 0
        logger.debug(f"Average agent age: {average_age}")
        return average_age
    except Exception as e:
        logger.error(f"Error calculating average agent age: {e}")
        return 0


def average_agent_structural_integrity(env, num_agents=None):
    try:

        agent_structural_integrity_grid = env.state_grid[:, :, STR_IDX]
        if num_agents is None:
            num_agents = count_agents(env)
        total_structural_integrity = agent_structural_integrity_grid.sum()
        average_structural_integrity = (
            total_structural_integrity / num_agents if num_agents > 0 else 0
        )
        logger.debug(
            f"Average agent structural integrity: {average_structural_integrity}"
        )
        return average_structural_integrity
    except Exception as e:
        logger.error(f"Error calculating average agent structural_integrity: {e}")
        return 0


def _calculate_nutrient_counts(env, indices, nutrient_col):
    return env.state_grid[:, :, nutrient_col][indices].sum()


def count_nutrients(env, agent_type_def=AGENT_TYPE_DEF):
    try:
        type_grid = env.type_grid
        nutrient_counts = {}

        def add_nutrient_count(label, indices, nutrient_col):
            nutrient_counts[label] = _calculate_nutrient_counts(
                env, indices, nutrient_col
            )

        air_indices = type_grid == agent_type_def.types.AIR
        soil_indices = type_grid == agent_type_def.types.EARTH
        root_indices = type_grid[:, :] == agent_type_def.types.AGENT_ROOT
        leaf_indices = type_grid[:, :] == agent_type_def.types.AGENT_LEAF
        flower_indices = type_grid[:, :] == agent_type_def.types.AGENT_FLOWER

        air_nutrient_col = EN_ST + AIR_NUTRIENT_RPOS
        soil_nutrient_col = EN_ST + EARTH_NUTRIENT_RPOS

        add_nutrient_count("Air Nutrients in Air", air_indices, air_nutrient_col)
        add_nutrient_count("Soil Nutrients in Soil", soil_indices, soil_nutrient_col)
        add_nutrient_count("Air Nutrients in Roots", root_indices, air_nutrient_col)
        add_nutrient_count("Air Nutrients in Leafs", leaf_indices, air_nutrient_col)
        add_nutrient_count("Air Nutrients in Flowers", flower_indices, air_nutrient_col)
        add_nutrient_count("Soil Nutrients in Roots", root_indices, soil_nutrient_col)
        add_nutrient_count("Soil Nutrients in Leafs", leaf_indices, soil_nutrient_col)
        add_nutrient_count(
            "Soil Nutrients in Flowers", flower_indices, soil_nutrient_col
        )

        logger.debug(f"Nutrient counts: {nutrient_counts}")
        return nutrient_counts
    except Exception as e:
        logger.error(f"Error counting nutrients: {e}")
        return {}


def count_agent_types(env, agent_type_def=AGENT_TYPE_DEF):
    try:
        flattened_ids = jp.array(env.type_grid.flatten(), dtype=jp.int32)
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


def count_plants(env):
    try:
        agent_id_grid = env.agent_id_grid
        agent_id_grid_nonzero = agent_id_grid[agent_id_grid != 0]
        num_plants = len(np.unique(agent_id_grid_nonzero.flatten()))
        logger.debug(f"Total number of plants: {num_plants}")
        return num_plants
    except Exception as e:
        logger.error(f"Error counting plants: {e}")
        return 0
