from collections import Counter

import jax.numpy as jp


class AgentTypeDef:
    class types:
        AGENT_UNSPECIALIZED = 0
        AGENT_ROOT = 1
        AGENT_LEAF = 2
        AGENT_FLOWER = 3

    def __init__(self):
        self.type_names = {
            self.types.AGENT_UNSPECIALIZED: "Agent Unspecialized",
            self.types.AGENT_ROOT: "Agent Root",
            self.types.AGENT_LEAF: "Agent Leaf",
            self.types.AGENT_FLOWER: "Agent Flower",
        }


class EnvTypeDef:
    class types:
        VOID = 0
        AIR = 1
        EARTH = 2
        IMMOVABLE = 3
        SUN = 4
        OUT_OF_BOUNDS = 5

    def __init__(self):
        self.type_names = {
            self.types.VOID: "Void",
            self.types.AIR: "Air",
            self.types.EARTH: "Earth",
            self.types.IMMOVABLE: "Immovable",
            self.types.SUN: "Sun",
            self.types.OUT_OF_BOUNDS: "Out of Bounds",
        }


env_type_def = EnvTypeDef()
agent_type_def = AgentTypeDef()


def count_cell_types(env, type_def=env_type_def):
    flattened_types = env.type_grid.flatten()
    type_counts = jp.bincount(flattened_types, length=len(type_def.type_names))
    readable_counts = Counter(
        {
            type_def.type_names.get(i, f"Unknown Type {i}"): int(count)
            for i, count in enumerate(type_counts)
            if count > 0
        }
    )
    return readable_counts


def count_agent_types(env, agent_type_def=agent_type_def):
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
    return readable_counts
