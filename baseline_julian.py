# Overriding the default environment logic with a custom one
import self_organising_systems.biomakerca.env_logic as env_logic

import overrides.env_logic_override as env_override
from environment_utils import EnvironmentHistory, count_agent_types

env_logic.process_energy = env_override.process_energy


import self_organising_systems.biomakerca.step_maker as step_maker

# Overriding the default step maker with a custom one
import overrides.step_maker_override as step_maker_override

step_maker.step_env = step_maker_override.step_env


import jax.random as jr
import mediapy as media
import numpy as np
from IPython.display import Video
from jax import vmap
from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.agent_logic import BasicAgentLogic
from self_organising_systems.biomakerca.mutators import (
    BasicMutator,
    RandomlyAdaptiveMutator,
)

from biomaker_utils import perform_simulation, start_simulation
from configs.seasons_config import SeasonsConfig

base_config = SeasonsConfig()

st_env, env_config = evm.get_env_and_config(
    base_config.ec_id, width_type=base_config.env_width_type
)
env_config.soil_unbalance_limit = base_config.soil_unbalance_limit
env_config.nutrient_cap = np.asarray([20, 20])
env_config.specialize_cost = np.asarray([0.030, 0.030])

agent_logic = BasicAgentLogic(
    env_config, minimal_net=base_config.agent_model == "minimal"
)
env_config.max_lifetime = 2000

sd = (
    1e-2
    if base_config.mutator_type == "basic" and base_config.agent_model == "basic"
    else 1e-3
)
mutator = (
    BasicMutator(sd=sd, change_perc=0.2)
    if base_config.mutator_type == "basic"
    else RandomlyAdaptiveMutator(init_sd=sd, change_perc=0.2)
)

print("\n\nCurrent config:")
print("\n".join("%s: %s" % item for item in vars(env_config).items()))

ku, key = jr.split(base_config.key)
programs = vmap(agent_logic.initialize)(jr.split(ku, base_config.n_max_programs))
programs = vmap(mutator.initialize)(jr.split(ku, programs.shape[0]), programs)

env = st_env

file_version = "v3"

print(base_config.out_file)

APPEND_ALL_SEASONS_IN_ONE_VIDEO = True
environmentHistory = EnvironmentHistory()

if not APPEND_ALL_SEASONS_IN_ONE_VIDEO:
    base_config.out_file = f"spring{file_version}.mp4"

frame = start_simulation(env, base_config, env_config)
with media.VideoWriter(
    base_config.out_file, shape=frame.shape[:2], fps=base_config.fps, crf=18
) as video:

    # Spring 1
    base_config.AIR_DIFFUSION_RATE = 0.09
    base_config.SOIL_DIFFUSION_RATE = 0.09

    step, env, programs, env_history = perform_simulation(
        env,
        programs,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        video,
        frame,
        season="Spring 1",
    )
    environmentHistory.add_all(env_history, season="Spring 1")
    # Summer 1
    base_config.AIR_DIFFUSION_RATE = 0.1
    base_config.SOIL_DIFFUSION_RATE = 0.1

    step, env, programs, env_history = perform_simulation(
        env,
        programs,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        video,
        frame,
        step=step if APPEND_ALL_SEASONS_IN_ONE_VIDEO else 0,
        season="Summer 1",
    )
    environmentHistory.add_all(env_history, season="Summer 1")
    # Autumn 1
    base_config.AIR_DIFFUSION_RATE = 0.04
    base_config.SOIL_DIFFUSION_RATE = 0.04

    step, env, programs, env_history = perform_simulation(
        env,
        programs,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        video,
        frame,
        step=step if APPEND_ALL_SEASONS_IN_ONE_VIDEO else 0,
        season="Autumn 1",
    )

    environmentHistory.add_all(env_history, season="Autumn 1")
    # Winter 1
    base_config.AIR_DIFFUSION_RATE = 0.01
    base_config.SOIL_DIFFUSION_RATE = 0.01

    step, env, programs, env_history = perform_simulation(
        env,
        programs,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        video,
        frame,
        step=step if APPEND_ALL_SEASONS_IN_ONE_VIDEO else 0,
        season="Winter 1",
    )

    environmentHistory.add_all(env_history, season="Winter 1")
    # Spring 2
    base_config.AIR_DIFFUSION_RATE = 0.09
    base_config.SOIL_DIFFUSION_RATE = 0.09

    step, env, programs, env_history = perform_simulation(
        env,
        programs,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        video,
        frame,
        step=step if APPEND_ALL_SEASONS_IN_ONE_VIDEO else 0,
        season="Spring 2",
    )

    environmentHistory.add_all(env_history, season="Spring 2")
    # Summer 2
    base_config.AIR_DIFFUSION_RATE = 0.1
    base_config.SOIL_DIFFUSION_RATE = 0.1

    step, env, programs, env_history = perform_simulation(
        env,
        programs,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        video,
        frame,
        step=step if APPEND_ALL_SEASONS_IN_ONE_VIDEO else 0,
        season="Summer 2",
    )

    environmentHistory.add_all(env_history, season="Summer 2")
    # Autumn 2
    base_config.AIR_DIFFUSION_RATE = 0.04
    base_config.SOIL_DIFFUSION_RATE = 0.04

    step, env, programs, env_history = perform_simulation(
        env,
        programs,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        video,
        frame,
        step=step if APPEND_ALL_SEASONS_IN_ONE_VIDEO else 0,
        season="Autumn 2",
    )
    environmentHistory.add_all(env_history, season="Autumn 2")
    # Winter 2
    base_config.AIR_DIFFUSION_RATE = 0.01
    base_config.SOIL_DIFFUSION_RATE = 0.01

    step, env, programs, env_history = (
        perform_simulation(
            env,
            programs,
            base_config,
            env_config,
            agent_logic,
            mutator,
            key,
            video,
            frame,
            step=step if APPEND_ALL_SEASONS_IN_ONE_VIDEO else 0,
            season="Winter 2",
        )
    )
    environmentHistory.add_all(env_history, season="Winter 2")

environmentHistory.plot_agent_type_hist(filter_keys={"Root", "Leaf", "Flower"})

agent_type_counts = count_agent_types(env)

print("Agent Types:", agent_type_counts)

# if not APPEND_ALL_SEASONS_IN_ONE_VIDEO:
Video(base_config.out_file)
