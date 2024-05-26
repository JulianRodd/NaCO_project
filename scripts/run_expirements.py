import os
import pickle

import cv2
import jax.random as jr
import numpy as np
import self_organising_systems.biomakerca.env_logic as env_logic
from jax import vmap
from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.agent_logic import BasicAgentLogic
from self_organising_systems.biomakerca.mutators import (
    BasicMutator,
    RandomlyAdaptiveMutator,
)

import overrides.env_logic_override as env_override
from utils.constants import logger
from utils.environment_utils import EnvironmentHistory

env_logic.process_energy = env_override.process_energy

import self_organising_systems.biomakerca.step_maker as step_maker

import overrides.step_maker_override as step_maker_override

step_maker.step_env = step_maker_override.step_env


from configs.seasons_config import SeasonsConfig
from utils.biomaker_util_no_video import perform_simulation

USE_WANDB = False
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_ORIGIN = (5, 15)
TEXT_FONT_SCALE = 0.5
TEXT_COLOR = (0, 0, 0)
TEXT_THICKNESS = 1
# Constants
NUM_SIMS = 25
MAX_WORKERS = 1
TWENTY_YEAR_DAYS = 20 * 365
SEASON_TYPES = ["warm", "cold"]
BURN_IN_FOLDER = "twenty_year_burn_in"
BURN_IN_CONFIG_NAME = "twenty_year_burn_in"
BURN_IN_YEARS = 20
BURN_IN_DAYS_PER_YEAR = 365
DAYS_PER_YEAR = 365
SIMULATION_YEARS = 5
PICKLE_DIR = "pickles"
EARLY_EXTINCTION_MONTH_COUNT = 6


def pad_text(img, text):
    new_height = img.shape[0] // 15
    new_height = new_height if new_height % 2 == 0 else new_height + 1
    img = np.concatenate([np.ones([new_height, img.shape[1], img.shape[2]]), img], 0)
    img = cv2.putText(
        img,
        text,
        TEXT_ORIGIN,
        TEXT_FONT,
        TEXT_FONT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    return img


def make_configs(base_config: SeasonsConfig):
    config_dict = {
        name: getattr(base_config, name)
        for name in dir(base_config)
        if not name.startswith("__")
    }
    for key, value in config_dict.items():
        if "key" in key or isinstance(value, np.ndarray):
            config_dict[key] = [int(i) if i > 0 else float(i) for i in value]

    env, env_config = evm.get_env_and_config(
        base_config.ec_id,
        width_type=base_config.env_width_type,
        h=base_config.frame_height,
    )
    env_config.soil_unbalance_limit = base_config.soil_unbalance_limit
    env_config.nutrient_cap = base_config.nutrient_cap
    env_config.specialize_cost = base_config.specialize_cost

    agent_logic = BasicAgentLogic(
        env_config, minimal_net=base_config.agent_model == "minimal"
    )
    env_config.max_lifetime = base_config.max_lifetime

    standard_deviation = (
        1e-2
        if base_config.mutator_type == "basic" and base_config.agent_model == "basic"
        else 1e-3
    )
    mutator = (
        BasicMutator(sd=standard_deviation, change_perc=0.2)
        if base_config.mutator_type == "basic"
        else RandomlyAdaptiveMutator(init_sd=standard_deviation, change_perc=0.2)
    )

    print("\n\nCurrent config:")
    print("\n".join(f"{key}: {value}" for key, value in vars(env_config).items()))

    ku, key = jr.split(base_config.key)
    programs = vmap(agent_logic.initialize)(jr.split(ku, base_config.n_max_programs))
    programs = vmap(mutator.initialize)(jr.split(ku, programs.shape[0]), programs)

    return env, base_config, env_config, agent_logic, mutator, key, programs


def run_seasons(
    env,
    base_config,
    env_config,
    agent_logic,
    mutator,
    key,
    programs,
    days_since_start=0,
    folder="",
    sim=0,
    fail_on_extinction=False,
):
    environment_history = EnvironmentHistory(
        base_config, days_since_start, folder, sim, USE_WANDB
    )

    step = 0
    for year in range(base_config.years):
        extinction_counter = 0
        for month_name, month_params in base_config.month_params.items():
            step, env, programs, env_history = perform_simulation(
                env,
                programs,
                base_config,
                month_params,
                env_config,
                agent_logic,
                mutator,
                key,
                None,
                None,
                step=step,
                season=f"{month_name} {year + 1}",
            )
            environment_history.add_all(
                env_history, month_params["Season"], month_name, year
            )
            agent_count = environment_history.return_agent_count_of_last_env()
            if agent_count == 0:
                extinction_counter += 1

            if agent_count > 0:
                extinction_counter = 0

            if extinction_counter >= EARLY_EXTINCTION_MONTH_COUNT:

                if fail_on_extinction:
                    raise ValueError(
                        f"Early extinction detected in simulation {sim} at year {year} and month {month_name}"
                    )
                else:
                    logger.info(
                        f"Early extinction detected in simulation {sim} at year {year} and month {month_name}"
                    )

    return programs, env, environment_history


def run_single_simulation(type_of_january, sim, twenty_year_burn_in_env):
    if type_of_january == "warm":
        experiment_config = SeasonsConfig(
            "warm_winter_month",
            SIMULATION_YEARS,
            DAYS_PER_YEAR,
            january_air_diffusion_rate=0.07,
            simulation=sim,
        )
    else:
        experiment_config = SeasonsConfig(
            "regular_winter_month", SIMULATION_YEARS, DAYS_PER_YEAR, simulation=sim
        )

    env, base_config, env_config, agent_logic, mutator, key, programs = make_configs(
        experiment_config
    )
    folder = "warm_winter_month" if type_of_january == "warm" else "basic_seasons"
    programs, env, environment_history = run_seasons(
        twenty_year_burn_in_env,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        programs,
        days_since_start=TWENTY_YEAR_DAYS,
        folder=folder,
        sim=sim,
    )

    environment_history.save_results(folder, sim)


def run_experiments(sim, burn_in_env):
    for type_of_january in SEASON_TYPES:
        run_single_simulation(type_of_january, sim, burn_in_env)


def run_burn_in_simulation(sim):
    pickle_path = os.path.join(PICKLE_DIR, f"burn_in_env_sim_{sim}.pkl")

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            burn_in_env = pickle.load(f)
        logger.info(
            f"Loaded burn-in environment from {pickle_path} for simulation {sim}"
        )
        return burn_in_env

    seed_simulation = sim
    while True:
        try:
            # Configuration for the twenty-year burn-in phase
            twenty_year_burn_in = SeasonsConfig(
                BURN_IN_CONFIG_NAME,
                BURN_IN_YEARS,
                BURN_IN_DAYS_PER_YEAR,
                simulation=seed_simulation,
            )
            env, base_config, env_config, agent_logic, mutator, key, programs = (
                make_configs(twenty_year_burn_in)
            )

            programs, burn_in_env, environment_history = run_seasons(
                env,
                base_config,
                env_config,
                agent_logic,
                mutator,
                key,
                programs,
                days_since_start=0,
                folder=BURN_IN_FOLDER,
                sim=sim,
                fail_on_extinction=True,
            )

            # Save the burn-in environment to a pickle file
            with open(pickle_path, "wb") as f:
                pickle.dump(burn_in_env, f)
            logger.info(
                f"Saved burn-in environment to {pickle_path} for simulation {sim}"
            )

            return burn_in_env

        except ValueError as e:
            logger.warning(f"Early extinction detected: {e}. Retrying with new seed.")
            seed_simulation += 100


def main():
    os.makedirs(PICKLE_DIR, exist_ok=True)

    burn_in_environments = [None] * NUM_SIMS

    # Run burn-in simulations
    for sim in range(NUM_SIMS):
        burn_in_environments[sim] = run_burn_in_simulation(sim)

    # Run experiments for each simulation
    for sim in range(NUM_SIMS):
        run_experiments(sim, burn_in_environments[sim])


if __name__ == "__main__":
    main()
