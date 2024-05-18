import random

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

# Overriding the default environment logic with a custom one
from utils.environment_utils import EnvironmentHistory

env_logic.process_energy = env_override.process_energy

# Overriding the default step maker with a custom one
import self_organising_systems.biomakerca.step_maker as step_maker

import overrides.step_maker_override as step_maker_override

step_maker.step_env = step_maker_override.step_env

import concurrent.futures

from configs.seasons_config import SeasonsConfig
from utils.biomaker_util_no_video import perform_simulation, start_simulation


def pad_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    orgin = (5, 15)
    fontScale = 0.5
    color = (0, 0, 0)
    thickness = 1

    new_h = img.shape[0] // 15
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    img = np.concatenate([np.ones([new_h, img.shape[1], img.shape[2]]), img], 0)
    img = cv2.putText(img, text, orgin, font, fontScale, color, thickness, cv2.LINE_AA)
    return img


def make_configs(base_config: SeasonsConfig):
    run_id = random.randint(0, 99999999)
    config_dict = dict(
        (name, getattr(base_config, name))
        for name in dir(base_config)
        if not name.startswith("__")
    )
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
):
    environmentHistory = EnvironmentHistory(
        base_config, days_since_start, folder, sim, True
    )

    frame = start_simulation(env, base_config, env_config)
    step = 0
    for year in range(base_config.years):
        for month_params in base_config.month_params.items():
            step, env, programs, env_history = perform_simulation(
                env,
                programs,
                base_config,
                month_params[1],
                env_config,
                agent_logic,
                mutator,
                key,
                None,
                frame,
                step=step,
                season=f"{month_params[0]} {year + 1}",
            )
            environmentHistory.add_all(
                env_history, month_params[1]["Season"], month_params[0], year
            )

    return programs, env, environmentHistory


def run_single_simulation(type_of_january, sim, twenty_year_burn_in):
    if type_of_january == "warm":
        expirement_configuration = SeasonsConfig(
            "warm_winter_month", 1, 365, january_air_diffusion_rate=0.07, sim=sim
        )
    else:
        expirement_configuration = SeasonsConfig(
            "regular_winter_month", 1, 365, sim=sim
        )

    env, base_config, env_config, agent_logic, mutator, key, programs = make_configs(
        twenty_year_burn_in
    )

    programs, burn_in_env, environmentHistory = run_seasons(
        env,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        programs,
        days_since_start=0,
        folder="twenty_year_burn_in",
        sim=sim,
    )
    new_days_since_start = 20 * 365

    env, base_config, env_config, agent_logic, mutator, key, programs = make_configs(
        expirement_configuration
    )
    folder = "warm_winter_month" if type_of_january == "warm" else "basic_seasons"
    programs, env, environmentHistory = run_seasons(
        burn_in_env,
        base_config,
        env_config,
        agent_logic,
        mutator,
        key,
        programs,
        days_since_start=new_days_since_start,
        folder=folder,
        sim=sim,
    )

    environmentHistory.save_results(folder, sim)


def main():

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for sim in range(2):
            twenty_year_runup = SeasonsConfig("twenty_year_burn-in", 1, 365, sim=sim)
            for type_of_january in ["warm", "cold"]:

                futures.append(
                    executor.submit(run_single_simulation, type_of_january, sim)
                )

        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
