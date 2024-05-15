import random
import json
import cv2
from datetime import datetime

import jax.random as jr
import mediapy as media
import numpy as np
import self_organising_systems.biomakerca.env_logic as env_logic
from jax import vmap
from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.agent_logic import BasicAgentLogic
from self_organising_systems.biomakerca.mutators import (
    BasicMutator,
    RandomlyAdaptiveMutator,
)
from tqdm import tqdm

# Overriding the default environment logic with a custom one
import overrides.env_logic_override as env_override

env_logic.process_energy = env_override.process_energy

# Overriding the default step maker with a custom one
import self_organising_systems.biomakerca.step_maker as step_maker
import overrides.step_maker_override as step_maker_override

step_maker.step_env = step_maker_override.step_env

from biomaker_utils import perform_evaluation, perform_simulation, start_simulation
from configs.seasons_config import SeasonsConfig


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
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    run_id = random.randint(0, 99999999)
    config_dict = dict(
        (name, getattr(base_config, name))
        for name in dir(base_config)
        if not name.startswith("__")
    )
    for key, value in config_dict.items():
        if "key" in key or isinstance(value, np.ndarray):
            config_dict[key] = [int(i) if i > 0 else float(i) for i in value]

    with open(f"{base_config.out_file}_config_{current_time}_{run_id}.json", "w") as file:
        json.dump(config_dict, file, indent=4)
    base_config.out_file = f"{base_config.out_file}_video_{current_time}_{run_id}.mp4"

    env, env_config = evm.get_env_and_config(
        base_config.ec_id, width_type=base_config.env_width_type, h=base_config.frame_height
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


def run_seasons(env, base_config, env_config, agent_logic, mutator, key, programs):
    frame = start_simulation(env, base_config, env_config)
    with media.VideoWriter(
        base_config.out_file, shape=frame.shape[:2], fps=base_config.fps, crf=18
    ) as video:
        step = 0
        for year in range(base_config.years):
            for season_name, season_periods in base_config.seasons.items():
                for time_period_info in tqdm(season_periods):
                    step, env, programs = perform_simulation(
                        env,
                        programs,
                        base_config,
                        time_period_info,
                        env_config,
                        agent_logic,
                        mutator,
                        key,
                        video,
                        frame,
                        step=step,
                        season=f"{season_name} {year + 1}",
                    )

    return programs, env


def count_agents(env):
    agent_types = env.state_grid

    # Count of each type of agent
    zeros = np.count_nonzero(agent_types == 0)  # unspecialized
    ones = np.count_nonzero(agent_types == 1)  # root
    twos = np.count_nonzero(agent_types == 2)  # leaf
    threes = np.count_nonzero(agent_types == 3)  # flower

    print("Count of unspecialized in grid: ", zeros)
    print("Count of roots in grid: ", ones)
    print("Count of leafs in grid: ", twos)
    print("Count of flowers in grid: ", threes)


def main():
    base_config = SeasonsConfig()

    print(base_config.n_frames)

    env, base_config, env_config, agent_logic, mutator, key, programs = make_configs(
        base_config
    )

    programs, env = run_seasons(
        env, base_config, env_config, agent_logic, mutator, key, programs
    )

    perform_evaluation(
        env, programs, env, env_config, agent_logic, mutator, base_config
    )


if __name__ == "__main__":
    main()


# def make_frame(env, step, speed, env_config, zoom_sz):
#     return pad_text(
#         zoom(evm.grab_image_from_env(env, env_config), zoom_sz),
#         "Step {:<7} Speed: {}x".format(step, speed),
#     )

# def count_agents_f(env, etd):
#     return etd.is_agent_fn(env.type_grid).sum()


# @partial(
#     jit,
#     static_argnames=["config", "agent_logic", "mutator", "n_steps", "n_max_programs"],
# )
# def evaluate_biome(
#     key,
#     st_env,
#     config,
#     agent_logic,
#     mutator,
#     n_steps,
#     init_program=None,
#     n_max_programs=128,
# ):
#     def body_f(i, carry):
#         key, env, programs, tot_agents_n = carry
#         ku, key = jr.split(key)

#         env, programs = step_env(
#             ku,
#             env,
#             config,
#             agent_logic,
#             programs,
#             do_reproduction=True,
#             mutate_programs=True,
#             mutator=mutator,
#         )

#         tot_agents_n += count_agents_f(env, config.etd)
#         return key, env, programs, tot_agents_n

#     if init_program is None:
#         ku, key = jr.split(key)
#         programs = vmap(agent_logic.initialize)(jr.split(ku, n_max_programs))
#         ku, key = jr.split(key)
#         programs = vmap(mutator.initialize)(jr.split(ku, programs.shape[0]), programs)
#     else:
#         programs = jp.repeat(init_program[None, :], n_max_programs, axis=0)

#     key, env, programs, tot_agents_n = jax.lax.fori_loop(
#         0, n_steps, body_f, (key, st_env, programs, 0)
#     )

#     # check whether they got extinct:
#     is_extinct = (count_agents_f(env, config.etd) == 0).astype(jp.int32)
#     return tot_agents_n, is_extinct


# def evaluation(env, programs, st_env, env_config, agent_logic, mutator, base_config):
#     # Evaluate the configuration
#     # With this code, you can either evaluate a randomly initialized model, or models extracted from the previous run.
#     # In the latter case, make sure that there are agents alive at the end of the simulation.

#     if base_config.what_to_evaluate == "initialization":
#         init_programs = None
#     else:
#         # Extract a living program from the final environment.
#         aid_flat = env.agent_id_grid.flatten()
#         is_agent_flat = evm.is_agent_fn(env.type_grid).flatten().astype(jp.float32)
#         n_alive_per_id = jax.ops.segment_sum(
#             is_agent_flat, aid_flat, num_segments=base_config.n_max_programs
#         )
#         alive_programs = programs[n_alive_per_id > 0]
#         print("Extracted {} programs.".format(alive_programs.shape[0]))
#         assert alive_programs.shape[0] >= base_config.n_eval_reps, "Not enough alive programs found."

#         init_programs = alive_programs[:base_config.n_eval_reps]

#     t_st = time.time()
#     key, ku = jr.split(base_config.eval_key)
#     b_tot_agents_n, b_is_extinct = jit(
#         vmap(
#             partial(
#                 evaluate_biome,
#                 key=key,
#                 st_env=st_env,
#                 config=env_config,
#                 agent_logic=agent_logic,
#                 mutator=mutator,
#                 n_steps=base_config.n_eval_steps,
#             )
#         )
#     )(jr.split(ku, base_config.n_eval_reps), init_program=init_programs)
#     print("Took", time.time() - t_st, "seconds")
#     print(
#         "Total number of agents",
#         b_tot_agents_n,
#         b_tot_agents_n.mean(),
#         b_tot_agents_n.std(),
#     )
#     print("Extinction events", b_is_extinct, b_is_extinct.mean(), b_is_extinct.std())


# def main():
#     base_config = BaselineConfig()

#     st_env, env_config = evm.get_env_and_config(base_config.ec_id, width_type=base_config.env_width_type)
#     env_config.soil_unbalance_limit = base_config.soil_unbalance_limit

#     agent_logic = BasicAgentLogic(env_config, minimal_net=base_config.agent_model == "minimal")

#     sd = 1e-2 if base_config.mutator_type == "basic" and base_config.agent_model == "basic" else 1e-3
#     mutator = (
#         BasicMutator(sd=sd, change_perc=0.2)
#         if base_config.mutator_type == "basic"
#         else RandomlyAdaptiveMutator(init_sd=sd, change_perc=0.2)
#     )

#     print("Current config:")
#     print("\n".join("%s: %s" % item for item in vars(env_config).items()))

#     ku, key = jr.split(base_config.key)
#     programs = vmap(agent_logic.initialize)(jr.split(ku, base_config.n_max_programs))
#     # ku, key = jr.split(base_config.key)
#     programs = vmap(mutator.initialize)(jr.split(ku, programs.shape[0]), programs)

#     env = st_env

#     step = 0

#     # Perform a simulation
#     frame = make_frame(env, step, speed=base_config.steps_per_frame, env_config=env_config, zoom_sz=base_config.zoom_sz)

#     with media.VideoWriter(base_config.out_file, shape=frame.shape[:2], fps=base_config.fps, crf=18) as video:
#         video.add_image(frame)
#         for i in tqdm.trange(base_config.n_frames):
#             if i in base_config.when_to_double_speed:
#                 base_config.steps_per_frame *= 2
#             if i in base_config.when_to_reset_speed:
#                 base_config.steps_per_frame = 1
#             for j in range(base_config.steps_per_frame):
#                 step += 1
#                 key, ku = jr.split(key)
#                 env, programs = step_env(
#                     ku,
#                     env,
#                     env_config,
#                     agent_logic,
#                     programs,
#                     do_reproduction=True,
#                     mutate_programs=True,
#                     mutator=mutator,
#                 )
#                 if base_config.replace_if_extinct and step % 50 == 0:
#                     # check if there is no alive cell.
#                     any_alive = jit(lambda type_grid: evm.is_agent_fn(type_grid).sum() > 0)(
#                         env.type_grid
#                     )
#                     if not any_alive:
#                         # Then place a new seed.
#                         agent_init_nutrients = (
#                             env_config.dissipation_per_step * 4 + env_config.specialize_cost
#                         )
#                         ku, key = jr.split(key)
#                         rpos = jp.stack(
#                             [
#                                 0,
#                                 jr.randint(ku, (), minval=0, maxval=env.type_grid.shape[1]),
#                             ],
#                             0,
#                         )
#                         ku, key = jr.split(key)
#                         raid = jr.randint(ku, (), minval=0, maxval=base_config.n_max_programs).astype(
#                             jp.uint32
#                         )
#                         repr_op = ReproduceOp(1.0, rpos, agent_init_nutrients * 2, raid)
#                         ku, key = jr.split(key)
#                         env = jit(partial(env_perform_one_reproduce_op, config=env_config))(
#                             ku, env, repr_op
#                         )

#                         # show it, though
#                         frame = make_frame(env, step, base_config.steps_per_frame)
#                         for stop_i in range(10):
#                             video.add_image(frame)

#             video.add_image(make_frame(env, step, base_config.steps_per_frame, env_config, base_config.zoom_sz))

#     evaluation(env, programs, st_env, env_config, agent_logic, mutator, base_config)
