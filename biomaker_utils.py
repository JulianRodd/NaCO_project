import time
from functools import partial

import cv2
import jax
import jax.numpy as jp
import jax.random as jr
import numpy as np
import tqdm
from jax import jit, vmap
from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.agent_logic import BasicAgentLogic
from self_organising_systems.biomakerca.display_utils import zoom
from self_organising_systems.biomakerca.env_logic import (
    ReproduceOp,
    env_perform_one_reproduce_op,
)
from self_organising_systems.biomakerca.mutators import (
    BasicMutator,
    RandomlyAdaptiveMutator,
)
from self_organising_systems.biomakerca.step_maker import step_env

from configs.base_config import BaselineConfig


def pad_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    orgin = (5, 15)
    fontScale = 0.5
    color = (0, 0, 0)
    thickness = 1

    # ensure to preserve even size (assumes the input size was even.
    new_h = img.shape[0] // 15
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    img = np.concatenate([np.ones([new_h, img.shape[1], img.shape[2]]), img], 0)
    img = cv2.putText(img, text, orgin, font, fontScale, color, thickness, cv2.LINE_AA)
    return img


def make_frame(env, step, speed, env_config, zoom_sz, season):
    if season == "":
      padText = "Step {:<7} Speed: {}x".format(step, speed)
    else: 
      padText = "Step {:<7} Speed: {}x   Season: {}".format(step, speed, season)
    return pad_text(
        zoom(evm.grab_image_from_env(env, env_config), zoom_sz),
        padText,
    )


def count_agents_f(env, etd):
    return etd.is_agent_fn(env.type_grid).sum()


def start_simulation(env, base_config, env_config):
    step = 0
    frame = pad_text(
        zoom(evm.grab_image_from_env(env, env_config), base_config.zoom_sz),
        "Step {:<7} Speed: {}x".format(step, base_config.steps_per_frame),
    )
    return frame


def perform_simulation(
    env, programs, base_config, env_config, agent_logic, mutator, key, video, frame, step=0, season=""
):
    video.add_image(frame)
    for i in tqdm.trange(base_config.n_frames):
        if i in base_config.when_to_double_speed:
            base_config.steps_per_frame *= 2
        if i in base_config.when_to_reset_speed:
            base_config.steps_per_frame = 1
        for j in range(base_config.steps_per_frame):
            step += 1
            key, ku = jr.split(key)
            env, programs = step_env(
                ku,
                env,
                env_config,
                agent_logic,
                programs,
                do_reproduction=True,
                mutate_programs=True,
                mutator=mutator,
                soil_diffusion_rate=base_config.SOIL_DIFFUSION_RATE,
                air_diffusion_rate=base_config.AIR_DIFFUSION_RATE,
            )
            if base_config.replace_if_extinct and step % 50 == 0:
                # check if there is no alive cell.
                any_alive = jit(lambda type_grid: evm.is_agent_fn(type_grid).sum() > 0)(
                    env.type_grid
                )
                if not any_alive:
                    # Then place a new seed.
                    agent_init_nutrients = (
                        env_config.dissipation_per_step * 4 + env_config.specialize_cost
                    )
                    ku, key = jr.split(key)
                    rpos = jp.stack(
                        [
                            0,
                            jr.randint(ku, (), minval=0, maxval=env.type_grid.shape[1]),
                        ],
                        0,
                    )
                    ku, key = jr.split(key)
                    raid = jr.randint(
                        ku, (), minval=0, maxval=base_config.n_max_programs
                    ).astype(jp.uint32)
                    repr_op = ReproduceOp(1.0, rpos, agent_init_nutrients * 2, raid)
                    ku, key = jr.split(key)
                    env = jit(partial(env_perform_one_reproduce_op, config=env_config))(
                        ku, env, repr_op
                    )

                    # show it, though
                    frame = make_frame(env, step, base_config.steps_per_frame, season)
                    for stop_i in range(10):
                        video.add_image(frame)

        video.add_image(
            make_frame(
                env,
                step,
                base_config.steps_per_frame,
                env_config,
                base_config.zoom_sz,
                season
            )
        )
    return step


@partial(
    jit,
    static_argnames=["config", "agent_logic", "mutator", "n_steps", "n_max_programs"],
)


def count_flowers(agent_logic, env): # added
    flower_id = agent_logic.config.etd.specialization_idxs.AGENT_FLOWER
    total_flowers = (env.agent_id_grid == flower_id).sum()

    agentTypes = env.agent_id_grid

    # Count of each type of agent
    zeros = np.count_nonzero(agentTypes == 0)
    ones = np.count_nonzero(agentTypes == 1)
    twos = np.count_nonzero(agentTypes == 2)
    threes = np.count_nonzero(agentTypes == 3)
    four = np.count_nonzero(agentTypes == 4)
    fives = np.count_nonzero(agentTypes == 5)

    print("Count of all 0 in grid: ", zeros)
    print("Count of all 1 in grid: ", ones)
    print("Count of all 2 in grid: ", twos)
    print("Count of all 3 in grid: ", threes)
    print("Count of all 4 in grid: ", four)
    print("Count of all 5 in grid: ", fives)

    return total_flowers

def evaluate_biome(
    key,
    st_env,
    config,
    agent_logic,
    mutator,
    n_steps,
    init_program=None,
    n_max_programs=128,
):
    def body_f(i, carry):
        key, env, programs, tot_agents_n = carry
        ku, key = jr.split(key)

        env, programs = step_env(
            ku,
            env,
            config,
            agent_logic,
            programs,
            do_reproduction=True,
            mutate_programs=True,
            mutator=mutator,
        )

        tot_agents_n += count_agents_f(env, config.etd)
        return key, env, programs, tot_agents_n

    if init_program is None:
        ku, key = jr.split(key)
        programs = vmap(agent_logic.initialize)(jr.split(ku, n_max_programs))
        ku, key = jr.split(key)
        programs = vmap(mutator.initialize)(jr.split(ku, programs.shape[0]), programs)
    else:
        programs = jp.repeat(init_program[None, :], n_max_programs, axis=0)

    key, env, programs, tot_agents_n = jax.lax.fori_loop(
        0, n_steps, body_f, (key, st_env, programs, 0)
    )

    # check whether they got extinct:
    is_extinct = (count_agents_f(env, config.etd) == 0).astype(jp.int32)
    return tot_agents_n, is_extinct


def perform_evaluation(
    env, programs, st_env, env_config, agent_logic, mutator, base_config
):
    # Evaluate the configuration
    # With this code, you can either evaluate a randomly initialized model, or models extracted from the previous run.
    # In the latter case, make sure that there are agents alive at the end of the simulation.

    if base_config.what_to_evaluate == "initialization":
        init_programs = None
    else:
        # Extract a living program from the final environment.
        aid_flat = env.agent_id_grid.flatten()
        is_agent_flat = (
            env_config.etd.is_agent_fn(env.type_grid).flatten().astype(jp.float32)
        )
        n_alive_per_id = jax.ops.segment_sum(
            is_agent_flat, aid_flat, num_segments=base_config.n_max_programs
        )
        alive_programs = programs[n_alive_per_id > 0]
        print("Extracted {} programs.".format(alive_programs.shape[0]))
        assert (
            alive_programs.shape[0] >= base_config.n_eval_reps
        ), "Not enough alive programs found."

        init_programs = alive_programs[: base_config.n_eval_reps]

    t_st = time.time()
    key, ku = jr.split(base_config.eval_key)
    b_tot_agents_n, b_is_extinct = jit(
        vmap(
            partial(
                evaluate_biome,
                st_env=st_env,
                config=env_config,
                agent_logic=agent_logic,
                mutator=mutator,
                n_steps=base_config.n_eval_steps,
            )
        )
    )(jr.split(ku, base_config.n_eval_reps), init_program=init_programs)
    print("Took", time.time() - t_st, "seconds")
    print(
        "Total number of agents",
        b_tot_agents_n,
        b_tot_agents_n.mean(),
        b_tot_agents_n.std(),
    )
    print("Extinction events", b_is_extinct, b_is_extinct.mean(), b_is_extinct.std())


    count_flowers(agent_logic, env)
    #print("Number of flowers: ", count_flowers(agent_logic, env))# added

    #logic = BasicAgentLogic(BasicAgentLogic)  # added
    # print("Total flowers produced:", agent_logic.flowers_produced)# added
    # print("Total seeds dispersed:", agent_logic.seeds_dispersed)# added


def main():
    base_config = BaselineConfig()

    st_env, env_config = evm.get_env_and_config(
        base_config.ec_id, width_type=base_config.env_width_type
    )
    env_config.soil_unbalance_limit = base_config.soil_unbalance_limit

    agent_logic = BasicAgentLogic(
        env_config, minimal_net=base_config.agent_model == "minimal"
    )

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

    print("Current config:")
    print("\n".join("%s: %s" % item for item in vars(env_config).items()))

    ku, key = jr.split(base_config.key)
    programs = vmap(agent_logic.initialize)(jr.split(ku, base_config.n_max_programs))
    programs = vmap(mutator.initialize)(jr.split(ku, programs.shape[0]), programs)

    env = st_env

    perform_simulation(
        env, programs, base_config, env_config, agent_logic, mutator, key
    )

    perform_evaluation(
        env, programs, st_env, env_config, agent_logic, mutator, base_config
    )


if __name__ == "__main__":
    main()


## Examples for modifying the config

## Regardless, to trigger the recomputation of step_env and similar,
## config needs to be a new object! So, first, we create a new copy.

# import copy
# env_config = copy.copy(env_config)

## Change simple isolated parameters (most of them)
# env_config.struct_integrity_cap = 100
# env_config.max_lifetime = 500
## Vectors can be modified either by writing new vectors:
# env_config.dissipation_per_step = jp.array([0.02, 0.02])
## Or by multiplying previous values. Note that they are immutable!
# env_config.dissipation_per_step = env_config.dissipation_per_step * 2

## agent_state_size is trickier, because it influences env_state_size.
## So you can either create a new config:
## Note that you would have to insert all values that you don't want to take
## default initializations.
# env_config = evm.EnvConfig(agent_state_size=4)
## Or you can just modify env_state_size as well.
## (env_state_size = agent_state_size + 4) for now.
# env_config.agent_state_size = 4
# env_config.env_state_size = env_config.agent_state_size + 4
