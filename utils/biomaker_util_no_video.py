from functools import partial

import jax.numpy as jp
import jax.random as jr
from jax import jit
from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.env_logic import (
    ReproduceOp,
    env_perform_one_reproduce_op,
)
from self_organising_systems.biomakerca.step_maker import step_env


def perform_simulation(
    env,
    programs,
    base_config,
    season_info,
    env_config,
    agent_logic,
    mutator,
    key,
    video,
    frame,
    step=0,
    season="",
):
    # video.add_image(frame)
    env_history = [env]
    for i in range(base_config.n_frames):
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
                soil_diffusion_rate=season_info["SOIL_DIFFUSION_RATE"],
                air_diffusion_rate=season_info["AIR_DIFFUSION_RATE"],
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
                    # frame = make_frame(env, step, base_config.steps_per_frame, season)
                    # for stop_i in range(10):
                    #     video.add_image(frame)
        env_history.append(env)

        # video.add_image(
        #     make_frame(
        #         env,
        #         step,
        #         base_config.steps_per_frame,
        #         env_config,
        #         base_config.zoom_sz,
        #         season
        #     )
        # )
    return step, env, programs, env_history
