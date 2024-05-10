# Overriding the default environment logic with a custom one
import overrides.env_logic_override as env_override
import self_organising_systems.biomakerca.env_logic as env_logic
env_logic.process_energy = env_override.process_energy
env_logic.balance_soil = env_override.balance_soil



# Overriding the default step maker with a custom one
import overrides.step_maker_override as step_maker_override
import self_organising_systems.biomakerca.step_maker as step_maker
step_maker.step_env = step_maker_override.step_env


import jax.random as jr
import mediapy as media
from IPython.display import Video
from jax import vmap
from self_organising_systems.biomakerca import environments as evm
from self_organising_systems.biomakerca.agent_logic import BasicAgentLogic
from self_organising_systems.biomakerca.mutators import (
    BasicMutator, RandomlyAdaptiveMutator)

from biomaker_utils import perform_evaluation, perform_simulation, start_simulation
from configs.seasons_config import SeasonsConfig as Config


config = Config()

st_env, env_config = evm.get_env_and_config(config.ec_id, width_type=config.env_width_type)
env_config.soil_unbalance_limit = config.soil_unbalance_limit
agent_logic = BasicAgentLogic(env_config, minimal_net=config.agent_model == "minimal")

sd = 1e-2 if config.mutator_type == "basic" and config.agent_model == "basic" else 1e-3
mutator = (
    BasicMutator(sd=sd, change_perc=0.2)
    if config.mutator_type == "basic"
    else RandomlyAdaptiveMutator(init_sd=sd, change_perc=0.2)
)

print("\n\nCurrent config:")
print("\n".join("%s: %s" % item for item in vars(env_config).items()))

ku, key = jr.split(config.key)
programs = vmap(agent_logic.initialize)(jr.split(ku, config.n_max_programs))
programs = vmap(mutator.initialize)(jr.split(ku, programs.shape[0]), programs)

env = st_env

APPEND_ALL_SEASONS_IN_ONE_VIDEO = False

# Spring
config.AIR_DIFFUSION_RATE = 0.13
config.SOIL_DIFFUSION_RATE = 0.13

if not APPEND_ALL_SEASONS_IN_ONE_VIDEO:
	config.out_file = "spring.mp4"

frame = start_simulation(env, config, env_config)
with media.VideoWriter(
	config.out_file, shape=frame.shape[:2], fps=config.fps, crf=18
) as video:
	step = perform_simulation(
		env, programs, config, env_config, agent_logic, mutator, key, video, frame, season="Spring"
	)

# if not APPEND_ALL_SEASONS_IN_ONE_VIDEO:
# Video(config.out_file)

# Winter
config.AIR_DIFFUSION_RATE = 0.1
config.SOIL_DIFFUSION_RATE = 0.01

if not APPEND_ALL_SEASONS_IN_ONE_VIDEO:
	config.out_file = "winter.mp4"
	frame = start_simulation(env, config, env_config)
with media.VideoWriter(
	config.out_file, shape=frame.shape[:2], fps=config.fps, crf=18
) as video:
	step = perform_simulation(
		env, programs, config, env_config, agent_logic, mutator, key, video, frame, step = step if APPEND_ALL_SEASONS_IN_ONE_VIDEO else 0, season="Winter"
	)

# Video(config.out_file)