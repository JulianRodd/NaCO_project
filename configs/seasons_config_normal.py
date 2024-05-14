import jax.random as jr
import numpy as np


class SeasonsConfig:
    name = "seasons_config"
    out_file = "output/normal_temprature"
    ec_id = "pestilence"  # @param ['persistence', 'pestilence', 'collaboration', 'sideways']
    env_width_type = "landscape"  # @param ['wide', 'landscape', 'square', 'petri']
    nutrient_cap = np.asarray([25, 25])
    specialize_cost = np.asarray([0.028, 0.028])
    max_lifetime = 3650  # 10 years

    frame_height = 150

    # Set soil_unbalance_limit to 0 to reproduce the original environment. Set it to 1/3 for having self-balancing environments (recommended).
    soil_unbalance_limit = 1 / 3  # @param [0, "1/3"] {type:"raw"}

    agent_model = "minimal"  # @param ['minimal', 'extended']
    mutator_type = "basic"  # @param ['basic', 'randomly_adaptive']
    key = jr.PRNGKey(43)

    # How many unique programs (organisms) are allowed in the simulation.
    n_max_programs = 25

    # if True, every 50 steps we check whether the agents go extinct. If they did,
    # we replace a seed in the environment.
    replace_if_extinct = False

    # on what FRAME to double speed.
    when_to_double_speed = []
    # on what FRAME to reset speed.
    when_to_reset_speed = []
    fps = 30
    # zoom_sz affects the size of the image. If this number is not even, the resulting
    # video *may* not be supported by all renderers.
    zoom_sz = 4

    # how many steps per frame we start with. This gets usually doubled many times
    # during the simulation.
    # In the article, we usually use 2 or 4 as the starting value, sometimes 1.
    steps_per_frame = 1

    ### Evaluation ###
    what_to_evaluate = "extracted"  # @param ["initialization", "extracted"]
    n_eval_steps = 100
    n_eval_reps = 1
    eval_key = jr.PRNGKey(123)

    seasons = {

        "Winter": [
            {  # December
                "AIR_DIFFUSION_RATE": 0.03,
                "SOIL_DIFFUSION_RATE": 0.03,
            },
            {  # January
                "AIR_DIFFUSION_RATE": 0.03,
                "SOIL_DIFFUSION_RATE": 0.03,
            },
            {  # Februari
                "AIR_DIFFUSION_RATE": 0.04,
                "SOIL_DIFFUSION_RATE": 0.04,
            },
        ],
                "Spring": [
            {  # March
                "AIR_DIFFUSION_RATE": 0.07,
                "SOIL_DIFFUSION_RATE": 0.07,
            },
            {  # April
                "AIR_DIFFUSION_RATE": 0.07,
                "SOIL_DIFFUSION_RATE": 0.07,
            },
            {  # May
                "AIR_DIFFUSION_RATE": 0.07,
                "SOIL_DIFFUSION_RATE": 0.07,
            },
        ],
        "Summer": [
            {  # June
                "AIR_DIFFUSION_RATE": 0.1,
                "SOIL_DIFFUSION_RATE": 0.1,
            },
            {  # July
                "AIR_DIFFUSION_RATE": 0.1,
                "SOIL_DIFFUSION_RATE": 0.1,
            },
            {  # August
                "AIR_DIFFUSION_RATE": 0.1,
                "SOIL_DIFFUSION_RATE": 0.1,
            },
        ],
        "Autumn": [
            {  # September
                "AIR_DIFFUSION_RATE": 0.07,
                "SOIL_DIFFUSION_RATE": 0.07,
            },
            {  # October
                "AIR_DIFFUSION_RATE": 0.06,
                "SOIL_DIFFUSION_RATE": 0.06,
            },
            {  # November
                "AIR_DIFFUSION_RATE": 0.05,
                "SOIL_DIFFUSION_RATE": 0.05,
            },
        ],
    }

    years = 2

    # The number of frames of the video. This is NOT the number of steps.
    # The total number of steps depend on the number of steps per frame, which can
    # vary over time.
    # In the article, we generally use 500 or 750 frames.
    # for our NaCo project, we want each frame to represent a day, so we devide 365 by the number of time periods in a year
    days_in_year = 365
    n_frames = int(days_in_year / sum([len(periods) for periods in seasons.values()]))
