import jax.random as jr
import numpy as np


class SeasonsConfig:
    def __init__(self, name, years, days_in_year, january_air_diffusion_rate=0.03, slight_alteration = False, simulation = 0):
        self.name = name
        self.years = years
        self.days_in_year = days_in_year
        self.january_air_diffusion_rate = january_air_diffusion_rate
        self.simulation = simulation
        self.key = jr.PRNGKey(simulation)
        self.month_params = {
            "January": {
                "Season": "Winter",
                "AIR_DIFFUSION_RATE": self.january_air_diffusion_rate,
                "SOIL_DIFFUSION_RATE": 0.03,
            },
            "February": {
                "Season": "Winter",
                "AIR_DIFFUSION_RATE": 0.04,
                "SOIL_DIFFUSION_RATE": 0.04,
            },
            "March": {
                "Season": "Spring",
                "AIR_DIFFUSION_RATE": 0.07,
                "SOIL_DIFFUSION_RATE": 0.07,
            },
            "April": {
                "Season": "Spring",
                "AIR_DIFFUSION_RATE": 0.075,
                "SOIL_DIFFUSION_RATE": 0.075,
            },
            "May": {
                "Season": "Spring",
                "AIR_DIFFUSION_RATE": 0.08,
                "SOIL_DIFFUSION_RATE": 0.08,
            },
            "June": {
                "Season": "Summer",
                "AIR_DIFFUSION_RATE": 0.1,
                "SOIL_DIFFUSION_RATE": 0.1,
            },
            "July": {
                "Season": "Summer",
                "AIR_DIFFUSION_RATE": 0.11,
                "SOIL_DIFFUSION_RATE": 0.11,
            },
            "August": {
                "Season": "Summer",
                "AIR_DIFFUSION_RATE": 0.1,
                "SOIL_DIFFUSION_RATE": 0.1,
            },
            "September": {
                "Season": "Autumn",
                "AIR_DIFFUSION_RATE": 0.09,
                "SOIL_DIFFUSION_RATE": 0.09,
            },
            "October": {
                "Season": "Autumn",
                "AIR_DIFFUSION_RATE": 0.07,
                "SOIL_DIFFUSION_RATE": 0.07,
            },
            "November": {
                "Season": "Autumn",
                "AIR_DIFFUSION_RATE": 0.05,
                "SOIL_DIFFUSION_RATE": 0.05,
            },
            "December": {
                "Season": "Winter",
                "AIR_DIFFUSION_RATE": 0.035,
                "SOIL_DIFFUSION_RATE": 0.035,
            },
        }
        if slight_alteration:
            # each months air and soil diffusion rate is slightly altered by multiplying by a random number between 0.9 and 1.1
            # normal distribution with mean 1 and std 0.05
            for month in self.month_params:
                self.month_params[month]["AIR_DIFFUSION_RATE"] *= np.random.normal(1, 0.05)
                self.month_params[month]["SOIL_DIFFUSION_RATE"] *= np.random.normal(1, 0.05)
        self.n_frames = int(
            self.days_in_year
            / len(self.month_params)
        )

    out_file = "output/seasons"
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


