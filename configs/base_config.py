import jax.random as jr


class BaselineConfig:
    out_file = "video_seasons.mp4"
    ec_id = "persistence"       # @param ['persistence', 'pestilence', 'collaboration', 'sideways']
    env_width_type = "landscape"   # @param ['wide', 'landscape', 'square', 'petri']
    
    # Set soil_unbalance_limit to 0 to reproduce the original environment. Set it to 1/3 for having self-balancing environments (recommended).
    soil_unbalance_limit = 1/3  # @param [0, "1/3"] {type:"raw"}

    agent_model = "minimal"  # @param ['minimal', 'extended']
    mutator_type = "basic"  # @param ['basic', 'randomly_adaptive']
    key = jr.PRNGKey(43)

    # How many unique programs (organisms) are allowed in the simulation.
    n_max_programs = 5

    # if True, every 50 steps we check whether the agents go extinct. If they did,
    # we replace a seed in the environment.
    replace_if_extinct = False

    # The number of frames of the video. This is NOT the number of steps.
    # The total number of steps depend on the number of steps per frame, which can
    # vary over time.
    # In the article, we generally use 500 or 750 frames.
    n_frames = 125

    # on what FRAME to double speed.
    when_to_double_speed = []
    # on what FRAME to reset speed.
    when_to_reset_speed = []
    fps = 20
    # zoom_sz affects the size of the image. If this number is not even, the resulting
    # video *may* not be supported by all renderers.
    zoom_sz = 4

    # how many steps per frame we start with. This gets usually doubled many times
    # during the simulation.
    # In the article, we usually use 2 or 4 as the starting value, sometimes 1.
    steps_per_frame = 2

    ### Evaluation ###
    what_to_evaluate = "extracted"  # @param ["initialization", "extracted"]
    n_eval_steps = 100
    n_eval_reps = 1
    eval_key = jr.PRNGKey(123)
    # From 0 until 0.15, default = 0.1
    AIR_DIFFUSION_RATE = 0.1
    SOIL_DIFFUSION_RATE = 0.1
