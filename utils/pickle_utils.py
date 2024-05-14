import pickle

from utils.constants import logger


def dump_environment(env, base_config):
    if env is None:
        return
    file_name = (
        f"pickles/environment_{base_config.years}_{base_config.days_in_year}.pkl"
    )
    with open(file_name, "wb") as handle:
        pickle.dump(env, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_environment(years, days_in_year):
    file_name = f"pickles/environment_{years}_{days_in_year}.pkl"
    try:
        with open(file_name, "rb") as handle:
            env = pickle.load(handle)
            return env
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error loading environment: {e}")
        return None
