import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("utils_logger")

SEASON_COLORS = {
    "Spring": "palegreen",
    "Summer": "lightcoral",
    "Autumn": "navajowhite",
    "Winter": "lightskyblue",
}

MATERIAL_COLORS = {
    "Void": "black",
    "Air": "lightblue",
    "Soil": "saddlebrown",
    "Immovable": "black",
    "Sun": "orange",
    "Out of Bounds": "red",
    "Unassigned": "grey",
    "Root": "chocolate",
    "Leaf": "lime",
    "Flower": "deeppink",
    "Plant Count": "red",
    "Air Nutrients in Air": "deepskyblue",
    "Soil Nutrients in Soil": "chocolate",
    "Soil Nutrients in Immovable": "saddlebrown",
    "Air Nutrients in Sun": "gold",
    "Air Nutrients in Roots": "darkorange",
    "Air Nutrients in Leafs": "lime",
    "Air Nutrients in Flowers": "deeppink",
    "Soil Nutrients in Roots": "saddlebrown",
    "Soil Nutrients in Leafs": "green",
    "Soil Nutrients in Flowers": "darkmagenta",
}

# grabbed these from environment.py
STR_IDX = 0
AGE_IDX = 1
EN_ST = 2
A_INT_STATE_ST = 4
EARTH_NUTRIENT_RPOS = 0
AIR_NUTRIENT_RPOS = 1


class AgentTypeDef:
    class types:
        VOID = 0
        AIR = 1
        EARTH = 2
        IMMOVABLE = 3
        SUN = 4
        OUT_OF_BOUNDS = 5
        AGENT_UNSPECIALIZED = 6
        AGENT_ROOT = 7
        AGENT_LEAF = 8
        AGENT_FLOWER = 9

    def __init__(self):
        self.type_names = {
            self.types.VOID: "Void",
            self.types.AIR: "Air",
            self.types.EARTH: "Soil",
            self.types.IMMOVABLE: "Immovable",
            self.types.SUN: "Sun",
            self.types.OUT_OF_BOUNDS: "Out of Bounds",
            self.types.AGENT_UNSPECIALIZED: "Unassigned",
            self.types.AGENT_ROOT: "Root",
            self.types.AGENT_LEAF: "Leaf",
            self.types.AGENT_FLOWER: "Flower",
        }


AGENT_TYPE_DEF = AgentTypeDef()
