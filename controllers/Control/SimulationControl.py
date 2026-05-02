# =============================================================================
# SIMULATION CONTROL MODULE
# =============================================================================
# Basic simulation control for robot positioning and randomization
# =============================================================================

from math import sqrt
import sys, os
from dotenv import load_dotenv
import random
from webots_path import configure_webots_python_path

load_dotenv()
configure_webots_python_path()
from controller import Supervisor

TIME_STEP = 16

supervisor = Supervisor()

robot_node = supervisor.getFromDef("SUPER")
trans_field = robot_node.getField("translation")
rot_field = robot_node.getField("rotation")

rndx = random.uniform(-5, 5)
rndy = random.uniform(-5, 5)
POS = [rndx, rndy, 0]
trans_field.setSFVec3f(POS)
rot = random.uniform(0, 6.28319)
angle = [0, 0, 1, rot]
rot_field.setSFRotation(angle)
