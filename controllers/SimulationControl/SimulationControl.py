from math import sqrt
from controller import Supervisor
import random

TIME_STEP = 16

supervisor = Supervisor()

# get handle to robot's translation field
robot_node = supervisor.getFromDef("SUPER")
trans_field = robot_node.getField("translation")
rot_field = robot_node.getField("rotation")

#while supervisor.step(TIME_STEP) != -1:

    #may     compute travelled distance
    #need   values = trans_field.getSFVec3f()
    #later  dist = sqrt(values[0] * values[0] + values[2] * values[2])
            #print("a=%d, b=%d -> dist=%g" % (a, b, dist))

        # set the cubes position
rndx = random.uniform(-5, 5)
rndy = random.uniform(-5, 5)
POS = [rndx, rndy, 0]
trans_field.setSFVec3f(POS)
rot = random.uniform(0, 6.28319)
angle = [0, 0, 1, rot]
rot_field.setSFRotation(angle)
