from controller import Supervisor

# This is a secondary controller to control the simulation
class SimControl:
    def __init__(self):
        # Create supervisor
        try:
            self.__world = Supervisor()
        except TypeError:
            raise Exception("World not loaded")

        # Setting the position to the middle
        try:
            self.__robot_node = self.__world.getFromDef("WEBOT")
        except TypeError:
            raise Exception("Robot node not found")

        # get the time step of the current world.
        self.__timestep = int(self.__world.getBasicTimeStep())

    # Returns timestamp
    def getTimestep(self):
       return self.__timestep

    # Sets the position of the robot
    def setRobotPosition(self, position):
        position_field = self.__robot_node.getField("translation")
        position_field.setSFVec3f(position)