"""Control controller."""
import collections as col
import math
import os.path
import os
import random
import time
from abc import ABC
from typing import Optional

import cv2
import gym
import numpy as np
from gym import spaces
from gym.core import ActType
from tensorflow import keras
from controller import Robot, Camera
#from SimulationControl import SimControl
os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://1234/WEBOT"

#Checking the memory usage
#from memory_profiler import profile

# This checks for overlap of the new point on any point already created
def anyOverlap(prev_points, cur_point, max_size) -> bool:
    for point in prev_points:
        dist = math.sqrt((point[0] - cur_point[0]) ** 2 + (point[1] - cur_point[1]) ** 2)
        if dist <= max_size:
            return True

    return False

# Returns the norm of a vector
def norm(v):
  cur_sum = float(0)
  for i in range(len(v)):
    cur_sum += v[i]**2
  return cur_sum** 0.5

# This is a testing function to print all nodes in the simulation
def printAllChildren(root):
    try:
        node_field = root.getField("children")
    except TypeError:
        raise Exception("Root has no children")
    else:
        print(node_field)

    # Get the number of nodes in the field
    num_nodes = node_field.getCount()
    print("Num nodes: " + str(num_nodes))

    # Loop through all the nodes and print their names
    for i in range(num_nodes):
        node = node_field.getMFNode(i)
        print(node.getName())

# This is the main class that inherits from environments
class CustomEnv(gym.Env, ABC):
    # Constructor setting up sensors and motors
    def __init__(self):
        # Reward parameters for the reward function. Tune during training
        self.target_reward = 1.0
        self.walking_reward = 10.0
        self.collision_reward = -10.0
        self.falling_reward = -10.0

        # Load the walking critic model
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        critic_path = os.path.join(parent_dir, '..', 'Critic.h5')
        self.critic = keras.models.load_model(critic_path)

        # Starting position of the robot
        self.start_pos = [0.0, 0.0, 0.285]

        # This is a step maximum to prevent infinite loop
        self.max_step = 10000000
        self.cur_step = 0
        self.step_pause = 0.001

        # For falling and collisions
        self.cancel_sim = False
        self.accel_threshold = 9.81*2 # Currently 2g
        self.height_threshold = 0.75 # Check this val

        # Motor constraints (Maybe change)
        self.begin_motor_pos = 0.0
        self.motor_max_pos_rad = np.pi
        self.motor_min_pos_rad = -np.pi
        self.motor_max_pos_deg = 359.99
        self.motor_min_pos_deg = 0.0
        self.motor_num = 12
        self.obs_num = 4

        # The space containing all the motors
        self.mot_space = spaces.Box(
            low=self.motor_min_pos_deg,
            high=self.motor_max_pos_deg,
            shape=(self.motor_num,),
            dtype=np.float32
        )

        # Camera constraints
        self.cam_fidelity = 512

        # Space containing the camera
        self.cam_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.cam_fidelity, self.cam_fidelity,),
            dtype=np.uint8
        )

        # Gyro constraints
        self.gyro_max = 32767
        self.gyro_min = -32767

        # Space containing the gyro
        self.gyro_space = spaces.Box(
            low=self.gyro_min,
            high=self.gyro_max,
            shape=(3,),
            dtype=np.float32
        )

        # Accel constraints. See if this is right
        self.accel_min = np.array([-10.0, -10.0, -10.0])
        self.accel_max = np.array([10.0, 10.0, 10.0])

        # Space containing the accel
        self.accel_space = spaces.Box(
            low=self.accel_min,
            high=self.accel_max,
            shape=(3,),
            dtype=np.float32
        )

        # The full observation space containing all sensors
        self.observation_space = spaces.Tuple((
            self.cam_space,
            self.mot_space,
            self.gyro_space,
            self.accel_space
        ))

        # Setting a buffer to hold old states and the current state
        self.buffer = col.deque([], maxlen=self.obs_num)

        # The action space for output. 12 motors
        self.action_space = spaces.Box(
                low=self.motor_min_pos_rad,
                high=self.motor_max_pos_rad,
                shape=(self.motor_num,),
                dtype=np.float32)

        # Init accel, gyro, global pos, target pos, camera, and dist
        self.position = self.start_pos
        #self.sim = SimControl()
       # self.sim.setRobotPosition(self.position)

        try:
            self.robot = Robot()
            self.robot = self.robot.created
            self.timestep = int(self.robot.basic_time_step)
        except TypeError:
            raise Exception("Robot not loaded")

        # Setting the devices
        motor_devices_name = ["PelvR", "PelvYR", "LegUpperR", "PelvL", "PelvYL", "LegUpperL", "LegLowerR", "LegLowerL",
                              "AnkleR", "FootR", "AnkleL", "FootL"]
        self.motor_devices = []
        # TODO I haven't been using this. Change to position sensor
        self.motor_positions = []

        # initialize devices
        for i, name in enumerate(motor_devices_name):
            self.motor_devices.append(self.robot.getDevice(name))
            self.motor_devices[i].setPosition(self.begin_motor_pos)
            self.motor_positions.append(self.begin_motor_pos)

        # Setting the camera
        self.cam = self.robot.getDevice("Camera")
        self.cam.enable(self.timestep)
        self.rawImage = self.cam.getImage
        self.image = self.convertImage()

        # Setting the gyro and accel
        self.gyro = self.robot.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        self.accel = self.robot.getDevice("Accelerometer")
        self.accel.enable(self.timestep)

        # Setting the random objects to avoid
        self.play_radius = 100
        self.max_obj_size = 1
        self.num_objects = 0  # For initial testing
        self.objects, self.objects_pos = self.placeObjects(self.num_objects, self.play_radius, self.max_obj_size)

        # Target that the robot walks to
        self.target = (random.uniform(-1, 1) * self.play_radius, random.uniform(-1, 1) * self.play_radius)

    # This resets the scene. Returns the starting position of everything
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):

        # Setting the position to the middle
        #self.switchRobotReference()
        self.position = self.start_pos
       # position_field = self.robot_node.getField("translation")
       # position_field.setSFVec3f(self.position)

        # Resetting motors
        for i in range(len(self.motor_devices)):
            self.motor_devices[i].setPosition(self.begin_motor_pos)
            self.motor_positions[i] = self.begin_motor_pos

        # Resetting the camera
        self.cam.disable()
        self.cam.enable(self.timestep)

        # Resetting sensors
        self.gyro.disable()
        self.gyro.enable(self.timestep)
        self.accel.disable()
        self.accel.enable(self.timestep)

        # Resetting map and target
        for obj in self.objects:
            obj.remove()

        self.objects = self.placeObjects(self.num_objects, self.play_radius, self.max_obj_size)
        self.target = (random.uniform(-1, 1) * self.play_radius, random.uniform(-1, 1) * self.play_radius)

        # Take an observation and return it
        obs = self.observe()
        for i in range(self.obs_num):
            self.buffer.append(obs)

        # TODO Check if this is the right thing to return
        return self.buffer

    # This randomly places objects
    # TODO I don't think this will work yet
    def placeObjects(self, num_objects, max_radius, max_size):
        objects_pos = set()
        objects_pos.add((0, 0))  # To protect robot at start
        objects = []

        for i in range(num_objects):
            # Random position generated
            pos_dif = False
            pos_x = 0.0
            pos_y = 0.0

            while not pos_dif:
                pos_x = random.uniform(-1, 1) * max_radius
                pos_y = random.uniform(-1, 1) * max_radius

                # Ensuring no repeat positions
                if not anyOverlap(objects_pos, (pos_x, pos_y), max_size):
                    pos_dif = True
                    objects_pos.add((pos_x, pos_y))

            # Making a copy of the box node
            # TODO might have to pass in the .proto file
            cur_obj = self.world.getFromDef("box")
            cur_obj.getField("translation").setSFVec3f([pos_x, pos_y, 0])

            # Random size generated
            size_x = random.uniform(0, 1) * max_size
            size_y = random.uniform(0, 1) * max_size
            size_z = random.uniform(0, 1) * max_size
            cur_obj.getField("size").setSFVec3f([size_x, size_y, size_z])

            # Setting the name of the object and adding it to the world
            cur_obj.getField("name").setSFString("box"+str(i))
            objects.append(cur_obj)
            # TODO How do I fix this?
            self.world.getRoot().addChild()

        return objects, objects_pos

    # Executed at each time step
    def step(self, action: ActType):
        # Take the action
        self.takeAction(action)
        self.cur_step += 1

        # Calculate reward
        reward, fallen, collision = self.rewardCalc()

        # Should the simulation be stopped
        done = self.isDone(fallen, collision)

        # Take the next observation and add to buffer
        self.buffer.append(self.observe())

        # TODO Check if this is the right return type
        return np.array(self.buffer), reward, done

    # This determines if the simulation needs to be reset
    def isDone(self, fallen, collision) -> bool:
        # Wait one update cycle to cancel the operation. For reward update
        if self.cancel_sim:
            return True

        if fallen or collision:
            self.cancel_sim = True

        # Max amount of steps
        if self.cur_step >= self.max_step:
            print("Max amount of steps reached")
            return True

        return False

    # This calculates the reward from the action
    def rewardCalc(self):
        # Getting closer to the target
        cur_pos = self.robot_node.getPosition()
        dist = math.sqrt(((cur_pos[0]-self.target[0])**2) + ((cur_pos[1]-self.target[1])**2))

        # How close to a step it is. Outputs a 1 or 0 (sigmoid)
        # TODO might have to change activation to softmax
        is_step = self.critic.predict(self.criticDataPrep())

        # Any collisions
        collision = anyOverlap(self.objects_pos, self.position, self.max_obj_size)

        # Fell over
        fallen = self.fellOver()

        # Total reward
        reward = (self.target_reward*(1/dist))+(self.walking_reward*is_step)+(self.collision_reward*collision)+(self.falling_reward*fallen)

        return reward, fallen, collision

    # This function executes the desired action. Sets motor positions.
    def takeAction(self, action: ActType):
        # Move the motors to their new positions
        for i, mot in enumerate(self.motor_devices):
            mot.setPosition(action[i])

        # Step the simulation
        self.robot.step(self.timestep)
        time.sleep(self.step_pause)

        # Reset robot position
        self.position = self.robot_node.getField("translation").value

    # This gets an image from the camera
    def convertImage(self):
        camera_data = np.empty(shape=(self.cam_fidelity, self.cam_fidelity), dtype='float')
        # Convert image to gray
        for r in range(self.cam_fidelity):
            for c in range(self.cam_fidelity):
                camera_data[r, c] = Camera.imageGetGray(self.rawImage, self.cam_fidelity, r, c)

        # Rescale pixel values to be between 0 and 1
        camera_data = camera_data.astype(np.float32) / 255.0

        # Add a channel dimension
        camera_data = np.expand_dims(camera_data, axis=2)

        return camera_data

    # This takes an observation of the environment
    def observe(self):
        observation = (self.image,
                       self.getMotorPos(),
                       self.gyro.getValues(),
                       self.accel.getValues)

        return observation

    # This function checks if the robot fell
    def fellOver(self) -> bool:
        # Height too low
        if self.robot_node.getPosition()[2] <= self.height_threshold:
            print("Height threshold broken")
            return True

        # Acceleration too much in any direction
        accel_values = np.array(self.accel.getValues())
        accel_magnitude = np.linalg.norm(accel_values)
        if accel_magnitude >= self.accel_threshold:
            print("Acceleration threshold broken")
            return True

        return False

    # Gets data from the motors and covert to positive degrees
    def getMotorPos(self):
        # Take in data and convert to degrees
        # TODO might read from the actual sensors
        motors = [np.rad2deg(self.motor_devices[i].getTargetPosition()) for i in range(self.motor_num)]

        # Convert all angles to positive
        for i in range(len(motors)):
            if motors[i] < 0:
                motors[i] += 360.0

        return motors

    # Preps the data to be passed into the critic
    def criticDataPrep(self):
        # Grab motor positions
        mot_obs = np.zeros((self.obs_num, self.motor_num))
        for i, obs in enumerate(self.buffer):
            mot_obs[i] = obs[1]

        # Normalize the data
        tmp = np.reshape(mot_obs, (self.obs_num * self.motor_num))
        tmp_norm = norm(tmp)
        mot_obs = mot_obs / tmp_norm

        return mot_obs

    # The changes reference to the robot. Can't have two references (node or robot)
    def switchRobotReference(self):
        if (self.robot is None) and (self.robot_node is not None):
            self.robot_node = None
            try:
                self.robot = Robot()
                self.robot = self.robot.created
            except TypeError:
                raise Exception("Robot not loaded")

        elif (self.robot is not None) and (self.robot_node is None):
            self.robot = None
            try:
                self.robot_node = self.world.getFromDef("WEBOT")
            except TypeError:
                raise Exception("Robot node not found")

        else:
            print("Error switching robot references")

