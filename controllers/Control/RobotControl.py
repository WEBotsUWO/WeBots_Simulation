"""Control controller."""
import collections as col
import math
import os.path
import os
import random
import time
from abc import ABC
from typing import Optional
import sys
from dotenv import load_dotenv

load_dotenv()


sys.path.append(os.getenv('PYTHONPATH'))
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType
import tensorflow as tf
from tensorflow import keras
from controller import Robot, Camera, Supervisor, GPS
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

        JOINT_LIMITS = {
            #  name        window°   vmax°/s   tmax Nm
            "PelvR"   :  ( 35,       180,      6.0 ),
            "PelvYR"  :  ( 45,        90,      4.0 ),
            "LegUpperR": (120,       120,      8.0 ),
            "PelvL"   :  ( 35,       180,      6.0 ),
            "PelvYL"  :  ( 45,        90,      4.0 ),
            "LegUpperL": (120,       120,      8.0 ),
            "LegLowerR": ( 90,       160,      5.0 ),
            "LegLowerL": ( 90,       160,      5.0 ),
            "AnkleR"  :  ( 45,       200,      3.0 ),
            "FootR"   :  ( 25,       200,      2.5 ),
            "AnkleL"  :  ( 45,       200,      3.0 ),
            "FootL"   :  ( 25,       200,      2.5 ),
        }

        try:
            self.robot = Supervisor() 
            self.robot = self.robot.created
            self.timestep = int(self.robot.basic_time_step)
        except TypeError:
            raise Exception("Robot not loaded")
        
        parent_dir  = os.path.dirname(os.path.abspath(__file__))
        critic_path = os.path.join(parent_dir, 'Critic.keras')

        self.critic = None
        try:
            print(f"[CustomEnv] Loading critic: {critic_path}")
            assert os.path.exists(critic_path), f"Critic file not found at {critic_path}"
            self.critic = keras.models.load_model(critic_path)
        except Exception as e:
            print(f"[CustomEnv] WARNING: critic unavailable ({e}). Using 0.0 for is_step.")


   
        parent   = self.robot.getFromDef("boxs")
        children = parent.getField("children")
        self.cubes = [children.getMFNode(i) for i in range(children.getCount())]  # CUBE protos or Solids
        self.cube_z0 = [n.getField("translation").getSFVec3f()[2] for n in self.cubes]

        # per-cube XY velocity [m/s] and yaw rate [rad/s]
        rng = np.random.default_rng(0)
        self.cube_vel = [rng.uniform(-0.3, 0.3, size=2).astype(float) for _ in self.cubes]
        self.cube_wz  = [float(rng.uniform(-0.8, 0.8)) for _ in self.cubes]
        


        # Starting position of the robot
        self.start_pos = [0.0, 0.0, 0.285]

        # This is a step maximum to prevent infinite loop
        self.max_step = 10000000
        self.cur_step = 0
        self.step_pause = 0.001

        # For falling and collisions
        self.cancel_sim = False
        self.accel_threshold = 9.81*2 # Currently 2g
        self.height_threshold = 0.15 # Check this val

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
            dtype=np.float64
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
            dtype=np.float64
        )

        # Accel constraints. See if this is right
        self.accel_min = np.array([-10.0, -10.0, -10.0])
        self.accel_max = np.array([10.0, 10.0, 10.0])

        # Space containing the accel
        self.accel_space = spaces.Box(
            low=self.accel_min,
            high=self.accel_max,
            shape=(3,),
            dtype=np.float64
        )
        self.gps_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float64)

        # The full observation space containing all sensors
        self.observation_space = spaces.Tuple((
            self.cam_space,
            self.mot_space,
            self.gyro_space,
            self.accel_space,
            self.gps_space,
        ))

        # Setting a buffer to hold old states and the current state
        self.buffer = col.deque([], maxlen=self.obs_num)

        # The action space for output. 12 motors
        self.action_space = spaces.Box(
                low=self.motor_min_pos_rad,
                high=self.motor_max_pos_rad,
                shape=(self.motor_num,),
                dtype=np.float64)

        # Init accel, gyro, global pos, target pos, camera, and dist
        self.position = self.start_pos
        #self.sim = SimControl()
       # self.sim.setRobotPosition(self.position)



        # Setting the devices
        motor_devices_name = ["PelvR", "PelvYR", "LegUpperR", "PelvL", "PelvYL", "LegUpperL", "LegLowerR", "LegLowerL",
                              "AnkleR", "FootR", "AnkleL", "FootL"]
        self.motor_devices = []
        self.joint_sensors = []
        
        

        # initialize devices
        for i, name in enumerate(motor_devices_name):
            self.motor_devices.append(self.robot.getDevice(name))
            self.motor_devices[i].setPosition(self.begin_motor_pos)
            
        #Initialize sensors
        for mot in self.motor_devices:
            sensor = mot.getPositionSensor()
            sensor.enable(self.timestep)
            self.joint_sensors.append(sensor)

        deg2rad = np.deg2rad

        # Capture robot’s *current* pose
        start_rad = np.array([m.getTargetPosition() for m in self.motor_devices],
                             dtype=np.float64)

        # Make per-joint arrays
        window   = deg2rad([JOINT_LIMITS[m.getName()][0] for m in self.motor_devices])
        vmax_rad = deg2rad([JOINT_LIMITS[m.getName()][1] for m in self.motor_devices])
        tmax_nm  = np.array([JOINT_LIMITS[m.getName()][2] for m in self.motor_devices],
                            dtype=np.float64)

        soft_min = start_rad - window
        soft_max = start_rad + window

        # never exceed Webots hard limits
        hard_min = np.array([m.getMinPosition() for m in self.motor_devices])
        hard_max = np.array([m.getMaxPosition() for m in self.motor_devices])
        soft_min = np.maximum(soft_min, hard_min)
        soft_max = np.minimum(soft_max, hard_max)

        # Configure each motor once
        for m, v_lim, t_lim in zip(self.motor_devices, vmax_rad, tmax_nm):
            m.setPosition(float('inf'))   # velocity-control mode
            m.setVelocity(v_lim)          # per-joint speed cap
            m.setTorque(t_lim)            # per-joint torque cap

        # Update Gym action_space (position targets)
        self.action_space = spaces.Box(low=soft_min,
                                       high=soft_max,
                                       dtype=np.float64)

    
        self._soft_min = soft_min
        self._soft_max = soft_max

        # Setting the camera
        self.cam = self.robot.getDevice("Camera")
        self.cam.enable(self.timestep)

        self.robot.step(self.timestep)      

        raw = self.cam.getImage()           
        self.image = self.convertImage(raw)

        # Setting the gyro and accel
        self.gyro = self.robot.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        self.dt = self.timestep / 1000.0        # ms → s
        self.orientation = np.zeros(3)          # roll, pitch, yaw  (rad)

        self.accel = self.robot.getDevice("Accelerometer")
        self.accel.enable(self.timestep)

        self.gps = self.robot.getDevice("gps")   
        self.gps.enable(self.timestep) 

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
            #self.joint_sensors[i] = self.begin_motor_pos

        # Resetting the camera
        self.cam.disable()
        self.cam.enable(self.timestep)

        # Resetting sensors
        self.gyro.disable()
        self.gyro.enable(self.timestep)
        self.accel.disable()
        self.accel.enable(self.timestep)

        # Resetting map and target
        if getattr(self, "objects", None):
            for node in list(self.objects):
                try:
                    node.remove()
                except Exception:
                    pass
                
        # re-spawn and UNPACK the tuple that placeObjects returns
        self.objects, self.objects_pos = self.placeObjects(
            self.num_objects, self.play_radius, self.max_obj_size
        )
        self.target = (random.uniform(-1, 1) * self.play_radius, random.uniform(-1, 1) * self.play_radius)

        self.buffer.clear()
        o = self.observe()
        for _ in range(self.obs_num):
            self.buffer.append(o)
        obs  = self._stack_obs()
        info = {}
        return obs, info

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

        terminated = bool(fallen or collision)
        truncated  = bool(self.cur_step >= self.max_step)

        obs  = self._stack_obs()
        info = {}

        return obs, reward, terminated, truncated, info

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
        """Compute task reward, fall flag, and collision flag."""

        # ── 1. Distance-to-target term (use GPS X-Y) ────────────────────────────
        cur_xy     = np.array(self.position[:2], dtype=np.float32)    # [x, y] from GPS
        target_xy  = np.array(self.target,      dtype=np.float32)     # [x, y]
        dist       = np.linalg.norm(cur_xy - target_xy) + 1e-6        # ε avoids div-by-zero

        # reward grows as we get closer
        dist_rwd = self.target_reward * (1.0 / dist)

        # ── 2. Walking-gait term from the critic ───────────────────────────────
        if self.critic is None:
            is_step = 0.0
        else:
            pred = self.critic.predict(self.criticDataPrep(), verbose=0)
            is_step = float(np.squeeze(pred))   # robust to shapes like (1,1) or (1,)
        gait_rwd = self.walking_reward * is_step

        # ── 3. Collision & fall penalties ──────────────────────────────────────
        collision = anyOverlap(self.objects_pos, self.position, self.max_obj_size)
        fallen    = self.fellOver()

        coll_pen  = self.collision_reward * float(collision)
        fall_pen  = self.falling_reward   * float(fallen)

        # ── 4. Sum components ──────────────────────────────────────────────────
        reward = dist_rwd + gait_rwd + coll_pen + fall_pen

        return reward, fallen, collision
    
    def set_cube_velocity(self, i, vx, vy, wz=0.0):
        self.cube_vel[i] = np.array([vx, vy], float)
        self.cube_wz[i]  = float(wz)

    # This function executes the desired action. Sets motor positions.
    def takeAction(self, action: ActType):
        action = np.clip(action, self._soft_min, self._soft_max)
        for mot, tgt in zip(self.motor_devices, action):
            mot.setPosition(tgt)        # Webots will move at the per-joint vmax
        self.robot.step(self.timestep)
        for i, (n, z0) in enumerate(zip(self.cubes, self.cube_z0)):
            vx, vy = self.cube_vel[i]
            wz     = self.cube_wz[i]

            # translate on the plane
            tf = n.getField("translation")
            x, y, z = tf.getSFVec3f()
            tf.setSFVec3f([x + vx * self.dt, y + vy * self.dt, z0])

            # rotate about Z
            rf = n.getField("rotation")
            axis_angle = rf.getSFRotation()
            ang = axis_angle[3]
            rf.setSFRotation([0, 0, 1, ang + wz * self.dt])

            n.resetPhysics()   

        omega = np.array(self.gyro.getValues())          # rad/s
        self.orientation += omega * self.dt              # θ_new = θ_old + ω·dt
        time.sleep(self.step_pause)

        joint_angles = [s.getValue() for s in self.joint_sensors]

        # Reset robot position
        self.position = self.gps.getValues()
        #self.position = self.robot_node.getField("translation").value

    # This gets an image from the camera
    def convertImage(self, raw_image):
        w, h = self.cam.getWidth(), self.cam.getHeight()    
        gray = np.empty((h, w), dtype=np.float64)

        for y in range(h):
            for x in range(w):
                gray[y, x] = Camera.imageGetGray(raw_image, w, x, y) / 255.0

        return gray[..., None]    # (H, W, 1)

    def _clamp_cube_planar(self, node, z0):
        # lock Z translation on the node itself
        t_f = node.getField("translation")
        t   = list(t_f.getSFVec3f()); t[2] = z0
        t_f.setSFVec3f(t)

        # keep only yaw (about world Z) on the node itself
        R   = np.array(node.getOrientation()).reshape(3, 3)   # world rotation
        yaw = math.atan2(R[1, 0], R[0, 0])
        node.getField("rotation").setSFRotation([0, 0, 1, yaw])

        # if this node is (or wraps) a Solid, zero forbidden velocities & flush momentum
        try:
            v = list(node.getVelocity())                      # [vx, vy, vz, wx, wy, wz]
            v[2] = 0.0; v[3] = 0.0; v[4] = 0.0
            node.setVelocity(v)
            node.resetPhysics()
        except Exception:
            pass  # some PROTOs won’t expose Solid methods; rotation/translation still locked

    def _stack_obs(self):
        imgs   = np.stack([o[0] for o in self.buffer], axis=0)  # (N,H,W,1)
        joints = np.stack([o[1] for o in self.buffer], axis=0)  # (N,12)
        gyros  = np.stack([o[2] for o in self.buffer], axis=0)  # (N,3)
        accels = np.stack([o[3] for o in self.buffer], axis=0)  # (N,3)
        gpss   = np.stack([o[4] for o in self.buffer], axis=0)  # (N,3)
        return (imgs, joints, gyros, accels, gpss)

    # This takes an observation of the environment
    def observe(self):
        image  = self.convertImage(self.cam.getImage())
        joints = np.array([s.getValue() for s in self.joint_sensors], dtype=np.float64)
        gyro   = np.array(self.gyro.getValues(),  dtype=np.float64)
        accel  = np.array(self.accel.getValues(), dtype=np.float64)
        gps    = np.array(self.gps.getValues(),   dtype=np.float64)
        return (image, joints, gyro, accel, gps)

    # This function checks if the robot fell
    def fellOver(self) -> bool:
        z_pos = self.position[2]                 # self.position is GPS [x, y, z]
        if z_pos <= self.height_threshold:
            print(f"Height threshold broken: z = {z_pos:.3f} m")
            return True
 
        accel_values     = np.array(self.accel.getValues())   # m/s²
        accel_magnitude  = np.linalg.norm(accel_values)
        if accel_magnitude >= self.accel_threshold:
            print(f"Acceleration threshold broken: |a| = {accel_magnitude:.2f} m/s²")
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
       """Return a (1, obs_num, motor_num) array ready for self.critic.predict()."""

       # stack the last `obs_num` motor-position rows from the buffer
       mot_obs = np.zeros((self.obs_num, self.motor_num), dtype=np.float32)
       for i, obs in enumerate(self.buffer):
           mot_obs[i] = obs[1]                         # obs[1] = motor angles

       # normalise (L2) – add ε to avoid div-by-zero
       flat      = mot_obs.reshape(-1)
       denom     = np.linalg.norm(flat) + 1e-8
       mot_obs   = mot_obs / denom

       # add batch axis so shape becomes (1, 4, 12)
       mot_obs = np.expand_dims(mot_obs, axis=0)

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

