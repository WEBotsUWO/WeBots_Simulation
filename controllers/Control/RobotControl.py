# =============================================================================
# ROBOTIS OP3 HUMANOID ROBOT CONTROL MODULE
# =============================================================================
# Gymnasium environment for training humanoid walking behavior using
# evolutionary algorithms with Webots physics simulation
# =============================================================================
import collections as col
import math
import os.path
import os
import random
import time
import threading
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
os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://1234/WEBOT"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def anyOverlap(prev_points, cur_point, max_size) -> bool:
    for point in prev_points:
        dist = math.sqrt((point[0] - cur_point[0]) ** 2 + (point[1] - cur_point[1]) ** 2)
        if dist <= max_size:
            return True

    return False

def norm(v):
  cur_sum = float(0)
  for i in range(len(v)):
    cur_sum += v[i]**2
  return cur_sum** 0.5

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

# =============================================================================
# CUSTOM ENVIRONMENT CLASS
# =============================================================================
class CustomEnv(gym.Env, ABC):
    def __init__(self, port=None, instance_id=None):
        # Reward system parameters for human-like locomotion
        self.forward_reward = 8.0
        self.stability_reward = 4.0
        self.energy_penalty = -0.05
        self.smoothness_reward = 2.0
        self.gait_reward = 6.0
        self.collision_penalty = -30.0
        self.fall_penalty = -50.0
        self.height_bonus = 3.0
        self._stall_steps = 0
        self._last_joints = None
        self._last_y = 0.0
        self._cpg_on_until = 0.0
        self._prev_action = None

        # Joint limits based on human anatomy
        JOINT_LIMITS = {
            "PelvR"   :  ( 30,       120,      6.0 ),
            "PelvYR"  :  ( 40,        80,      4.0 ),
            "LegUpperR": ( 90,       100,      8.0 ),
            "PelvL"   :  ( 30,       120,      6.0 ),
            "PelvYL"  :  ( 40,        80,      4.0 ),
            "LegUpperL": ( 90,       100,      8.0 ),
            "LegLowerR": ( 70,       120,      5.0 ),
            "LegLowerL": ( 70,       120,      5.0 ),
            "AnkleR"  :  ( 30,       150,      3.0 ),
            "FootR"   :  ( 20,       150,      2.5 ),
            "AnkleL"  :  ( 30,       150,      3.0 ),
            "FootL"   :  ( 20,       150,      2.5 )
        }

        # Instance identification
        self.port = port
        self.instance_id = instance_id or f"instance_port_{port}" if port else "single_instance"
        
        try:
            self.robot = Supervisor() 
            self.robot = self.robot.created
            # Use smaller timestep for smoother simulation (e.g., 8ms instead of default)
            self.timestep = 8  # milliseconds
            
            # Set simulation to fastest mode (no speed limit)
            self.robot.simulationSetMode(self.robot.SIMULATION_MODE_FAST)
            
            # Configure controller URL if port is specified (for multi-instance)
            if port:
                controller_url = f"ipc://{port}/WEBOT"
                os.environ["WEBOTS_CONTROLLER_URL"] = controller_url
                print(f"Instance {self.instance_id} configured for port {port}")
            
            self.robot_node = self.robot.getSelf()  # Get the robot node itself
            if self.robot_node is None:
                self.robot_node = self.robot.getFromDef("ROBOT")  # Common DEF name
                if self.robot_node is None:
                    print(f"Warning: {self.instance_id} could not get robot node reference")
            
        except TypeError:
            raise Exception(f"Robot not loaded for {self.instance_id}")
        
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

        self.rng = np.random.default_rng()
        
        self.teleport_boxes()
        


        self.start_pos = [0.0, 0.0, 0.285]

        self.max_step = 10000000
        self.cur_step = 0

        self.cancel_sim = False
        self.accel_threshold = 9.81 * 8.0
        self.height_threshold = 0.09
        self.min_walking_speed = 0.01

        # Initial standing positions for joint reset
        self.initial_standing_positions = [
            0.0,   # PelvR
            0.0,   # PelvYR
            0.0,   # LegUpperR
            0.0,   # PelvL
            0.0,   # PelvYL
            0.0,   # LegUpperL
            0.0,   # LegLowerR
            0.0,   # LegLowerL
            0.0,   # AnkleR
            0.0,   # FootR
            0.0,   # AnkleL
            0.0    # FootL
        ]
        
        self.actual_start_positions = None
        
        self.begin_motor_pos = 0.0
        self.motor_max_pos_rad = np.pi
        self.motor_min_pos_rad = -np.pi
        self.motor_max_pos_deg = 359.99
        self.motor_min_pos_deg = 0.0
        self.motor_num = 12
        self.obs_num = 4

        self.mot_space = spaces.Box(
            low=self.motor_min_pos_deg,
            high=self.motor_max_pos_deg,
            shape=(self.motor_num,),
            dtype=np.float64
        )
        

        self.cam_fidelity = 512

        self.cam_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.cam_fidelity, self.cam_fidelity,),
            dtype=np.uint8
        )

        self.gyro_max = 32767
        self.gyro_min = -32767

        self.gyro_space = spaces.Box(
            low=self.gyro_min,
            high=self.gyro_max,
            shape=(3,),
            dtype=np.float64
        )

        self.accel_min = np.array([-10.0, -10.0, -10.0])
        self.accel_max = np.array([10.0, 10.0, 10.0])

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
        
        

        # initialize devices with proper standing positions
        for i, name in enumerate(motor_devices_name):
            self.motor_devices.append(self.robot.getDevice(name))
            # Set to proper standing position for each joint
            self.motor_devices[i].setPosition(self.initial_standing_positions[i])
            
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
        self._vmax_rad = np.array([np.deg2rad(JOINT_LIMITS[m.getName()][1]) for m in self.motor_devices], dtype=np.float64)
        self._tmax_nm  = np.array([JOINT_LIMITS[m.getName()][2] for m in self.motor_devices], dtype=np.float64)

        for m, v_lim, t_lim in zip(self.motor_devices, self._vmax_rad, self._tmax_nm):
            m.setPosition(float('inf'))   # velocity-control mode
            m.setVelocity(float(v_lim))   # cap
            m.setTorque(float(t_lim))     # cap

        # never exceed Webots hard limits
        # build “soft” range around current pose, then clamp to Webots hard limits
        soft_min = start_rad - window                    # radians
        soft_max = start_rad + window

        hard_min = np.array([m.getMinPosition() for m in self.motor_devices], dtype=np.float64)
        hard_max = np.array([m.getMaxPosition() for m in self.motor_devices], dtype=np.float64)

        soft_min = np.maximum(soft_min, hard_min)       # clamp to hard limits
        soft_max = np.minimum(soft_max, hard_max)

        # stash for control + define action space
        self._soft_min = soft_min
        self._soft_max = soft_max

        self.action_space = spaces.Box(
            low=self._soft_min,
            high=self._soft_max,
            dtype=np.float64
        )

        # Setting the camera
        self.cam = self.robot.getDevice("Camera")
        self.cam.enable(self.timestep)

        self.robot.step(self.timestep)      

        raw = self.cam.getImage()           
        self.image = self.convertImage(raw)

        # Setting the gyro and accel
        self.gyro = self.robot.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        self.dt = self.timestep / 1000.0
        self.orientation = np.zeros(3)

        self.accel = self.robot.getDevice("Accelerometer")
        self.accel.enable(self.timestep)

        self.gps = self.robot.getDevice("gps")   
        self.gps.enable(self.timestep) 

        self.play_radius = 100
        self.max_obj_size = 1
        self.num_objects = 0
        self.objects, self.objects_pos = self.placeObjects(self.num_objects, self.play_radius, self.max_obj_size)

        self.target = (random.uniform(-1, 1) * self.play_radius, random.uniform(-1, 1) * self.play_radius)
        
        self.current_action = np.zeros(self.motor_num)
        self.action_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.ml_thread = None
        self.running = True
        
        self.is_paused = False
        self.last_step_time = time.time()
        self.step_timeout = 5.0
        
        self.episode_count = 0
        self.total_steps = 0
        self.learning_active = False
        self.start_time = time.time()
        
        self.prev_position = np.array([0.0, 0.0, 0.285])
        self.prev_joint_positions = None
        self.step_count_for_gait = 0
        
        self.capture_initial_positions()
        
        print(f"Environment initialized successfully: {self.instance_id}")
        print(f"Robot positioned in neutral upright stance for training")


    # =============================================================================
    # MOTOR CONTROL METHODS
    # =============================================================================
    
    def _apply_motor_caps(self):
        """Reapply per-joint speed/torque caps after any reset/settle."""
        for m, v_lim, t_lim in zip(self.motor_devices, self._vmax_rad, self._tmax_nm):
            m.setVelocity(float(v_lim))
            m.setTorque(float(t_lim))
    

    # =============================================================================
    # ENVIRONMENT RESET AND SETUP
    # =============================================================================
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):

        self.position = self.start_pos

        for i in range(len(self.motor_devices)):
            self.motor_devices[i].setPosition(self.initial_standing_positions[i])

        self.cam.disable()
        self.cam.enable(self.timestep)

        self.gyro.disable()
        self.gyro.enable(self.timestep)
        self.accel.disable()
        self.accel.enable(self.timestep)

        self.teleport_boxes()
        
        self.target = (random.uniform(-1, 1) * self.play_radius, random.uniform(-1, 1) * self.play_radius)

        self.buffer.clear()
        o = self.observe()
        for _ in range(self.obs_num):
            self.buffer.append(o)
        obs  = self._stack_obs()
        info = {}
        return obs, info

    def placeObjects(self, num_objects, max_radius, max_size):
        objects_pos = set()
        objects_pos.add((0, 0))  # To protect robot at start
        objects = []

        for i in range(num_objects):
            pos_dif = False
            pos_x = 0.0
            pos_y = 0.0

            while not pos_dif:
                pos_x = random.uniform(-1, 1) * max_radius
                pos_y = random.uniform(-1, 1) * max_radius

                if not anyOverlap(objects_pos, (pos_x, pos_y), max_size):
                    pos_dif = True
                    objects_pos.add((pos_x, pos_y))

            cur_obj = self.world.getFromDef("box")
            cur_obj.getField("translation").setSFVec3f([pos_x, pos_y, 0])

            size_x = random.uniform(0, 1) * max_size
            size_y = random.uniform(0, 1) * max_size
            size_z = random.uniform(0, 1) * max_size
            cur_obj.getField("size").setSFVec3f([size_x, size_y, size_z])

            cur_obj.getField("name").setSFString("box"+str(i))
            objects.append(cur_obj)
            self.world.getRoot().addChild()

        return objects, objects_pos
    

    # =============================================================================
    # MOVEMENT ANALYSIS AND STALL DETECTION
    # =============================================================================
    
    def _detect_and_handle_stall(self):
        """If little movement for N steps, engage a short CPG burst."""
        joints = np.array([s.getValue() for s in self.joint_sensors])
        y = float(self.position[1])

        joint_delta = 0.0 if self._last_joints is None else float(np.max(np.abs(joints - self._last_joints)))
        y_progress  = y - self._last_y

        moving_forward = y_progress > 0.001
        joints_moving  = joint_delta > 0.002

        if (not moving_forward) and (not joints_moving):
            self._stall_steps += 1
        else:
            self._stall_steps = 0

        self._last_joints = joints
        self._last_y = y

        if self._stall_steps > int(1000/self.timestep):
            self._cpg_on_until = time.time() + 2.0
            self._stall_steps = 0
    
    def teleport_boxes(self):
        """Teleport boxes to random locations within 3-20m of robot"""
        if not hasattr(self, 'cubes') or len(self.cubes) == 0:
            return
            
        try:
            robot_x, robot_y = self.position[:2] if hasattr(self, 'position') else [0, 0]
            min_dist = 3.0
            max_dist = 6.0
            
            for i, (cube, z0) in enumerate(zip(self.cubes, self.cube_z0)):
                distance = self.rng.uniform(min_dist, max_dist)
                angle = self.rng.uniform(0, 2 * np.pi)
                
                x = robot_x + distance * np.cos(angle)
                y = robot_y + distance * np.sin(angle)
                
                translation_field = cube.getField("translation")
                if translation_field:
                    translation_field.setSFVec3f([x, y, z0])
                
                rotation_field = cube.getField("rotation")
                if rotation_field:
                    rot_angle = self.rng.uniform(0, 2 * np.pi)
                    rotation_field.setSFRotation([0, 0, 1, rot_angle])
                
                cube.resetPhysics()
            
            print(f"Teleported {len(self.cubes)} boxes within 3-20m of robot")
            
        except Exception as e:
            print(f"Error teleporting boxes: {e}")
    
    def check_box_collision(self):
        """Check collision with current box positions (optimized)"""
        if not hasattr(self, 'cubes') or len(self.cubes) == 0:
            return False
            
        robot_x, robot_y = self.position[:2]
        collision_radius = 1.0
        
        for cube in self.cubes:
            try:
                translation_field = cube.getField("translation")
                if translation_field:
                    box_pos = translation_field.getSFVec3f()
                    box_x, box_y = box_pos[:2]
                    
                    dist = np.sqrt((robot_x - box_x)**2 + (robot_y - box_y)**2)
                    if dist < collision_radius:
                        return True
            except:
                continue
        
        return False

    # =============================================================================
    # SIMULATION HEALTH AND MONITORING
    # =============================================================================
    
    def check_simulation_health(self):
        """Check if simulation is paused or stuck"""
        current_time = time.time()
        
        # Check if simulation is responding
        if current_time - self.last_step_time > self.step_timeout:
            print("Warning: Simulation appears to be paused or stuck")
            self.is_paused = True
            return False
        
        self.is_paused = False
        return True
    
    # Continuous simulation loop - runs independently 
    def run_continuous_simulation(self):
        """Runs simulation at full speed with safety checks"""
        print("Starting continuous simulation loop...")
        print("Simulation running - Press Ctrl+C to stop safely")
        
        while self.running:
            try:
                # Safety check - detect if simulation is paused
                if not self.check_simulation_health():
                    print("Simulation paused - waiting...")
                    time.sleep(1.0)
                    continue
                self._detect_and_handle_stall()

                with self.action_lock:
                    action = self.current_action.copy()

                # If CPG is active, override action
                if time.time() < self._cpg_on_until:
                    tsec = time.time() - self.start_time
                    action = self._cpg_action(tsec)
                
                # Get current action thread-safely
                with self.action_lock:
                    action = self.current_action.copy()
                
                # Apply action and step simulation
                self.setAction(action)
                
                # Try to step simulation - detect if it fails
                step_start = time.time()
                step_result = self.robot.step(self.timestep)
                
                # Check if step succeeded (non-zero return means simulation running)
                if step_result == -1:
                    print("Simulation terminated by Webots")
                    break
                    
                self.last_step_time = time.time()
                self.updateSimulation()
                
                self.cur_step += 1
                self.total_steps += 1
                
                # Check termination conditions less frequently
                if self.cur_step % 100 == 0:
                    reward, fallen, collision = self.rewardCalc()
                    print(f"[{self.instance_id}] Episode {self.episode_count}, Step {self.cur_step}: Height={self.position[2]:.3f}, Fallen={fallen}, Collision={collision}")
                    
                    if fallen:
                        print(f"[{self.instance_id}] Robot fell - resetting...")
                        self.reset_simulation_state()
                    elif collision:
                        print(f"[{self.instance_id}] Collision detected - resetting...")
                        self.reset_simulation_state()
                
                # Auto-save progress periodically
                if self.should_save_model():
                    self.save_progress()
                
                # Long-running status updates
                if self.total_steps % 10000 == 0:
                    hours = (time.time() - self.start_time) / 3600
                    avg_episode_length = self.total_steps / max(1, self.episode_count)
                    print(f"Running for {hours:.1f} hours, Total steps: {self.total_steps}, Episodes: {self.episode_count}, Avg episode length: {avg_episode_length:.1f}")
                    
                    # Force save every 10k steps as backup
                    if self.learning_active:
                        self.save_progress()
                
            except KeyboardInterrupt:
                print("\nReceived Ctrl+C - shutting down safely...")
                break
            except Exception as e:
                print(f"Simulation error: {e}")
                print("Attempting to recover...")
                time.sleep(1.0)  # Brief pause before retry
                continue
        
        print("Continuous simulation loop ended.")
    
    # =============================================================================
    # SIMULATION STATE MANAGEMENT
    # =============================================================================
    
    def reset_simulation_state(self):
        """Reset robot to exact upright position from simulation start"""
        try:
            print("Resetting robot to captured upright position...")
            
            # Method 1: Complete physics reset using Supervisor API
            if self.robot_node is not None:
                # Reset position (translation field)
                translation_field = self.robot_node.getField("translation")
                if translation_field is not None:
                    translation_field.setSFVec3f(self.start_pos)
                    print(f"Set translation to: {self.start_pos}")
                
                # Reset orientation to upright (rotation field)
                rotation_field = self.robot_node.getField("rotation")
                if rotation_field is not None:
                    # Reset to upright orientation (no rotation around any axis)
                    rotation_field.setSFRotation([0, 0, 1, 0])  # [axis_x, axis_y, axis_z, angle]
                    print("Reset orientation to upright")
                
                # Reset physics to clear velocities and forces
                self.robot_node.resetPhysics()
                print("Physics reset applied")
                
                # Method 2: Skip Supervisor joint reset (nodes not accessible)
                print("Supervisor joint reset not available - using motor reset instead")
                joint_reset_success = False
                
            else:
                print("Warning: No robot node reference - using motor reset only")
                joint_reset_success = False
                
            # Method 3: Motor position reset (primary reset method)
            print("Using reliable motor position commands...")
            self.log_motor_positions(self.initial_standing_positions, "(RESET TARGETS)")
            for i, motor in enumerate(self.motor_devices):
                motor.setPosition(self.initial_standing_positions[i])
                motor.setVelocity(0.05)  # small settling velocity instead of 0

            # Step B: let it settle
            for step in range(15):
                if step % 5 == 0:
                    for i, motor in enumerate(self.motor_devices):
                        motor.setPosition(self.initial_standing_positions[i])
                self.robot.step(self.timestep)

            # Step C: RESTORE normal caps so actions move joints again
            self._apply_motor_caps()
            
            # Verify standing position was applied correctly
            print("\n=== CHECKING RESET RESULTS ===")
            current_positions = [sensor.getValue() for sensor in self.joint_sensors]
            self.log_motor_positions(current_positions, "(AFTER RESET)")
            self.verify_standing_position()
            
            # Update position tracking
            self.position = self.start_pos.copy()
            
            # Reset step counter for this episode
            self.cur_step = 0
            self.cancel_sim = False
            self.episode_count += 1  # Track episodes
            
            # Teleport boxes to new random locations
            self.teleport_boxes()
            
            # Clear buffer safely
            with self.buffer_lock:
                self.buffer.clear()
                # Refill buffer with current observation
                obs = self.observe()
                for _ in range(self.obs_num):
                    self.buffer.append(obs)
            
            print(f"Episode {self.episode_count} started - Simulation state reset complete")
        except Exception as e:
            print(f"Reset error: {e}")
            import traceback
            traceback.print_exc()



    # =============================================================================
    # CENTRAL PATTERN GENERATOR (CPG)
    # =============================================================================
    
    def _cpg_action(self, t):
        """Simple symmetric gait around standing pose in radians, output in [-1,1] then scaled."""
        # base = self.initial_standing_positions (radians)
        base = np.array(self.initial_standing_positions, dtype=np.float64)

        # amplitudes (radians): modest swings
        A_hip   = 0.20
        A_knee  = 0.25
        A_ankle = 0.10

        # frequency (Hz) and phase offsets
        f   = 0.9
        w   = 2*np.pi*f
        ph  = 0.0
        phL = 0.0
        phR = np.pi  # out of phase

        # indices for readability
        PelvR, PelvYR, HipR, PelvL, PelvYL, HipL, KneeR, KneeL, AnkleR, FootR, AnkleL, FootL = range(12)
        act = base.copy()

        # Hips
        act[HipL]  = base[HipL]  + A_hip  * np.sin(w*t + phL)
        act[HipR]  = base[HipR]  + A_hip  * np.sin(w*t + phR)
        # Knees (lead hips by 90°)
        act[KneeL] = base[KneeL] + A_knee * np.sin(w*t + phL + np.pi/2)
        act[KneeR] = base[KneeR] + A_knee * np.sin(w*t + phR + np.pi/2)
        # Ankles (counter knees slightly)
        act[AnkleL]= base[AnkleL]+ A_ankle* np.sin(w*t + phL - np.pi/2)
        act[AnkleR]= base[AnkleR]+ A_ankle* np.sin(w*t + phR - np.pi/2)

        # Clamp to soft limits
        act = np.clip(act, self._soft_min, self._soft_max)

        # Convert from absolute rad targets to normalized [-1,1] for setAction scaling path
        norm = 2.0*(act - self._soft_min)/(self._soft_max - self._soft_min) - 1.0
        return norm
    
    def verify_standing_position(self):
        """Verify that robot is in proper upright position after reset (values in radians)"""
        try:
            current_positions = [sensor.getValue() for sensor in self.joint_sensors]
            motor_names = [
                "PelvR", "PelvYR", "LegUpperR", "PelvL", "PelvYL", "LegUpperL",
                "LegLowerR", "LegLowerL", "AnkleR", "FootR", "AnkleL", "FootL"
            ]
            
            print("\n=== UPRIGHT POSITION VERIFICATION (RADIANS) ===")
            all_correct = True
            tolerance = 0.01  # Tight tolerance in radians (~0.6 degrees)
            for i, (name, current, target) in enumerate(zip(motor_names, current_positions, self.initial_standing_positions)):
                error = abs(current - target)
                current_deg = np.rad2deg(current)
                target_deg = np.rad2deg(target)
                error_deg = np.rad2deg(error)
                
                status = "OK" if error < tolerance else ("WARN" if error < 0.05 else "FAIL")
                if error >= tolerance:
                    all_correct = False
                    
                print(f"  {status} {name:12}: {current:7.4f} rad ({current_deg:6.2f}°) | target: {target:7.4f} rad ({target_deg:6.2f}°) | error: {error_deg:5.2f}°")
            
            if all_correct:
                print("Robot successfully reset to upright position")
            else:
                print("Warning: Some joints not perfectly aligned - but within acceptable range")
                
        except Exception as e:
            print(f"Position verification error: {e}")
    
    def log_motor_positions(self, positions, context=""):
        """Log motor positions for debugging"""
        motor_names = [
            "PelvR", "PelvYR", "LegUpperR", "PelvL", "PelvYL", "LegUpperL",
            "LegLowerR", "LegLowerL", "AnkleR", "FootR", "AnkleL", "FootL"
        ]
        
        print(f"\n=== MOTOR POSITIONS LOG {context} ===")
        for i, (pos, name) in enumerate(zip(positions, motor_names)):
            pos_deg = np.rad2deg(pos)
            print(f"  {name:12}: {pos:8.4f} rad ({pos_deg:7.2f}°)")
        return positions  # Return unchanged
    
    
    def set_neutral_standing_position(self):
        """Set robot to neutral standing position within motor limits"""
        try:
            print("\n=== SETTING ROBOT TO NEUTRAL STANDING POSITION ===")
            
            # Define stable standing positions - closer to straight legs for stability
            # These should be very stable positions for a humanoid robot
            neutral_positions = [
                0.0,    # PelvR - no hip abduction
                0.0,    # PelvYR - no hip rotation  
                0.0,    # LegUpperR - straight hip
                0.0,    # PelvL - no hip abduction
                0.0,    # PelvYL - no hip rotation
                0.0,    # LegUpperL - straight hip
                0.1,    # LegLowerR - very slight knee flexion for stability
                0.1,    # LegLowerL - very slight knee flexion for stability
                0.0,    # AnkleR - neutral ankle
                0.0,    # FootR - neutral foot
                0.0,    # AnkleL - neutral ankle
                0.0     # FootL - neutral foot
            ]
            
            # Ensure all positions are within motor limits
            max_limit = 2.8  # Slightly under the 2.82743 limit for safety
            min_limit = -2.8
            
            safe_positions = []
            for pos in neutral_positions:
                safe_pos = max(min_limit, min(max_limit, pos))
                safe_positions.append(safe_pos)
            
            # Apply these positions to the robot
            print("Applying neutral standing positions...")
            for i, motor in enumerate(self.motor_devices):
                motor.setPosition(safe_positions[i])
                motor.setVelocity(0.0)
            
            # Let the robot settle into position
            print("Allowing robot to settle into neutral position...")
            for _ in range(20):
                self.robot.step(self.timestep)
            
            return safe_positions
            
        except Exception as e:
            print(f"Error setting neutral position: {e}")
            return [0.0] * 12  # Return zeros as fallback

    def capture_initial_positions(self):
        """Set robot to neutral position then capture those positions"""
        try:
            print("\n=== SETTING UP INITIAL STANDING POSITION ===")
            
            # First, set robot to a neutral standing position within limits
            neutral_positions = self.set_neutral_standing_position()
            
            # Wait additional time for complete settling
            for _ in range(10):
                self.robot.step(self.timestep)
                
            # Now capture the actual sensor readings after settling
            settled_positions = [sensor.getValue() for sensor in self.joint_sensors]
            self.actual_start_positions = settled_positions.copy()
            
            # Log the captured positions for debugging
            self.log_motor_positions(settled_positions, "(CAPTURED AFTER NEUTRAL POSITIONING)")
            
            # Force use of stable neutral positions instead of captured (potentially unstable) ones
            print("Using stable neutral positions instead of captured positions for reliability")
            self.initial_standing_positions = neutral_positions.copy()
            
            self.log_motor_positions(self.initial_standing_positions, "(STABLE STANDING REFERENCE)")
            
        except Exception as e:
            print(f"Error: Could not capture initial positions: {e}")
            print("Using default neutral positions for reset")
            self.initial_standing_positions = [0.0] * 12
    
    # ML sampling loop - runs in background thread
    # =============================================================================
    # MACHINE LEARNING INTEGRATION
    # =============================================================================
    
    def ml_sampling_loop(self):
        """Continuously samples actions from ML model without blocking simulation"""
        print("Starting ML sampling thread...")
        while self.running:
            try:
                # Get observation and update buffer thread-safely
                obs = self.observe()
                with self.buffer_lock:
                    if len(self.buffer) >= self.obs_num:  # Only update if buffer is full
                        self.buffer.append(obs)
                
                # Use action from external source (like evolutionary trainer) or random fallback
                with self.action_lock:
                    if hasattr(self, 'current_action') and self.current_action is not None:
                        # Use action provided by external trainer (e.g., SingleInstanceEvolution)
                        new_action = self.current_action.copy()
                        action_source = "neural_network"
                    else:
                        # Fallback to random action if no external action provided
                        new_action = self.action_space.sample()
                        self.current_action = new_action
                        action_source = "random"
                
                # Debug: occasionally show action source
                if hasattr(self, 'action_source_counter'):
                    self.action_source_counter += 1
                else:
                    self.action_source_counter = 0
                
                if self.action_source_counter % 300 == 0:  # Every 10 seconds
                    print(f"Action source: {action_source} | Action sample: {new_action[:4]}")
                    
                # ML sampling rate (30Hz independent of simulation)
                time.sleep(1/30)  # 30 FPS ML decisions
                
            except Exception as e:
                print(f"ML sampling error: {e}")
                time.sleep(0.1)  # Brief pause before retrying
                continue
        
        print("ML sampling thread ended.")
    
    def enable_continuous_learning(self, learning_rate=0.001, save_interval=1000):
        """Enable continuous learning mode for long-running training"""
        self.learning_active = True
        self.learning_rate = learning_rate
        self.save_interval = save_interval
        self.experience_buffer = []
        self.last_save_step = 0
        
        print(f"Continuous learning enabled - LR: {learning_rate}, Save every {save_interval} steps")
    
    def collect_experience(self, obs, action, reward, next_obs, done):
        """Collect experience for continuous learning"""
        if self.learning_active:
            experience = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done,
                'step': self.total_steps,
                'episode': self.episode_count
            }
            self.experience_buffer.append(experience)
            
            # Limit buffer size to prevent memory issues
            if len(self.experience_buffer) > 10000:
                self.experience_buffer.pop(0)  # Remove oldest experience
    
    def should_save_model(self):
        """Check if it's time to save the model"""
        return (self.learning_active and 
                self.total_steps - self.last_save_step >= self.save_interval)
    
    def save_progress(self):
        """Save learning progress and statistics"""
        if not self.learning_active:
            return
            
        try:
            # Save basic statistics
            stats = {
                'total_steps': self.total_steps,
                'episodes': self.episode_count,
                'runtime_hours': (time.time() - self.start_time) / 3600,
                'avg_episode_length': self.total_steps / max(1, self.episode_count)
            }
            
            print(f"Progress saved - Steps: {stats['total_steps']}, Episodes: {stats['episodes']}, Runtime: {stats['runtime_hours']:.1f}h")
            self.last_save_step = self.total_steps
            
            # TODO: Implement actual model saving when we integrate ML training
            
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    # Traditional step method for compatibility
    # =============================================================================
    # GYMNASIUM INTERFACE METHODS
    # =============================================================================
    
    def step(self, action: ActType):
        # Update action thread-safely
        with self.action_lock:
            self.current_action = action
        
        # Just return current state (simulation runs continuously)
        obs = self.observe()
        with self.buffer_lock:
            self.buffer.append(obs)
        
        reward, fallen, collision = self.rewardCalc()
        
        terminated = bool(fallen or collision)
        truncated = bool(self.cur_step >= self.max_step)
        
        with self.buffer_lock:
            obs_stacked = self._stack_obs()
        info = {}
        
        return obs_stacked, reward, terminated, truncated, info

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

    # =============================================================================
    # REWARD CALCULATION
    # =============================================================================
    
    def rewardCalc(self):
        """Compute human-like walking reward with multiple components."""
        
        # Get current state
        current_pos = np.array(self.position, dtype=np.float32)
        fallen = self.fellOver()
        collision = self.check_box_collision()
        
        # ── 1. Forward Progress Reward ─────────────────────────────────────────
        # Reward movement in positive Y direction with speed consideration
        forward_velocity = current_pos[1] - self.prev_position[1]  # Y-axis progress
        
        # Reward any forward movement to encourage initial learning
        forward_speed = forward_velocity / self.dt if self.dt > 0 else 0
        
        if forward_speed > 0:  # Any forward movement gets some reward
            if forward_speed >= self.min_walking_speed:
                # Good walking speed - full reward
                speed_factor = min(1.0, forward_speed / 0.3)  # Normalize to optimal speed
                forward_rwd = self.forward_reward * speed_factor
            else:
                # Slow but forward movement - partial reward to encourage learning
                forward_rwd = self.forward_reward * 0.2 * (forward_speed / self.min_walking_speed)
        else:
            forward_rwd = 0.0  # No reward for backward/no movement
        
        # ── 2. Stability & Posture Rewards ─────────────────────────────────────
        # Reward staying upright (height maintenance and low tilt)
        height_ratio = current_pos[2] / 0.285  # Normalized to starting height
        
        # Reward maintaining good height (human-like upright posture)
        if height_ratio >= 0.9:  # Excellent posture
            height_rwd = self.height_bonus * 1.0
        elif height_ratio >= 0.75:  # Good posture
            height_rwd = self.height_bonus * 0.5
        else:  # Poor posture
            height_rwd = 0.0
        
        # Stability from gyro (penalize excessive rotation but allow natural sway)
        gyro_values = np.array(self.gyro.getValues())
        gyro_magnitude = np.linalg.norm(gyro_values)
        # Human walking involves natural body sway, so be more tolerant
        stability_rwd = self.stability_reward * max(0, 1.0 - gyro_magnitude / 3.0)
        stability_rwd += height_rwd
        
        # ── 3. Movement Encouragement & Energy Balance ────────────────────────
        # Encourage leg movement but penalize excessive movement
        current_joints = np.array([s.getValue() for s in self.joint_sensors])
        
        if self.prev_joint_positions is not None:
            joint_velocities = np.abs(current_joints - self.prev_joint_positions)
            total_movement = np.sum(joint_velocities)
            
            # Focus on leg joint movement (indices 2, 5, 6, 7 are the main leg joints)
            leg_joint_indices = [2, 5, 6, 7]  # LegUpperR, LegUpperL, LegLowerR, LegLowerL
            leg_movement = np.sum(joint_velocities[leg_joint_indices])
            
            # Encourage moderate leg movement
            if leg_movement > 0.01:  # Reward any significant leg movement
                movement_reward = 2.0 * min(1.0, leg_movement / 0.1)  # Scale to reasonable movement
            else:
                movement_reward = 0.0
            
            # Penalize excessive total movement (energy efficiency)
            if total_movement > 0.2:
                energy_pen = self.energy_penalty * (total_movement - 0.2)
            else:
                energy_pen = 0.0
        else:
            movement_reward = 0.0
            energy_pen = 0.0
        
        self.prev_joint_positions = current_joints.copy()
        
        # ── 4. Smoothness Reward ──────────────────────────────────────────────
        # Reward smooth, controlled movements (human walking has natural acceleration patterns)
        accel_values = np.array(self.accel.getValues())
        accel_magnitude = np.linalg.norm(accel_values)
        # Normalize by gravity - humans experience 1-2g during normal walking
        normalized_accel = accel_magnitude / 9.81
        
        # Reward range that allows natural walking dynamics (0.8-1.8g is normal for walking)
        if 0.8 <= normalized_accel <= 1.8:
            smoothness_rwd = self.smoothness_reward * 1.0  # Full reward for natural range
        elif normalized_accel <= 2.5:
            smoothness_rwd = self.smoothness_reward * 0.5  # Partial reward for acceptable range
        else:
            smoothness_rwd = 0.0  # No reward for excessive acceleration
        
        # ── 5. Enhanced Gait Pattern Reward ────────────────────────────────────
        # More sophisticated gait analysis for human-like walking
        left_hip = current_joints[2]    # LegUpperL
        right_hip = current_joints[5]   # LegUpperR  
        left_knee = current_joints[6]   # LegLowerL
        right_knee = current_joints[7]  # LegLowerR
        
        # Hip alternation (primary gait indicator)
        hip_difference = abs(left_hip - right_hip)
        hip_gait = self.gait_reward * min(1.0, hip_difference / 0.3)
        
        # Knee coordination (secondary gait indicator)
        knee_coordination = abs(left_knee - right_knee)
        knee_gait = self.gait_reward * 0.5 * min(1.0, knee_coordination / 0.4)
        
        # Combine gait components
        gait_rwd = hip_gait + knee_gait
        
        # ── 6. Collision & Fall Penalties ─────────────────────────────────────
        coll_pen = self.collision_penalty if collision else 0.0
        fall_pen = self.fall_penalty if fallen else 0.0
        
        # ── 7. Combine All Components ─────────────────────────────────────────
        reward = (forward_rwd + stability_rwd + movement_reward + smoothness_rwd + 
                 gait_rwd + energy_pen + coll_pen + fall_pen)
        
        # Debug: occasionally log reward components to understand what's happening
        if hasattr(self, 'reward_debug_counter'):
            self.reward_debug_counter += 1
        else:
            self.reward_debug_counter = 0
        
        if self.reward_debug_counter % 100 == 0:  # Log every 100 reward calculations
            print(f"\n=== REWARD DEBUG (STEP {self.reward_debug_counter}) ===")
            print(f"  Forward speed: {forward_speed:.4f} m/s → reward: {forward_rwd:.3f}")
            print(f"  Height ratio: {height_ratio:.3f} → stability: {stability_rwd:.3f}")
            print(f"  Leg movement: → reward: {movement_reward:.3f}")
            print(f"  Gait pattern: {gait_rwd:.3f}")
            print(f"  Energy penalty: {energy_pen:.3f}")
            print(f"  Total reward: {reward:.3f}")
        
        # Update tracking
        self.prev_position = current_pos.copy()
        self.step_count_for_gait += 1
        
        return reward, fallen, collision
    
    def set_cube_velocity(self, i, vx, vy, wz=0.0):
        self.cube_vel[i] = np.array([vx, vy], float)
        self.cube_wz[i]  = float(wz)

    # =============================================================================
    # ACTION EXECUTION
    # =============================================================================
    
    def setAction(self, action: ActType):
        # If actions are in [-1, 1] range (from neural network), scale to joint ranges
        if np.all(np.abs(action) <= 1.0):
            action = self._soft_min + (action + 1.0) * 0.5 * (self._soft_max - self._soft_min)

        action = np.clip(action, self._soft_min, self._soft_max)

        # low-pass filter to smooth control
        if self._prev_action is None:
            smoothed = action
        else:
            smoothed = 0.8*self._prev_action + 0.2*action
        self._prev_action = smoothed

        for mot, tgt in zip(self.motor_devices, smoothed):
            mot.setPosition(float(tgt))
        
        # Debug: occasionally log actions to see if they're reasonable
        if hasattr(self, 'action_debug_counter'):
            self.action_debug_counter += 1
        else:
            self.action_debug_counter = 0
        
        if self.action_debug_counter % 200 == 0:  # Log every 200 actions (every ~4 seconds at 50Hz)
            motor_names = ["PelvR", "PelvYR", "LegUpperR", "PelvL", "PelvYL", "LegUpperL", 
                          "LegLowerR", "LegLowerL", "AnkleR", "FootR", "AnkleL", "FootL"]
            print(f"\n=== ACTION DEBUG (STEP {self.action_debug_counter}) ===")
            for i, (name, val) in enumerate(zip(motor_names, action)):
                print(f"  {name:12}: {val:7.4f} rad ({np.rad2deg(val):6.2f}°)")
        
        for mot, tgt in zip(self.motor_devices, action):
            mot.setPosition(tgt)        # Webots will move at the per-joint vmax
    
    def updateSimulation(self):
        # Update robot position from GPS (most important)
        self.position = self.gps.getValues()
        
        # Update orientation (lightweight calculation)
        omega = np.array(self.gyro.getValues())
        self.orientation += omega * self.dt

    # =============================================================================
    # SENSOR DATA PROCESSING
    # =============================================================================
    
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
        # This should only be called within buffer_lock
        buffer_copy = list(self.buffer)  # Make a safe copy
        if len(buffer_copy) == 0:
            return None
        imgs   = np.stack([o[0] for o in buffer_copy], axis=0)  # (N,H,W,1)
        joints = np.stack([o[1] for o in buffer_copy], axis=0)  # (N,12)
        gyros  = np.stack([o[2] for o in buffer_copy], axis=0)  # (N,3)
        accels = np.stack([o[3] for o in buffer_copy], axis=0)  # (N,3)
        gpss   = np.stack([o[4] for o in buffer_copy], axis=0)  # (N,3)
        return (imgs, joints, gyros, accels, gpss)

    def observe(self):
        image  = self.convertImage(self.cam.getImage())
        joints = np.array([s.getValue() for s in self.joint_sensors], dtype=np.float64)
        gyro   = np.array(self.gyro.getValues(),  dtype=np.float64)
        accel  = np.array(self.accel.getValues(), dtype=np.float64)
        gps    = np.array(self.gps.getValues(),   dtype=np.float64)
        return (image, joints, gyro, accel, gps)

    # =============================================================================
    # PHYSICS AND COLLISION DETECTION
    # =============================================================================
    
    def fellOver(self) -> bool:
        z_pos = self.position[2]                 # self.position is GPS [x, y, z]
        
        # Height check - only trigger when really down
        if z_pos <= self.height_threshold:
            print(f"FALL DETECTED - Height: z = {z_pos:.3f} m (threshold: {self.height_threshold:.3f} m)")
            return True
 
        # Acceleration check - only trigger on severe impacts
        accel_values = np.array(self.accel.getValues())   # m/s²
        accel_magnitude = np.linalg.norm(accel_values)
        if accel_magnitude >= self.accel_threshold:
            print(f"FALL DETECTED - Acceleration: |a| = {accel_magnitude:.2f} m/s² (threshold: {self.accel_threshold:.2f} m/s²)")
            return True

        # Debug: occasionally show current values to verify thresholds
        if hasattr(self, 'fall_debug_counter'):
            self.fall_debug_counter += 1
        else:
            self.fall_debug_counter = 0
        
        if self.fall_debug_counter % 500 == 0:  # Every 10 seconds at 50Hz
            print(f"Robot stable - Height: {z_pos:.3f} m, Accel: {accel_magnitude:.2f} m/s²")

        return False

    def getMotorPos(self):
        # Take in data and convert to degrees
        # TODO might read from the actual sensors
        motors = [np.rad2deg(self.motor_devices[i].getTargetPosition()) for i in range(self.motor_num)]

        # Convert all angles to positive
        for i in range(len(motors)):
            if motors[i] < 0:
                motors[i] += 360.0

        return motors

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
    
    # =============================================================================
    # ASYNCHRONOUS SIMULATION CONTROL
    # =============================================================================
    
    def start_async_simulation(self):
        """Start optimized simulation with ML thread only"""
        if self.ml_thread is None or not self.ml_thread.is_alive():
            self.running = True
            
            # Start ML sampling thread
            self.ml_thread = threading.Thread(target=self.ml_sampling_loop, daemon=True)
            self.ml_thread.start()
        
        # Run continuous simulation in main thread (optimized for speed)
        self.run_continuous_simulation()
    
    def stop_async_simulation(self):
        """Stop ML thread"""
        self.running = False
        if self.ml_thread and self.ml_thread.is_alive():
            self.ml_thread.join(timeout=1.0)

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

