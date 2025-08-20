
import signal, sys, time
import gymnasium as gym
from  RobotControl import CustomEnv          

signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

def main():
    env = CustomEnv()
    obs, info = env.reset()

    episode_reward = 0.0                         
    for t in range(1_000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            print(f"Episode finished after {t} steps ; reward={episode_reward:.3f}")
            obs, info = env.reset()
            episode_reward = 0.0                

        time.sleep(0.01)         

    env.close()



if __name__ == "__main__":
    main()
