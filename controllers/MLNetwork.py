from abc import ABC
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers
import gym


# This is the model class that inherits from model
class model(tf.keras.Model, ABC):
    # This is the constructor for the inherited model class. Defines the network
    def __init__(self):
        super().__init__()
        # This model architecture is up for debate
        self.i1 = layers.Input(shape=(512, 512, 4))
        self.i2 = layers.Input(shape=(18, 4))  # TODO Shape has to include other sensors
        self.r = layers.Reshape((72,))
        self.c1 = layers.Conv2D(32, 8, strides=4, activation="relu")
        self.c2 = layers.Conv2D(64, 4, strides=2, activation="relu")
        self.c3 = layers.Conv2D(64, 3, strides=1, activation="relu")
        self.f = layers.Flatten()
        self.d = layers.Dense(512, activation="relu")
        self.out = layers.Dense(12, activation="sigmoid")

    # This is called and has the two data streams separated
    def call(self, inputs, training=None, mask=None):
        # Unpacking the camera and motors
        i1 = self.i1(inputs[0])
        i2 = self.i2(inputs[1])
        r = self.r(i2)
        x = layers.concatenate(i1, r)

        # Convolution and dense layers
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.f(x)
        x = self.d(x)
        x = self.out(x)
        return x  # This returns a tensor of probabilities


class agent:
    # This is the constructor for the agent
    def __init__(self):
        self.model = model()
        self.learning_rate = 0.00025
        self.clipnorm = None
        self.gamma = 1.0
        self.num_motors = 12
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.clipnorm)

    # This takes an array of motor movements based on the state
    def act(self, state):
        camera, motors = state
        prob = self.model([np.array([camera]), np.array([motors])])
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample(self.num_motors)
        return action.numpy()

    # This is the new loss function. It is negative log loss
    def log_loss(self, prob, action, reward):
        dist = tfp.distributions.Categorical(prob=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -tf.reduce_mean(log_prob * reward)
        return loss

    # This function will use backpropagation to update the network
    def train(self, states, rewards, actions):
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        # Calculating the old rewards
        for r in rewards:
            sum_reward = r + self.gamma * sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        # Training step with backpropagation
        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as tape:
                camera, motors = state
                p = self.model([np.array([camera]), np.array([motors])], training=True)
                loss = self.log_loss(p, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))


def main():
    env_name = 'WalkingEnv-v0'
    gym.envs.register(id=env_name, entry_point='Control.RobotControl:CustomEnv')
    env = gym.make(env_name)

    robot = agent()
    training_steps = 500  # Might look at changing this value

    for step in range(training_steps):
        done = False
        state = env.reset()  # Return the correct values here
        total_reward = 0
        rewards = []
        states = []
        actions = []

        # Looping until something breaks
        while not done:
            action = robot.act(state)
            next_state, reward, done = env.step(action)  # Make sure this returns correctly
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            total_reward += reward

            if done:
                robot.train(states, rewards, actions)
                print("total reward after {} steps is {}".format(step, total_reward))


if __name__ == '__main__':
    main()
