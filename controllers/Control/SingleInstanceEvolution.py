# =============================================================================
# SINGLE INSTANCE EVOLUTIONARY TRAINING MODULE
# =============================================================================
# Evolutionary algorithm for training humanoid robot walking using a single
# Webots simulation instance with multiple neural network models
# =============================================================================
import time
import os
import json
import numpy as np
import tensorflow as tf
import signal
import sys
import random

# =============================================================================
# NEURAL NETWORK MODEL DEFINITION
# =============================================================================

class SimpleRobotPolicy(tf.keras.Model):
    """Simplified robot policy for evolutionary training"""
    
    def __init__(self, action_dim=12):
        super().__init__()
        
        self.vision_conv1 = tf.keras.layers.Conv2D(8, 32, strides=16, activation='relu')
        self.vision_conv2 = tf.keras.layers.Conv2D(16, 16, strides=8, activation='relu') 
        self.vision_flatten = tf.keras.layers.Flatten()
        self.vision_fc = tf.keras.layers.Dense(64, activation='relu')
        
        self.proprio_fc = tf.keras.layers.Dense(32, activation='relu')
        
        self.action_head = tf.keras.layers.Dense(action_dim, activation='tanh')
        self.value_head = tf.keras.layers.Dense(1)
    
    def call(self, observations):
        images, joints, gyros, accels, gps = observations
        
        latest_image = images[:, -1, :, :, :]
        latest_joints = joints[:, -1, :]
        latest_gyros = gyros[:, -1, :]
        latest_accels = accels[:, -1, :]
        latest_gps = gps[:, -1, :]
        
        x_vision = self.vision_conv1(latest_image)
        x_vision = self.vision_conv2(x_vision)
        x_vision = self.vision_flatten(x_vision)
        x_vision = self.vision_fc(x_vision)
        
        proprio_input = tf.concat([latest_joints, latest_gyros, latest_accels, latest_gps], axis=-1)
        x_proprio = self.proprio_fc(proprio_input)
        
        combined = tf.concat([x_vision, x_proprio], axis=-1)
        
        actions = self.action_head(combined)
        values = self.value_head(combined)
        
        return actions, values

# =============================================================================
# POPULATION MANAGEMENT
# =============================================================================

class ModelPopulation:
    """Manages a population of models for evolution"""
    
    def __init__(self, population_size=4, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.models = []
        self.optimizers = []
        self.fitness_scores = []
        self.generations = 0
        
        for i in range(population_size):
            model = SimpleRobotPolicy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
            dummy_obs = [
                tf.zeros((1, 4, 512, 512, 1)),
                tf.zeros((1, 4, 12)),
                tf.zeros((1, 4, 3)),
                tf.zeros((1, 4, 3)),
                tf.zeros((1, 4, 3))
            ]
            model(dummy_obs)
            
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.fitness_scores.append(0.0)
        
        print(f"Population initialized: {population_size} models with {self.models[0].count_params():,} parameters each")
    
    def get_current_model(self, model_index):
        """Get specific model from population"""
        return self.models[model_index], self.optimizers[model_index]
    
    def update_fitness(self, model_index, fitness):
        """Update fitness score for a model"""
        self.fitness_scores[model_index] = fitness
    
    def evolve_population(self):
        """Evolve the population - keep best, mutate others"""
        if all(score == 0.0 for score in self.fitness_scores):
            return
        
        best_idx = np.argmax(self.fitness_scores)
        best_fitness = self.fitness_scores[best_idx]
        best_model = self.models[best_idx]
        
        print(f"Generation {self.generations}: Best model {best_idx} with fitness {best_fitness:.2f}")
        
        self._save_best_model(best_idx, best_fitness)
        
        new_models = []
        new_optimizers = []
        
        new_models.append(self._copy_model(best_model))
        new_optimizers.append(tf.keras.optimizers.Adam(learning_rate=0.001))
        
        for i in range(1, self.population_size):
            mutated_model = self._copy_model(best_model)
            self._mutate_model(mutated_model)
            new_models.append(mutated_model)
            new_optimizers.append(tf.keras.optimizers.Adam(learning_rate=0.001))
        
        self.models = new_models
        self.optimizers = new_optimizers
        self.fitness_scores = [0.0] * self.population_size
        self.generations += 1
        
        print(f"Population evolved to generation {self.generations}")
    
    def _copy_model(self, source_model):
        """Create a copy of a model"""
        new_model = SimpleRobotPolicy()
        
        dummy_obs = [
            tf.zeros((1, 4, 512, 512, 1)),
            tf.zeros((1, 4, 12)),
            tf.zeros((1, 4, 3)),
            tf.zeros((1, 4, 3)),
            tf.zeros((1, 4, 3))
        ]
        new_model(dummy_obs)
        
        new_model.set_weights(source_model.get_weights())
        return new_model
    
    def _mutate_model(self, model):
        """Apply random mutations to model weights"""
        for layer in model.trainable_variables:
            if random.random() < self.mutation_rate:
                noise = tf.random.normal(tf.shape(layer), stddev=0.1)
                layer.assign_add(noise * 0.1)
    
    def _save_best_model(self, model_index, fitness):
        """Save the best performing model"""
        try:
            models_dir = "champion_models"
            os.makedirs(models_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = f"evolution_gen_{self.generations}_model_{model_index}_fitness_{fitness:.1f}_{timestamp}"
            model_path = os.path.join(models_dir, f"{model_name}.weights.h5")  # Fixed extension
            
            self.models[model_index].save_weights(model_path)
            
            metadata = {
                'generation': int(self.generations),
                'model_index': int(model_index),
                'fitness': float(fitness),
                'timestamp': str(timestamp),
                'population_size': int(self.population_size),
                'mutation_rate': float(self.mutation_rate)
            }
            
            with open(os.path.join(models_dir, f"{model_name}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved best model: {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")

# =============================================================================
# EVOLUTIONARY TRAINING MANAGER
# =============================================================================

class SingleInstanceEvolutionTrainer:
    """Evolution trainer using single Webots instance"""
    
    def __init__(self, population_size=4, episodes_per_model=5):
        self.population_size = population_size
        self.episodes_per_model = episodes_per_model
        
        self.population = ModelPopulation(population_size)
        self.env = None
        self.running = True
        
        self.total_episodes = 0
        self.start_time = time.time()
    
    def start_training(self):
        """Start evolutionary training with proper async setup"""
        print("Starting single-instance evolutionary training...")
        
        from RobotControl import CustomEnv
        self.env = CustomEnv(instance_id="evolution_trainer")
        self.env.enable_continuous_learning()
        
        obs, info = self.env.reset()
        print(f"Environment initialized and reset complete")
        print(f"Observation shape: images={obs[0].shape}, joints={obs[1].shape}")
        
        import threading
        self.sim_thread = threading.Thread(target=self._run_simulation_background, daemon=True)
        self.sim_thread.start()
        
        print("Background simulation started - robot will now move")
        print("Environment setup complete - ready for model evaluation")
        
        generation = 0
        
        try:
            while self.running:
                print(f"\n=== GENERATION {generation} ===")
                
                for model_idx in range(self.population_size):
                    if not self.running:
                        break
                    
                    print(f"\nTesting model {model_idx + 1}/{self.population_size}")
                    fitness = self._evaluate_model(model_idx)
                    self.population.update_fitness(model_idx, fitness)
                    
                    print(f"Model {model_idx} fitness: {fitness:.2f}")
                
                if self.running:
                    self.population.evolve_population()
                    generation += 1
                
                hours = (time.time() - self.start_time) / 3600
                print(f"\nTraining status: {self.total_episodes} episodes in {hours:.1f} hours")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.env:
                self.env.close()
    
    def _evaluate_model(self, model_index):
        """Evaluate a specific model by controlling the background simulation"""
        model, optimizer = self.population.get_current_model(model_index)
        
        episode_rewards = []
        experiences = []
        
        for episode in range(self.episodes_per_model):
            if not self.running:
                break
            
            print(f"    Starting episode {episode + 1} with model {model_index}")
            self.env.reset_simulation_state()  # Reset the robot state
            
            total_reward = 0
            step_count = 0
            last_position = None
            episode_start_time = time.time()
            
            while self.running and step_count < 300:  # Shorter episodes for faster evaluation
                try:
                    current_obs = self.env.observe()
                    
                    if len(self.env.buffer) >= self.env.obs_num:
                        try:
                            stacked_obs = self.env._stack_obs()
                            if stacked_obs is not None:
                                obs_tensor = [
                                    tf.expand_dims(tf.constant(stacked_obs[0], dtype=tf.float32), 0),  # images
                                    tf.expand_dims(tf.constant(stacked_obs[1], dtype=tf.float32), 0),  # joints  
                                    tf.expand_dims(tf.constant(stacked_obs[2], dtype=tf.float32), 0),  # gyros
                                    tf.expand_dims(tf.constant(stacked_obs[3], dtype=tf.float32), 0),  # accels
                                    tf.expand_dims(tf.constant(stacked_obs[4], dtype=tf.float32), 0)   # gps
                                ]
                            else:
                                action = self.env.action_space.sample()
                                continue
                            
                            actions, values = model(obs_tensor)
                            action = actions.numpy().flatten()
                            
                            noise = np.random.normal(0, 0.15, size=action.shape)
                            action = np.clip(action + noise, -1.0, 1.0)
                        except Exception as e:
                            print(f"Model inference error: {e}")
                            action = self.env.action_space.sample()
                    else:
                        action = self.env.action_space.sample()
                    
                    with self.env.action_lock:
                        self.env.current_action = action
                    
                    time.sleep(0.02)
                    
                    reward, fallen, collision = self.env.rewardCalc()
                    total_reward += reward
                    
                    with self.env.buffer_lock:
                        stacked_obs = self.env._stack_obs()
                    
                    if stacked_obs is not None:
                        experiences.append({
                            'obs': stacked_obs,
                            'action': action,
                            'reward': reward,
                            'done': fallen or collision
                        })
                    
                    step_count += 1
                    
                    if fallen or collision:
                        print(f"      Episode ended: fallen={fallen}, collision={collision} at step {step_count}")
                        break
                    
                    if step_count % 50 == 0:
                        print(f"      Step {step_count}: reward={total_reward:.1f}, height={self.env.position[2]:.3f}")
                
                except Exception as e:
                    print(f"Evaluation error: {e}")
                    break
            
            episode_rewards.append(total_reward)
            self.total_episodes += 1
            
            duration = time.time() - episode_start_time
            print(f"    Episode {episode + 1}: reward={total_reward:.1f}, steps={step_count}, duration={duration:.1f}s")
        
        if len(experiences) > 20:
            self._simple_learning(model, optimizer, experiences[-100:])
        
        avg_fitness = np.mean(episode_rewards) if episode_rewards else 0.0
        return avg_fitness
    
    def _simple_learning(self, model, optimizer, experiences):
        """Simple learning update"""
        try:
            if len(experiences) < 10:
                return
            
            batch_obs = []
            batch_actions = []
            batch_rewards = []
            
            for exp in experiences:
                batch_obs.append(exp['obs'])
                batch_actions.append(exp['action'])
                batch_rewards.append(exp['reward'])
            
            batch_rewards = tf.constant(batch_rewards, dtype=tf.float32)
            batch_actions = tf.constant(batch_actions, dtype=tf.float32)
            
            reward_mean = tf.reduce_mean(batch_rewards)
            reward_std = tf.math.reduce_std(batch_rewards) + 1e-8
            normalized_rewards = (batch_rewards - reward_mean) / reward_std
            
            images = tf.stack([tf.constant(obs[0], dtype=tf.float32) for obs in batch_obs])
            joints = tf.stack([tf.constant(obs[1], dtype=tf.float32) for obs in batch_obs])
            gyros = tf.stack([tf.constant(obs[2], dtype=tf.float32) for obs in batch_obs])
            accels = tf.stack([tf.constant(obs[3], dtype=tf.float32) for obs in batch_obs])
            gps = tf.stack([tf.constant(obs[4], dtype=tf.float32) for obs in batch_obs])
            
            obs_batch = [images, joints, gyros, accels, gps]
            
            with tf.GradientTape() as tape:
                actions_pred, values_pred = model(obs_batch)
                
                policy_loss = tf.reduce_mean(
                    tf.square(batch_actions - actions_pred) * tf.expand_dims(normalized_rewards, -1)
                )
                
  
                value_loss = tf.reduce_mean(tf.square(values_pred - tf.expand_dims(batch_rewards, -1)))
                
                total_loss = policy_loss + 0.1 * value_loss
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
        except Exception as e:
            print(f"Learning error: {e}")
    
    def shutdown(self):
        """Shutdown trainer"""
        print("Shutting down evolutionary trainer...")
        self.running = False
        if self.env:
            self.env.close()
    
    def _run_simulation_background(self):
        """Run the continuous simulation in background - this makes the robot move"""
        try:
            # This runs the actual robot movement loop
            self.env.run_continuous_simulation()
        except Exception as e:
            print(f"Background simulation error: {e}")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def start_single_instance_evolution():
    """Start single instance evolutionary training"""
    
    trainer = SingleInstanceEvolutionTrainer(population_size=4, episodes_per_model=3)
    
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        trainer.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        trainer.start_training()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        trainer.shutdown()

if __name__ == "__main__":
    start_single_instance_evolution()