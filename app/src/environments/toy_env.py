"""Toy scheduling environment for initial learning."""

# Import statements - These are the tools we need
import numpy as np  # NumPy: For fast array operations (like managing job times)
from typing import Dict, Tuple, Optional, Any  # Type hints: Makes code clearer (like labels on boxes)
import gymnasium as gym  # Gymnasium: The RL environment framework (like a game engine)
from gymnasium import spaces  # Spaces: Defines what actions/observations look like

# Import our parent class (like inheriting traits from a parent)
from .base_env import BaseSchedulingEnv


class ToySchedulingEnv(BaseSchedulingEnv):
    """Simple scheduling environment with 2 machines and 5 jobs.
    
    Think of this like a small factory with 2 workers (machines) and 5 tasks (jobs).
    The AI needs to learn which worker should do which task.
    
    State Space (What the AI "sees"):
        - Machine loads (2 values): How busy is each worker? [0.3, 0.7] = Machine 0 is 30% loaded
        - Job scheduled flags (5 values): Which jobs are done? [1,1,0,0,0] = First 2 jobs done
        - Job processing times (5 values): How long each job takes [2,3,1,4,2] hours
        - Current time (1 value): What % of time is used? 0.2 = 20% of episode time used
        Total: 13 numbers, all scaled between 0 and 1 (easier for AI to learn)
        
    Action Space (What the AI can "do"):
        - 0-4: Schedule job 0, 1, 2, 3, or 4
        - 5: Wait (do nothing this turn)
        Like having 6 buttons the AI can press
        
    Rewards (How we "score" the AI):
        - Job completion: +10 points (like collecting a coin)
        - Efficient scheduling: +5 bonus (like a combo bonus)
        - Invalid action: -20 points (like hitting a wall)
        - Time penalty: -0.1 per step (like losing points every second)
        - All jobs done: +50 bonus (like finishing the level)
    """
    
    def __init__(self, 
                 n_machines: int = 2,      # How many workers/machines we have
                 n_jobs: int = 5,          # How many tasks to complete
                 max_episode_steps: int = 50,  # Max actions before timeout (like a timer)
                 min_job_time: int = 1,    # Shortest job duration (1 time unit)
                 max_job_time: int = 5,    # Longest job duration (5 time units)
                 seed: Optional[int] = None):  # Random seed (None = different each time)
        """Initialize toy scheduling environment.
        
        This is like setting up a new game level with specific rules.
        
        Args:
            n_machines: Number of machines (workers) available
            n_jobs: Number of jobs (tasks) to schedule
            max_episode_steps: Maximum actions allowed (like a turn limit)
            min_job_time: Minimum time any job can take
            max_job_time: Maximum time any job can take
            seed: Random seed for reproducibility (like a level code)
        """
        # Call parent class constructor (inherit basic scheduling features)
        super().__init__(n_machines, n_jobs, seed)
        
        # Store our specific settings
        self.max_episode_steps = max_episode_steps  # Remember our time limit
        self.min_job_time = min_job_time           # Remember min job duration
        self.max_job_time = max_job_time           # Remember max job duration
        
        # Define observation space (what the AI sees)
        # This is like defining what information appears on the game screen
        # State vector has: 2 machine loads + 5 job flags + 5 job times + 1 current time = 13 values
        self.observation_space = spaces.Box(
            low=0.0,      # Minimum value for any observation (all normalized)
            high=1.0,     # Maximum value for any observation (all normalized)
            shape=(n_machines + n_jobs * 2 + 1,),  # Shape: (2 + 5*2 + 1) = 13 dimensions
            dtype=np.float32  # Use 32-bit floats for efficiency
        )
        
        # Define action space (what buttons the AI can press)
        # Discrete = distinct choices, like a game controller with 6 buttons
        self.action_space = spaces.Discrete(n_jobs + 1)  # 5 jobs + 1 wait = 6 actions
        
        # Initialize state variables (these will store the current game state)
        self.machine_loads = None   # How long each machine will be busy
        self.job_scheduled = None   # Which jobs have been scheduled (True/False array)
        self.job_times = None       # How long each job takes
        self.current_step = None    # Current timestep (like a turn counter)
        
    def _reset_impl(self, options: Optional[Dict] = None) -> np.ndarray:
        """Reset environment to initial state.
        
        This is like starting a new game - everything goes back to the beginning.
        Called at the start of each episode (each training attempt).
        
        Returns:
            Initial observation (what the AI sees at the start)
        """
        # Reset machine loads to 0 (all machines are free)
        # np.zeros creates array of zeros: [0.0, 0.0] for 2 machines
        self.machine_loads = np.zeros(self.n_machines, dtype=np.float32)
        
        # Reset job scheduled flags (no jobs are scheduled yet)
        # Creates boolean array: [False, False, False, False, False] for 5 jobs
        self.job_scheduled = np.zeros(self.n_jobs, dtype=bool)
        
        # Generate random job processing times for this episode
        # This makes each episode different (like random enemy positions)
        if self.np_random is not None:  # If we have a seeded random generator
            # Generate random integers between min and max (inclusive)
            # E.g., [3, 1, 4, 2, 5] if min=1, max=5
            self.job_times = self.np_random.integers(
                self.min_job_time,      # Minimum value (inclusive)
                self.max_job_time + 1,  # Maximum value (+1 because exclusive)
                size=self.n_jobs        # How many random numbers
            ).astype(np.float32)        # Convert to float for calculations
        else:  # Fallback if no seeded generator (shouldn't happen normally)
            self.job_times = np.random.randint(
                self.min_job_time,
                self.max_job_time + 1,
                size=self.n_jobs
            ).astype(np.float32)
        
        # Reset step counter to 0 (start of episode)
        self.current_step = 0
        
        # Return the initial observation (what the AI sees)
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (game state) for the AI.
        
        This packages all information the AI needs into a single array.
        Like taking a screenshot of the game state.
        
        Returns:
            Observation vector with all state information normalized to [0, 1]
        """
        # Normalize machine loads (convert to 0-1 range)
        # If all jobs went to one machine, that would be the max load
        max_load = np.sum(self.job_times)  # Total time if one machine did everything
        # Divide current loads by max to get percentage (0.5 = 50% of max load)
        normalized_loads = self.machine_loads / max_load if max_load > 0 else self.machine_loads
        
        # Convert job scheduled flags to floats (False=0.0, True=1.0)
        # This makes it consistent with other float values
        job_flags = self.job_scheduled.astype(np.float32)
        
        # Normalize job times (convert to 0-1 range)
        # If max_job_time=5, then job_time=3 becomes 3/5=0.6
        normalized_times = self.job_times / self.max_job_time
        
        # Normalize current time (what fraction of episode is complete)
        # If we're at step 10 of 50, this is 10/50 = 0.2 (20% done)
        normalized_time = self.current_step / self.max_episode_steps
        
        # Combine all information into one array
        # Like packing all game info into one status bar
        obs = np.concatenate([
            normalized_loads,     # [0.3, 0.7] - Machine loads
            job_flags,           # [1, 1, 0, 0, 0] - Which jobs are done
            normalized_times,    # [0.6, 0.2, 0.8, 0.4, 1.0] - Job durations
            [normalized_time]    # [0.2] - Current time (in brackets to make it an array)
        ])
        
        # Return as float32 (standard for neural networks)
        return obs.astype(np.float32)
    
    def _step_impl(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one action in the environment.
        
        This is the core game loop - what happens when the AI presses a button.
        Like Mario jumping or a racing game turning.
        
        Args:
            action: Which button the AI pressed (0-5)
            
        Returns:
            Tuple of:
            - observation: New game state after the action
            - reward: Points earned/lost from this action
            - terminated: Did we finish all jobs? (victory!)
            - truncated: Did we run out of time? (game over!)
            - info: Extra information for debugging
        """
        # Initialize reward for this step
        reward = 0.0
        info = {}  # Dictionary to store extra information
        
        # Time penalty - lose 0.1 points each step (encourages speed)
        # Like losing points every second in a timed game
        reward -= 0.1
        self.current_step += 1  # Increment turn counter
        
        # Check if action is scheduling a job (not waiting)
        if action < self.n_jobs:  # Actions 0-4 are job scheduling
            job_id = action  # Which job to schedule
            
            # Check if this job was already scheduled
            if self.job_scheduled[job_id]:
                # Invalid action! Like trying to collect a coin twice
                reward -= 20  # Big penalty to teach AI this is wrong
                info['invalid_action'] = True  # Record what happened
            else:
                # Valid action - schedule the job
                
                # Find the least busy machine (load balancing strategy)
                # np.argmin returns the index of the smallest value
                machine_id = np.argmin(self.machine_loads)
                
                # Assign job to machine
                # Add job's duration to that machine's total work time
                self.machine_loads[machine_id] += self.job_times[job_id]
                self.job_scheduled[job_id] = True  # Mark job as done
                
                # Give base reward for completing a job
                reward += 10  # Like collecting a coin
                
                # Calculate bonus for good load balancing
                # Variance measures how "spread out" the loads are
                load_variance = np.var(self.machine_loads)
                # Maximum variance occurs when all jobs go to one machine
                max_variance = (np.sum(self.job_times) / 2) ** 2
                # Convert variance to bonus: low variance = high bonus
                balance_bonus = 5 * (1 - load_variance / max_variance) if max_variance > 0 else 5
                reward += balance_bonus
                
                # Store information about what happened
                info['scheduled_job'] = job_id
                info['on_machine'] = machine_id
                info['load_balance_bonus'] = balance_bonus
        
        # If action == n_jobs (5), it's a wait action - do nothing
        # Sometimes waiting might be strategic (but usually not in this simple env)
        
        # Check if all jobs are now scheduled (victory condition)
        all_scheduled = np.all(self.job_scheduled)  # True if all elements are True
        if all_scheduled:
            reward += 50  # Big bonus for completing all jobs! Like finishing a level
            info['all_jobs_scheduled'] = True
        
        # Check end conditions
        terminated = all_scheduled  # Episode ends successfully if all jobs done
        truncated = self.current_step >= self.max_episode_steps  # Time's up!
        
        # If episode is ending, calculate final statistics
        if terminated or truncated:
            # Makespan = total time to complete all jobs (max machine load)
            makespan = np.max(self.machine_loads)
            # Utilization = average load / max load (how well we used all machines)
            utilization = np.mean(self.machine_loads) / makespan if makespan > 0 else 0
            # Store these metrics
            info['final_makespan'] = float(makespan)
            info['final_utilization'] = float(utilization)
            info['jobs_scheduled'] = int(np.sum(self.job_scheduled))  # Count completed jobs
        
        # Get new observation after the action
        obs = self._get_observation()
        
        # Return everything the RL algorithm needs
        return obs, reward, terminated, truncated, info
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions.
        
        This tells the AI which buttons actually work right now.
        Like graying out unavailable menu options.
        
        Returns:
            Boolean array where True = valid action, False = invalid
        """
        # Create array of all True values (assume all actions valid)
        mask = np.ones(self.action_space.n, dtype=bool)  # [True, True, True, True, True, True]
        
        # Mark scheduled jobs as invalid (can't schedule twice)
        # ~ is the NOT operator, flips True/False
        mask[:self.n_jobs] = ~self.job_scheduled  # If job 0,1 done: [False, False, True, True, True, True]
        
        # Wait action is always valid
        mask[-1] = True  # Last action (wait) is always available
        
        return mask
    
    def _calculate_reward(self, action: int, valid_action: bool) -> float:
        """Calculate reward (used by base class if needed).
        
        We implement rewards directly in _step_impl for this environment,
        so this is just a placeholder to satisfy the base class interface.
        """
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is complete.
        
        Episode ends when:
        1. All jobs are scheduled (success!), OR
        2. We've used too many steps (timeout!)
        
        Returns:
            True if episode should end, False otherwise
        """
        # Check both victory and timeout conditions
        return np.all(self.job_scheduled) or self.current_step >= self.max_episode_steps
    
    def render(self, mode: str = 'human'):
        """Render (display) current environment state.
        
        This is like showing the game screen to a human.
        Useful for debugging and understanding what's happening.
        
        Args:
            mode: How to display ('human' = text output)
        """
        if mode == 'human':
            # Print current state in readable format
            print(f"\n=== Step {self.current_step}/{self.max_episode_steps} ===")
            print(f"Machine loads: {self.machine_loads}")  # [3.0, 5.0] = times
            print(f"Jobs scheduled: {self.job_scheduled}")  # [True, True, False, False, False]
            print(f"Job times: {self.job_times}")          # [2, 3, 1, 4, 5] = durations
            print(f"Scheduled: {np.sum(self.job_scheduled)}/{self.n_jobs} jobs")  # 2/5 jobs
            
            # If any work has been assigned, show efficiency metrics
            if np.any(self.machine_loads > 0):
                makespan = np.max(self.machine_loads)  # Time to finish everything
                utilization = np.mean(self.machine_loads) / makespan  # How balanced
                print(f"Current makespan: {makespan:.1f}")        # Total time
                print(f"Machine utilization: {utilization:.2%}")  # Balance percentage