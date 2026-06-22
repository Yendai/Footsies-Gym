from collections import deque
import gymnasium as gym
from ..envs.footsies import FootsiesEnv


class DynamicReactionTimeWrapper(gym.Wrapper):
    """
    Wrapper that dynamically adjusts frame delay based on agent actions and combo stance.
    
    Simulates human-like reaction time variation where standing still and focusing
    can slightly decrease average reaction time, while active actions increase it.
    In combo stance, reaction time is overall better (lower delay), but the tradeoff comes
    from the game mechanics (e.g., specials being minus on block) rather than increased delay.
    
    Parameters
    ----------
    env : gym.Env
        The FOOTSIES environment to wrap. Should have a `delayed_frame_queue` attribute and combo tracking.
    base_delay : int
        The base frame delay when the agent is performing actions in normal stance. Default: 5
    combo_base_delay : int
        The base frame delay in combo stance (should be lower for faster reaction). Default: 2
    min_delay : int
        Minimum possible frame delay. Default: 0
    max_delay : int
        Maximum possible frame delay. Default: 10
    delay_reduction_rate : float
        How much to reduce delay per frame while standing still in normal stance. Default: 0.1
    """

    def __init__(
        self, 
        env, 
        base_delay=5, 
        combo_base_delay=2,
        min_delay=0, 
        max_delay=10, 
        delay_reduction_rate=0.1,
    ):
        super().__init__(env)
        
        if not hasattr(env, 'delayed_frame_queue'):
            raise ValueError("DynamicReactionTimeWrapper requires an environment with a 'delayed_frame_queue' attribute")
        if not hasattr(env, 'combo_stance_active') or not hasattr(env, 'combo_armed'):
            raise ValueError("DynamicReactionTimeWrapper requires combo stance and arming state tracking")
        
        self.base_delay = base_delay
        self.combo_base_delay = combo_base_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay_reduction_rate = delay_reduction_rate
        
        self.current_delay = float(base_delay)
        self.frames_standing = 0

    def step(self, action):
        """
        Step the environment, adjusting frame delay based on the action taken and combo stance.
        
        Normal stance:
        - Standing still: delay gradually decreases (faster reaction)
        - Any action: delay resets to base_delay
        
        Combo stance:
        - Uses combo_base_delay as the baseline (generally faster)
        - Standing still: further reduces delay
        - Movement: maintains the combo_base_delay (no additional penalty)
        - The counterplay comes from game mechanics (e.g., minus on block) rather than artificial delay
        """
        # Extract movement information from action
        # Action is now (left, right, attack, combo_toggle) or handled by discretizer as int
        if isinstance(action, int):
            # Discrete action format from discretizer
            left = (action & 1) != 0
            right = (action & 2) != 0
            attack = (action & 4) != 0
            combo_toggle = (action & 8) != 0
        else:
            left, right, attack = action[:3]
            combo_toggle = action[3] if len(action) > 3 else False
        
        combo_window_active = self.env.combo_armed or (self.env.combo_stance_active and attack)
        is_moving = (left or right)
        
        if combo_window_active:
            effective_base = self.combo_base_delay
            
            if not is_moving and not attack:
                self.frames_standing += 1
                self.current_delay = max(
                    self.min_delay,
                    effective_base - (self.frames_standing * self.delay_reduction_rate)
                )
            else:
                self.frames_standing = 0
                self.current_delay = effective_base
        else:
            if left == 0 and right == 0 and attack == 0:
                self.frames_standing += 1
                self.current_delay = max(
                    self.min_delay,
                    self.base_delay - (self.frames_standing * self.delay_reduction_rate)
                )
            else:
                self.frames_standing = 0
                self.current_delay = self.base_delay
        
        # Update the frame delay in the underlying environment
        self._update_frame_delay(int(self.current_delay))
        
        return self.env.step(action)

    def _update_frame_delay(self, new_delay):
        """
        Update the frame delay queue in the underlying environment.
        
        This is done by recreating the deque with the new maxlen.
        """
        current_maxlen = self.env.delayed_frame_queue.maxlen
        target_maxlen = new_delay + 1  # +1 to match how the environment initializes it
        
        if current_maxlen != target_maxlen:
            # Preserve existing frames when resizing
            old_queue = list(self.env.delayed_frame_queue)
            # Take the most recent frames if shrinking, pad if expanding
            if len(old_queue) > new_delay:
                old_queue = old_queue[-new_delay:]
            self.env.delayed_frame_queue = deque(old_queue, maxlen=target_maxlen)

    def reset(self, **kwargs):
        """Reset the wrapper state along with the environment."""
        self.current_delay = float(self.base_delay)
        self.frames_standing = 0
        return self.env.reset(**kwargs)
