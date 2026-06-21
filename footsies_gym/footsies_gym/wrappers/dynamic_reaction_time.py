from collections import deque
import gymnasium as gym
from ..envs.footsies import FootsiesEnv


class DynamicReactionTimeWrapper(gym.Wrapper):
    """
    Wrapper that dynamically adjusts frame delay based on agent actions.
    
    Simulates human-like reaction time variation where standing still and focusing
    can slightly decrease average reaction time, while active actions increase it.
    
    Parameters
    ----------
    env : gym.Env
        The FOOTSIES environment to wrap. Should have a `delayed_frame_queue` attribute.
    base_delay : int
        The base frame delay when the agent is performing actions. Default: 5
    min_delay : int
        Minimum possible frame delay. Default: 0
    max_delay : int
        Maximum possible frame delay. Default: 10
    delay_reduction_rate : float
        How much to reduce delay per frame while standing still. Default: 0.1
    """

    def __init__(self, env, base_delay=5, min_delay=0, max_delay=10, delay_reduction_rate=0.1):
        super().__init__(env)
        
        if not hasattr(env, 'delayed_frame_queue'):
            raise ValueError("DynamicReactionTimeWrapper requires an environment with a 'delayed_frame_queue' attribute")
        
        self.base_delay = base_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay_reduction_rate = delay_reduction_rate
        
        self.current_delay = float(base_delay)
        self.frames_standing = 0

    def step(self, action):
        """
        Step the environment, adjusting frame delay based on the action taken.
        
        If the agent takes no action (standing still), the frame delay gradually decreases.
        If the agent takes any action, the delay resets to base_delay.
        """
        # action is tuple (left, right, attack)
        left, right, attack = action
        
        # If standing still (no movement, no attack)
        if left == 0 and right == 0 and attack == 0:
            self.frames_standing += 1
            # Reduce delay over time (faster reaction)
            self.current_delay = max(
                self.min_delay,
                self.base_delay - (self.frames_standing * self.delay_reduction_rate)
            )
        else:
            # Reset standing counter on any action
            self.frames_standing = 0
            # Reset delay back to base when acting
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
