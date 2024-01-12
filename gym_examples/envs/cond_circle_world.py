import gym
from gym import spaces
import pygame
import numpy as np


class CondCircleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, radius=1, in_out="in"):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.radius = radius
        self.in_out = in_out
        self.origin_coord = np.array([size/2, size/2])

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, size, shape=(2,), dtype=float)

        # x and y acceleration
        self.action_space = spaces.Box(-0.1, 0.1, shape=(2,), dtype=float)


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "distance_to_origin": np.sqrt(
                np.sum((self._agent_location - self.origin_coord) ** 2)
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.origin_coord + np.random.uniform(-1.5, 1.5, (2,))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        speed = np.sqrt(np.sum(action ** 2))
        direction = action / speed if speed > 0 else np.zeros_like(action)
        # We use `np.clip` to make sure we don't leave the grid
        velocity = np.clip(speed * direction, -0.1, 0.1)
        self._agent_location = np.clip(self._agent_location + velocity, 0, self.size)

        # An episode is done iff the agent has reached the target
        terminated = False
        info = self._get_info()
        distance = info["distance_to_origin"] 
        is_in = distance <= self.radius
            
        reward = 0
        if self.in_out == "in" and is_in:
            reward = 1
        elif self.in_out == "out" and not is_in:
            reward = 1
        observation = self._get_obs()

        # reward += 0.01 * np.sqrt(np.sum(self._agent_velocity ** 2))
        if distance >= 0.45 * self.size:
            reward = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 0, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Now we draw the agent
        center =  (self._agent_location + 0.5) * pix_square_size
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            center.flatten().tolist(),
            pix_square_size / 5,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
