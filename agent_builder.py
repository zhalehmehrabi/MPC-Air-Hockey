from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics
import casadi as ca
import numpy as np
from air_hockey_agent.MPCAgent import MPC

def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    return MyAgent(env_info, **kwargs)


class MyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None
        self.horizon = 20
        self.MPC = MPC(self.env_info, self.horizon)


    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        ee_pos = self.get_ee_pose(observation)[0].tolist()
        goal = self.get_puck_pos(observation)


        ref_traj = self.generate_line_points(ee_pos,goal,self.horizon* 2 + 2).tolist()
        action, x0_array = self.MPC.solve(ref_traj)

        action = self.reshape_pos_vel_array(action)




        return action

    def generate_line_points(self, start, end, n):
        """
        Generates n points along a straight line connecting the given start and end points.

        Parameters:
        start (numpy.ndarray): Starting point of the line in 3D space.
        end (numpy.ndarray): Ending point of the line in 3D space.
        n (int): Number of points to generate on the line.

        Returns:
        numpy.ndarray: 1D array of length 3*n containing stacked x, y, and z coordinates of n points on the line.
        """
        # Generate n evenly spaced points between start and end
        points = np.linspace(start, end, n)

        # Stack the x, y, and z coordinates of the points into a single 1D array
        stacked_points = np.hstack((points[:, 0], points[:, 1], points[:, 2]))

        return stacked_points

    def reshape_pos_vel_array(self, arr):
        """
        Reshapes an input array of shape (6,) containing position and velocity vectors into a 2D array of shape (3, 2).

        Parameters:
        arr (numpy.ndarray): Input array of shape (6,) containing position and velocity vectors.

        Returns:
        numpy.ndarray: 2D array of shape (2, 3) where each column contains a position and a velocity vector.
        """
        # Split the input array into position and velocity vectors
        pos = arr[:3]
        vel = arr[3:]

        # Stack the position and velocity vectors into a 2D array
        pos_vel = np.column_stack((pos, vel))

        return pos_vel.T



