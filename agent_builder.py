from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
import casadi as ca
import numpy as np
from air_hockey_agent.MPCAgent import MPC
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import socket
import mpc


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
    # return DummyAgent(env_info, **kwargs)


class MyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.horizon = 20
        self.MPC = MPC(self.env_info, self.horizon)
        self.history_predicted = []



        self.actual_pos = []

        self.history_of_pos1 = []
        self.history_of_pos2 = []
        self.history_of_pos3 = []
        self.history_of_vel1 = []
        self.history_of_vel2 = []
        self.history_of_vel3 = []
        self.time = []

        # self.last_action = np.zeros((1,self.MPC.n_controls))
        # self.last_action = np.array(self.MPC.initial_control_actions).reshape(1, self.MPC.n_controls)

        self.t = 0

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('localhost', 8080))
        print("waiting for connection")
        self.s.listen(1)
        self.conn, addr = self.s.accept()
        print("conneceted")
        self.sent = False


    def reset(self):
        self.history = np.array([])
        self.t = 0
        self.history_of_pos1 = []
        self.history_of_pos2 = []
        self.history_of_pos3 = []
        self.history_of_vel1 = []
        self.history_of_vel2 = []
        self.history_of_vel3 = []
        self.time = []
        self.sent = False

        self.time.append(self.t)

    def draw_action(self, observation):
        """ THIS IS FOR MATLAB
        ee_pos, _ = self.get_ee_pose(observation)
        ee_pos_for_MPC = robot_to_world(self.MPC.env_info["robot"]["base_frame"][0], ee_pos)[0]
        print("initial_state = ", ee_pos_for_MPC)
        # Send result back to socket
        result_str = ' '.join(str(x) for x in ee_pos_for_MPC.flatten())
        self.conn.sendall(result_str.encode())


        data = self.conn.recv(1024)

        data_str = data.decode('utf-8')
        data_list = [float(x) for x in data_str.split()]

        data_array = np.array(data_list)
        resh = data_array.reshape(2,3)
        pos = data_array[:3]
        vel = data_array[3:]

        data_array = np.vstack([pos, vel])
        """


        """
        checking JACOBIANS
        """
        # next_state_my = self.MPC.f(self.last_action[0,:], ee_pos)
        #
        # x_dot_muj = (jacobian(self.MPC.robot_model, self.robot_data, self.last_action[0,:3]) @ self.last_action[0,3:])[:3]
        # next_state_muj = ee_pos + self.MPC.dt * x_dot_muj[:3]

        ee_pos, _ = self.get_ee_pose(observation)
        ee_pos_for_MPC = robot_to_world(self.MPC.env_info["robot"]["base_frame"][0], ee_pos)[0]

        ref_traj = ee_pos_for_MPC + []
        # ref_traj = ee_pos + goal.tolist()
        action, x0_array = self.MPC.solve(ref_traj, np.tile(self.last_action,(self.MPC.N, 1)))

        return self.reshape_pos_vel_array(action)
        # return data_array

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
        pos_vel = arr.reshape(2,3)

        return pos_vel

    def _action_transform(self, action, obs):

        joint_pos = self.get_joint_pos(obs)
        joint_vel = self.get_joint_vel(obs)

        command = world_to_robot(self.MPC.env_info["robot"]["base_frame"][0], action)[0]
        success, new_joint_pos = inverse_kinematics(self.robot_model, self.robot_data, command)
        joint_velocities = (new_joint_pos - self.get_joint_pos(obs)) / self.MPC.env_info['dt']

        # if not success:
        #     self._fail_count += 1
        #     # new_joint_pos = joint_pos

        action = np.vstack([new_joint_pos, joint_velocities])
        return action
