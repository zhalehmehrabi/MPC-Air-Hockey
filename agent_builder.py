from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
import casadi as ca
import numpy as np
from air_hockey_agent.MPCAgent import MPC
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import socket


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
        self.last_action = np.array(self.MPC.initial_control_actions).reshape(1, self.MPC.n_controls)

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
        self.history_of_pos1.append(self.MPC.initial_control_actions[0])
        self.history_of_pos2.append(self.MPC.initial_control_actions[1])
        self.history_of_pos3.append(self.MPC.initial_control_actions[2])
        self.history_of_vel1.append(self.MPC.initial_control_actions[3])
        self.history_of_vel2.append(self.MPC.initial_control_actions[4])
        self.history_of_vel3.append(self.MPC.initial_control_actions[5])
        self.time.append(self.t)

    def draw_action(self, observation):


        if self.sent:
            result_array = self.get_ee_pose(observation)[0] + np.array(self.MPC.robot_base_pos)
            print("next_state = ", result_array)
            # Send result back to socket
            result_str = ' '.join(str(x) for x in result_array.flatten())
            self.conn.sendall(result_str.encode())
            self.sent = False
        # Initial pos of EE

        ee_pos = (self.get_ee_pose(observation)[0] + np.array(self.MPC.robot_base_pos)).tolist()
        # Reference pos of EE
        # goal = self.get_puck_pos(observation)
        goal = np.array([-0.8, 0, 0])

        data = self.conn.recv(1024)

        # Convert message to array of shape 1x3
        data_str = data.decode('utf-8')
        data_list = [float(x) for x in data_str.split()]
        data_array = np.array(data_list).reshape((2, 3))

        print(data_array)
        self.sent = True


        """
        checking JACOBIANS
        """
        # next_state_my = self.MPC.f(self.last_action[0,:], ee_pos)
        #
        # x_dot_muj = (jacobian(self.MPC.robot_model, self.robot_data, self.last_action[0,:3]) @ self.last_action[0,3:])[:3]
        # next_state_muj = ee_pos + self.MPC.dt * x_dot_muj[:3]




        # ref_traj = ee_pos + goal.tolist()
        # action, x0_array = self.MPC.solve(ref_traj, np.tile(self.last_action,(self.MPC.N, 1)))
        #
        #

        #
        # self.last_action = action
        # self.t += 1
        #
        # self.history_of_pos1.append(action[0,0])
        # self.history_of_pos2.append(action[0,1])
        # self.history_of_pos3.append(action[0,2])
        # self.history_of_vel1.append(action[0,3])
        # self.history_of_vel2.append(action[0,4])
        # self.history_of_vel3.append(action[0,5])
        # self.time.append(self.t)
        #
        # # create six scatter plots
        # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        #
        # axs[0, 0].scatter(self.time, self.history_of_pos1)
        # axs[0, 0].set_title('POS 1')
        #
        # axs[0, 1].scatter(self.time, self.history_of_pos2)
        # axs[0, 1].set_title('POS 2')
        #
        # axs[0, 2].scatter(self.time, self.history_of_pos3)
        # axs[0, 2].set_title('POS 3')
        #
        # axs[1, 0].scatter(self.time, self.history_of_vel1)
        # axs[1, 0].set_title('VEL 1')
        #
        # axs[1, 1].scatter(self.time, self.history_of_vel2)
        # axs[1, 1].set_title('VEL 2')
        #
        # axs[1, 2].scatter(self.time, self.history_of_vel3)
        # axs[1, 2].set_title('VEL 3')
        #
        # plt.show()
        # return self.reshape_pos_vel_array(action)
        return data_array
    def submit_to_history(self, trajectory):
        self.t += 1


        xx = []

        t = []
        # add my_list to history as a new row

        r = np.array(x0).reshape(3, 1)
        # add another list to history
        xx.append(r)

        # xx = np.hstack([xx, r])

        t.append(t0)

        u0 = np.zeros((self.N,self.n_controls))

        # maximum simulation time
        sim_tim = 20

        mpciter = 0
        xx1 = []
        u_cl = []
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

class DummyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

        velocity = np.zeros_like(self.hold_position)
        action = np.vstack([self.hold_position, velocity])
        return action