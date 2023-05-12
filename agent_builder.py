from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
import casadi as ca
import numpy as np
from air_hockey_agent.MPCAgent import MPC
import socket
from air_hockey_agent.mpc import MPCHigh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


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
        self.MPC = MPCHigh(self.env_info, self.horizon)

        # for visualization
        self.t0 = 0
        self.xx = []
        self.xx1 = []
        self.u_cl = []
        self.last_action = self.MPC._quad_u0




    def reset(self):
        self.t0 = 0
        self.xx = []
        self.xx1 = []
        self.u_cl = []
    def draw_action(self, observation):

        ee_pos, _ = self.get_ee_pose(observation)
        ee_pos_for_MPC = (robot_to_world(self.MPC.env_info["robot"]["base_frame"][0], ee_pos)[0])[:2].tolist()
        self.xs = [-0.8 , -0.3 ]
        ref_traj = ee_pos_for_MPC + self.xs
        # ref_traj = ee_pos + goal.tolist()
        action, x0_array = self.MPC.solve(ref_traj)

        state_as_input = x0_array[1,:2]
        command = np.concatenate([state_as_input, np.zeros(1)])
        inverse = self._action_transform(command, observation)




        """
        checking JACOBIANS
        """

        x_dot_me = self.MPC.f(ee_pos_for_MPC, np.array(self.last_action))

        x_dot_muj = (jacobian(self.MPC.robot_model, self.robot_data, self.last_action[:3]) @ self.last_action[3:])[:3]

        self.last_action = action.reshape(6).tolist()

        pos = action[:3].reshape(1,3)
        vel = action[3:].reshape(1,3)

        action = np.vstack([pos, vel])
        """
        for visualization
        """
        # for orientation it is considered to be zero for now
        orientation = 0
        self.t0 += self.MPC._dt
        self.xx.append(ee_pos_for_MPC)
        self.xx1.append(x0_array[:,:2])
        self.u_cl.append(action)

        """
        End of visualization
        """



        # return action
        return inverse
        # return data_array
    def __del__(self):
        print("slm slm")
        v = Visualization(self.t0, self.xx, self.xx1, self.u_cl, self.xs + [0], self.horizon, 0.3)

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

class Visualization():
    def __init__(self, t0, xx, xx1, u_cl, xs, horizon, robot_dim):
        self.t0 = t0
        self.xx = xx
        self.xx1 = xx1
        self.u_cl = u_cl
        self.xs =xs
        self.horizon = horizon
        self.robot_dim = robot_dim
        self.fig = plt.figure(500, figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)

        self.line_width = 1.5
        self.fontsize_labels = 14
        self.x_r_1 = []
        self.y_r_1 = []
        self.r = robot_dim/2
        ang = np.arange(0, 2*np.pi, 0.005)
        self.xp = self.r*np.cos(ang)
        self.yp = self.r*np.sin(ang)
        # fig = plt.figure(500, figsize=(8, 8))
        # ax = fig.add_subplot(111)
        patches = []

        self.Draw_MPC_point_stabilization_v1(self.t0, self.xx, self.xx1, self.u_cl, self.xs, self.horizon, self.robot_dim)



    def Draw_MPC_point_stabilization_v1(self,t, xx, xx1, u_cl, xs, N, rob_diam):


        self.line_width = 1.5
        self.fontsize_labels = 14
        self.x_r_1 = []
        self.x_r_1 = []
        self.r = rob_diam/2
        ang = np.arange(0, 2*np.pi, 0.005)
        self.xp = self.r*np.cos(ang)
        self.yp = self.r*np.sin(ang)
        # fig = plt.figure(500, figsize=(8, 8))
        # ax = fig.add_subplot(111)
        patches = []
        # Animate the robot motion
        #
        # for k in range(len(xx)):
        #     h_t = 0.14
        #     w_t = 0.09
        #     x1 = xs[0]
        #     y1 = xs[1]
        #     # th1 = xs[2]
        #     th1 = 0
        #
        #     x1_tri = [x1+h_t*np.cos(th1), x1+(w_t/2)*np.cos((np.pi/2)-th1), x1-(w_t/2)*np.cos((np.pi/2)-th1)]
        #     y1_tri = [y1+h_t*np.sin(th1), y1-(w_t/2)*np.sin((np.pi/2)-th1), y1+(w_t/2)*np.sin((np.pi/2)-th1)]
        #     ref_state = Polygon(np.array([x1_tri, y1_tri]).T, True)
        #     patches.append(ref_state)
        #     x1 = xx[k][0]
        #     y1 = xx[k][1]
        #     # th1 = xx[k,2]
        #     th1 = 0
        #     self.x_r_1.append(x1)
        #     self.x_r_1.append(y1)
        #     x1_tri = [x1+h_t*np.cos(th1), x1+(w_t/2)*np.cos((np.pi/2)-th1), x1-(w_t/2)*np.cos((np.pi/2)-th1)]
        #     y1_tri = [y1+h_t*np.sin(th1), y1-(w_t/2)*np.sin((np.pi/2)-th1), y1+(w_t/2)*np.sin((np.pi/2)-th1)]
        #     exhibited_traj = Line2D(self.x_r_1, self.x_r_1, linewidth=self.line_width, color='r')
        #     robot_pos = Polygon(np.array([x1_tri, y1_tri]).T, True, facecolor='r')
        #     robot_circle = Circle((x1, y1), r, fill=False, linestyle='--', color='r')
        #     ax.add_patch(robot_pos)
        #     ax.add_patch(robot_circle)
        #     ax.add_line(exhibited_traj)
        #     if k < len(xx) - 1:
        #         ax.plot(xx1[k][0:N, 0], xx1[k][0:N, 1], 'r--*')
        #     ax.set_xlabel('$x$-position (m)', fontsize=self.fontsize_labels)
        #     ax.set_ylabel('$y$-position (m)', fontsize=self.fontsize_labels)
        #     ax.set_xlim([-1, 0])
        #     ax.set_ylim([-0.5, 0.5])
        #     ax.set_aspect('equal')

        anim = FuncAnimation(self.fig, self.update, frames=range(len(self.xx)), interval=50)
        # anim.save('./animation.gif', writer='PillowWriter')
        # anim = animation.FuncAnimation(fig, animate, interval=100, blit=True)
        plt.show()
        # while True:
        #     try:
        #         plt.pause(0.1)
        #     except:
        #         break
        #
# def animate(i):
#     ax.imshow(frames[i])
#     return (ax,)

    def update(self, frame):
        # Clear previous plot contents
        self.ax.clear()
        N = self.horizon
        k = frame
        h_t = 0.14
        w_t = 0.09
        x1 = self.xs[0]
        y1 = self.xs[1]
        # th1 = xs[2]
        th1 = 0

        x1_tri = [x1 + h_t * np.cos(th1), x1 + (w_t / 2) * np.cos((np.pi / 2) - th1),
                  x1 - (w_t / 2) * np.cos((np.pi / 2) - th1)]
        y1_tri = [y1 + h_t * np.sin(th1), y1 - (w_t / 2) * np.sin((np.pi / 2) - th1),
                  y1 + (w_t / 2) * np.sin((np.pi / 2) - th1)]
        ref_state = Polygon(np.array([x1_tri, y1_tri]).T, True)
        x1 = self.xx[k][0]
        y1 = self.xx[k][1]
        # th1 = xx[k,2]
        th1 = 0
        self.x_r_1.append(x1)
        self.x_r_1.append(y1)
        x1_tri = [x1 + h_t * np.cos(th1), x1 + (w_t / 2) * np.cos((np.pi / 2) - th1),
                  x1 - (w_t / 2) * np.cos((np.pi / 2) - th1)]
        y1_tri = [y1 + h_t * np.sin(th1), y1 - (w_t / 2) * np.sin((np.pi / 2) - th1),
                  y1 + (w_t / 2) * np.sin((np.pi / 2) - th1)]
        exhibited_traj = Line2D(self.x_r_1, self.x_r_1, linewidth=self.line_width, color='r')
        robot_pos = Polygon(np.array([x1_tri, y1_tri]).T, True, facecolor='r')
        robot_circle = Circle((x1, y1), self.r, fill=False, linestyle='--', color='r')
        self.ax.add_patch(robot_pos)
        self.ax.add_patch(robot_circle)
        self.ax.add_patch(ref_state)
        # self.ax.add_line(exhibited_traj)
        if k < len(self.xx) - 1:
            self.ax.plot(self.xx1[k][0:N, 0], self.xx1[k][0:N, 1], 'r--*')
        self.ax.set_xlabel('$x$-position (m)', fontsize=self.fontsize_labels)
        self.ax.set_ylabel('$y$-position (m)', fontsize=self.fontsize_labels)
        self.ax.set_xlim([-1, 0])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_aspect('equal')

        # Set plot title
        self.ax.set_title(f"Frame {frame}")
        # plt.show()
