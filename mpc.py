"""
Standard MPC for Passing through a dynamic gate
"""
import casadi as ca
import numpy as np
import time
from os import system
from air_hockey_challenge.utils.transformations import robot_to_world
import copy
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian


#

#
class MPCHigh(object):
    """
    Nonlinear MPC
    """
    def __init__(self,airhockey_env_info, horizon_length):
        """
        Nonlinear MPC for quadrotor control        
        """
        self.env_info = airhockey_env_info
        self._N = horizon_length  # Horizon length
        self._dt = self.env_info['dt']  # Time step
        self.T = self._N * self._dt  # Total time
        """
        Here the constraints of the robot are defined using env_info
        """

        """
        Constraints according to table and mullet
        """
        # table length and width
        self.table_length = self.env_info["table"]["length"]
        self.table_width = self.env_info["table"]["width"]

        # Mallet radius
        self.mallet_radius = self.env_info["mallet"]["radius"]

        """
        Robot physical parameters
        """
        # Number of joints
        self.n_joints = self.env_info["robot"]["n_joints"]

        # Joint velocity and position
        n_control_action_per_joint = 2

        # Number of states 3 if we just use position of end-effector(xyz), 4 if we also add the velocity
        self.n_states = 2

        # Number of control actions is the number of joints multiplied by number of control action per joint
        self.n_controls = n_control_action_per_joint * self.n_joints

        self.n_controls_pos = self.n_joints
        self.n_controls_vel = self.n_joints
        """
        Robots joints constraints
        """

        # desired height of the end-effector
        self.ee_desired_height = [self.env_info["robot"]["ee_desired_height"]]

        # The joint position and velocity limits for constraint computation is 95% of the actual limits
        constraint_computation_radio = 0.95

        # Joint position limits, Limits are in Rad
        self.upper_limit_joint_pos = np.array(
            self.env_info["robot"]["joint_pos_limit"][1]) * constraint_computation_radio
        self.lower_limit_joint_pos = np.array(
            self.env_info["robot"]["joint_pos_limit"][0]) * constraint_computation_radio

        # Joint velocity limits, Limits are in Rad/S
        self.upper_limit_joint_vel = np.array(
            self.env_info["robot"]["joint_vel_limit"][1]) * constraint_computation_radio
        self.lower_limit_joint_vel = np.array(
            self.env_info["robot"]["joint_vel_limit"][0]) * constraint_computation_radio

        # Joint acceleration limits
        self.upper_limit_joint_acc = np.array(self.env_info["robot"]["joint_acc_limit"][1])
        self.lower_limit_joint_acc = np.array(self.env_info["robot"]["joint_acc_limit"][0])

        # Robot initial position, this is fixed
        self.initial_robot_pos = [-1.156, 1.300, 1.443]

        # Robot initial velocity
        self.initial_robot_vel = [0, 0, 0]

        """
        These are used in forward and backward kinematics
        """
        self.robot_model = copy.deepcopy(self.env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(self.env_info['robot']['robot_data'])

        self.robot_link = [0.55, 0.44, 0.44]

        self.robot_base_pos = [-1.51, 0, -0.1]

        # TODO check with amarildo
        # places that are ok to go for the EE
        self.table_x_lower = [- self.table_length / 2]
        self.table_x_upper = [0]

        self.table_y_lower = [-self.table_width / 2]
        self.table_y_upper = [self.table_width / 2]

        #
        # state dimension (x,y)           # end-effector position
        #
        self._s_dim = 2
        # action dimensions (pos1, pos2, pos3, vel1, vel2, vel3) joints pos and vel
        self._u_dim = 6
        
        # cost matrix for tracking the goal point
        # self._Q_goal = np.diag([
        #     100, 100, 100])
        self._Q_goal = np.diag([
            100, 100])

        # cost matrix for the action
        self._Q_u = np.diag([0.1, 0.1, 0.1, 1, 1, 1]) # pos1, pos2, pos3, vel1, vel2, vel3

        # initial state and control action

        s0 = forward_kinematics(self.robot_model, self.robot_data, self.initial_robot_pos)[0]

        self._quad_s0 = (robot_to_world(self.env_info["robot"]["base_frame"][0], s0)[0])[:2].tolist()

        self._quad_u0 = self.initial_robot_pos + self.initial_robot_vel

        self._initDynamics()

    def _initDynamics(self):
        # # # # # # # # # # # # # # # # # # # 
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # # 

        # px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        #
        px, py = ca.SX.sym('px'), ca.SX.sym('py')

        zz = ca.SX.zeros(1)

        base_frame = ca.SX.sym('base', 4, 4)

        ee_pos_in_world_frame = ca.vertcat(px, py, zz)

        ee_pos_in_robot_frame = self.world_to_robot(base_frame, ee_pos_in_world_frame)


        # -- conctenated vector

        # self._x = ca.vertcat(px, py, pz)
        self._x = ca.vertcat(px, py)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        wpos1, wpos2, wpos3 = ca.SX.sym('pos1'), ca.SX.sym('pos2'), ca.SX.sym('pos3')
        wvel1, wvel2, wvel3 = ca.SX.sym('vel1'), ca.SX.sym('vel2'), ca.SX.sym('vel3')

        # -- conctenated vector
        self._u = ca.vertcat(wpos1, wpos2, wpos3, wvel1, wvel2, wvel3)


        # # # # # # # # # # # # # # # # # # #
        # --------- Relation between x and u--
        # # # # # # # # # # # # # # # # # # #


        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #



        a1 = self.robot_link[0]
        a2 = self.robot_link[1]
        a3 = self.robot_link[2]

        # px = (a2 * ca.cos(wpos1 + wpos2)) + (a1 * ca.cos(wpos1)) + (a3 * ca.cos(wpos1 + wpos2 + wpos3))
        # py = (a2 * ca.sin(wpos1 + wpos2)) + (a1 * ca.sin(wpos1)) + (a3 * ca.sin(wpos1 + wpos2 + wpos3))


        # this is y
        j11 = -ee_pos_in_robot_frame[0][1]
        j12 = (-a2 * ca.sin(wpos1 + wpos2)) - (a3 * ca.sin(wpos1 + wpos2 + wpos3))
        j13 = -a3 * ca.sin(wpos1 + wpos2 + wpos3)

        # this is x
        j21 = ee_pos_in_robot_frame[0][0]
        j22 = (a2 * ca.cos(wpos1 + wpos2)) + (a3 * ca.cos(wpos1 + wpos2 + wpos3))
        j23 = a3 * ca.cos(wpos1 + wpos2 + wpos3)

        j1 = ca.horzcat(j11,j12,j13)
        j2 = ca.horzcat(j21,j22,j23)

        j = ca.vertcat(j1,j2)

        x_dot = j @ self._u[3:]

        self.f = ca.Function('f', [self._x, self._u], [x_dot], ['x', 'u'], ['ode'])

        # # Fold
        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, "openmp") # parallel
        
        # # # # # # # # # # # # # # # 
        # ---- loss function --------
        # # # # # # # # # # # # # # # 

        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self._s_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)
        
        #        
        cost_goal = Delta_s.T @ self._Q_goal @ Delta_s 
        cost_u = Delta_u.T @ self._Q_u @ Delta_u

        #
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_goal])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        #
        # # # # # # # # # # # # # # # # # # # # 
        # # ---- Non-linear Optimization -----
        # # # # # # # # # # # # # # # # # # # #
        self.nlp_w = []       # nlp variables
        self.nlp_w0 = []      # initial guess of nlp variables
        self.lbw = []         # lower bound of the variables, lbw <= nlp_x
        self.ubw = []         # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0      # objective 
        self.nlp_g = []       # constraint functions
        self.lbg = []         # lower bound of constrait functions, lbg < g
        self.ubg = []         # upper bound of constrait functions, g < ubg

        u_min = np.concatenate([self.lower_limit_joint_pos, self.lower_limit_joint_vel]).tolist()
        u_max = np.concatenate([self.upper_limit_joint_pos, self.upper_limit_joint_vel]).tolist()

        # ( v1 - v2 )/dt = acc => acc * dt = v1 - v2
        acc_min = (self.lower_limit_joint_acc * self._dt).tolist()
        acc_max = (self.upper_limit_joint_acc * self._dt).tolist()

        # TODO table boundary
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._s_dim)]
        x_max = [+x_bound for _ in range(self._s_dim)]
        #
        # x_min = [0, -0.5, 0.1]
        # x_max = [0.9, 0.5, 0.1]



        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        P = ca.SX.sym("P", self._s_dim+self._s_dim)
        X = ca.SX.sym("X", self._s_dim, self._N+1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        #
        X_next = fMap(X[:, :self._N], U)
        
        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._quad_s0
        self.lbw += x_min
        self.ubw += x_max
        
        # # starting point.
        self.nlp_g += [ X[:, 0] - P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max
        
        for k in range(self._N):
            #
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._quad_u0
            self.lbw += u_min
            self.ubw += u_max

            # for controlling the acceleration, we are just talking about velocity, so we take 3 last elements of U
            # TODO check this with amarildo

            if k != 0:
                self.nlp_g += [U[3:, k] - U[3:, k-1]]
                self.lbg += acc_min
                self.ubg += acc_max
            else:
                self.nlp_g += [U[3:, k]]
                self.lbg += acc_min
                self.ubg += acc_max
            # retrieve time constant
            # idx_k = self._s_dim+self._s_dim+(self._s_dim+3)*(k)
            # idx_k_end = self._s_dim+(self._s_dim+3)*(k+1)
            # time_k = P[ idx_k : idx_k_end]

            # cost for tracking the goal position
            cost_goal_k, cost_gap_k = 0, 0

            # if k >= self._N-1: # The goal postion.

            delta_s_k = (X[:, k+1] - P[self._s_dim:])
            cost_goal_k = f_cost_goal(delta_s_k)

            # else:
            #     # cost for tracking the moving gap
            #     delta_p_k = (X[:, k+1] - P[self._s_dim+(self._s_dim+3)*k : \
            #         self._s_dim+(self._s_dim+3)*(k+1)-3])
            #     cost_gap_k = f_cost_gap(delta_p_k)
            #


            # TODO here UK - Uk-1???
            zeros_array = ca.SX.zeros(3, 1)
            if k ==0:
                delta_u_k = U[:, k]- self._quad_u0
            else:
                delta_u_k = U[:, k] - ca.vcat([U[:3,k-1], zeros_array])
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k+1]]
            self.nlp_w0 += self._quad_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k+1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {'f': self.mpc_obj, 
            'x': ca.vertcat(*self.nlp_w), 
            'p': P,               
            'g': ca.vertcat(*self.nlp_g) }        
        
        # # # # # # # # # # # # # # # # # # # 
        # -- qpoases            
        # # # # # # # # # # # # # # # # # # # 
        # nlp_options ={
        #     'verbose': False, \
        #     "qpsol": "qpoases", \
        #     "hessian_approximation": "gauss-newton", \
        #     "max_iter": 100, 
        #     "tol_du": 1e-2,
        #     "tol_pr": 1e-2,
        #     "qpsol_options": {"sparse":True, "hessian_type": "posdef", "numRefinementSteps":1} 
        # }
        # self.solver = ca.nlpsol("solver", "sqpmethod", nlp_dict, nlp_options)
        # cname = self.solver.generate_dependencies("mpc_v1.c")  
        # system('gcc -fPIC -shared ' + cname + ' -o ' + self.so_path)
        # self.solver = ca.nlpsol("solver", "sqpmethod", self.so_path, nlp_options)
        

        # # # # # # # # # # # # # # # # # # # 
        # -- ipopt
        # # # # # # # # # # # # # # # # # # # 
        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0, 
            "print_time": False
        }
        
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)
        # jit (just-in-time compilation)
        # print("Generating shared library........")
        # cname = self.solver.generate_dependencies("mpc_v1.c")
        # system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + self.so_path) # -O3

        # # reload compiled mpc
        # print(self.so_path)
        # self.solver = ca.nlpsol("solver", "ipopt", self.so_path, ipopt_options)
        #

    def solve(self, ref_states):
        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
        self.sol = self.solver(
            x0=self.nlp_w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            p=ref_states, 
            lbg=self.lbg, 
            ubg=self.ubg)
        #
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._s_dim:self._s_dim+self._u_dim]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self._s_dim+self._u_dim:2*(self._s_dim+self._u_dim)]) + list(sol_x0[self._s_dim+self._u_dim:])
        
        #
        x0_array = np.reshape(sol_x0[:-self._s_dim], newshape=(-1, self._s_dim+self._u_dim))
        
        # return optimal action, and a sequence of predicted optimal trajectory.  
        return opt_u, x0_array
    
    def sys_dynamics(self, dt):
        M = 4       # refinement
        DT = dt/M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            k1 =DT*self.f(X, U)
            k2 =DT*self.f(X+0.5*k1, U)
            k3 =DT*self.f(X+0.5*k2, U)
            k4 =DT*self.f(X+k3, U)
            #
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6        
        # Fold
        F = ca.Function('F', [X0, U], [X])
        return F

    def world_to_robot(self, base_frame, translation):
        """
        Transfrom position and rotation (optional) from the world frame to the robot's base frame

        Args
        ----
        base_frame: ndarray, (4,4)
            The transformation matrix from the world to robot base frame
        translation: ndarray, (3,)
            The 3D position to be transformed
        rotation: optional, ndarray, (3, 3)
            The rotation in the matrix form to be tranformed

        Returns
        -------
        position: ndarray, (3,)
            The transformed 3D position
        rotation: ndarray, (3, 3)
            The transformed rotation in the matrix form

        """

        target = ca.SX.eye(4)
        target[:translation.size()[0], 3] = translation
        target_frame = ca.inv(base_frame) @ target

        return target_frame[:translation.size()[0], 3], target_frame[:3, :3]
            