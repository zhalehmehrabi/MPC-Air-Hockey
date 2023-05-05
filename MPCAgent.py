import casadi
import casadi as ca
import numpy as np
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
import copy





class MPC():
    def __init__(self, airhockey_env_info, horizon_length):

        # Constants
        self.env_info = airhockey_env_info
        self.N = horizon_length  # Horizon length
        self.dt = self.env_info['dt']  # Time step
        self.T = self.N * self.dt  # Total time

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
        self.n_states = 3

        # Number of control actions is the number of joints multiplied by number of control action per joint
        self.n_controls = n_control_action_per_joint * self.n_joints

        self.n_controls_pos = self.n_joints
        self.n_controls_vel = self.n_joints
        """
        Robots joints constraints
        """

        # desired height of the end-effector
        self.ee_desired_height = self.env_info["robot"]["ee_desired_height"]

        # The joint position and velocity limits for constraint computation is 95% of the actual limits
        constraint_computation_radio = 0.95

        # Joint position limits, Limits are in Rad
        self.upper_limit_joint_pos = np.array(self.env_info["robot"]["joint_pos_limit"][1]) * constraint_computation_radio
        self.lower_limit_joint_pos = np.array(self.env_info["robot"]["joint_pos_limit"][0]) * constraint_computation_radio

        # Joint velocity limits, Limits are in Rad/S
        self.upper_limit_joint_vel = np.array(self.env_info["robot"]["joint_vel_limit"][1]) * constraint_computation_radio
        self.lower_limit_joint_vel = np.array(self.env_info["robot"]["joint_vel_limit"][0]) * constraint_computation_radio

        # Joint acceleration limits
        self.upper_limit_joint_acc = np.array(self.env_info["robot"]["joint_acc_limit"][1])
        self.lower_limit_joint_acc = np.array(self.env_info["robot"]["joint_acc_limit"][0])

        # Robot initial position, this is fixed
        self.initial_robot_pos = np.array([-1.156, 1.300, 1.443])

        # Robot initial velocity
        self.initial_robot_vel = np.array([0, 0, 0])

        """
        These are used in forward and backward kinematics
        """
        self.robot_model = copy.deepcopy(self.env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(self.env_info['robot']['robot_data'])

        self.robot_link = [0.55, 0.44, 0.44]

        self.building_optimizer()

    def building_optimizer(self):
        """
        Penalization weight
        """
        # cost matrix for the reference tracking
        self.Q = np.diag([100, 100, 100]) # x, y, z of ee position

        # cost matrix for the action
        self.R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # pos and velocity of three joints

        """
        initial state and control action
        """

        # initial state, which is the initial position of end-effector
        self.initial_state = (forward_kinematics(self.robot_model, self.robot_data, self.initial_robot_pos)[0]).tolist()

        # initial control action
        self.initial_control_actions = np.concatenate([self.initial_robot_pos, self.initial_robot_vel]).tolist()


        # # # # # # # # # # # # # # # # # # #
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # #

        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        #

        # -- conctenated vector
        _x = ca.vertcat(px, py, pz)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        wpos1, wpos2, wpos3 = ca.SX.sym('pos1'), ca.SX.sym('pos2'), ca.SX.sym('pos3')
        wvel1, wvel2, wvel3 = ca.SX.sym('vel1'), ca.SX.sym('vel2'), ca.SX.sym('vel3')

        # -- conctenated vector
        _u = ca.vertcat(wpos1, wpos2, wpos3, wvel1, wvel2, wvel3)

        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #

        a1 = self.robot_link[0]
        a2 = self.robot_link[1]
        a3 = self.robot_link[2]

        j11 = (-a2 * ca.sin(wpos1 + wpos2)) - (a1 * ca.sin(wpos1)) - (a3 * ca.sin(wpos1 + wpos2 + wpos3))
        j12 = (-a2 * ca.sin(wpos1 + wpos2)) - (a3 * ca.sin(wpos1 + wpos2 + wpos3))
        j13 = -a3 * ca.sin(wpos1 + wpos2 + wpos3)

        j21 = (a2 * ca.cos(wpos1 + wpos2)) + (a1 * ca.cos(wpos1)) + (a3 * ca.cos(wpos1 + wpos2 + wpos3))
        j22 = (a2 * ca.cos(wpos1 + wpos2)) + (a3 * ca.cos(wpos1 + wpos2 + wpos3))
        j23 = a3 * ca.cos(wpos1 + wpos2 + wpos3)

        j1 = ca.horzcat(j11,j12,j13)
        j2 = ca.horzcat(j21,j22,j23)

        # this is because planner robot cannot control the height
        zero = 0

        j = ca.vertcat(j1,j2)

        x_dot = j @ _u[3:]

        x_dot = ca.vertcat(x_dot, zero)

        self.f = ca.Function('f', [_u], [x_dot], ['u'], ['ode'])

        # F = self.sys_dynamics(self._dt)
        # fMap = F.map(self._N, "openmp")
        #

        # todo change this to RK4 method
        x_next = _x + self.dt * x_dot

        self.f = ca.Function('f', [_u, _x], [x_next], ['_u', '_x'], ['x_next'])
        # self.f = self.fff.expand()

        # next_state_creator = self.sys_dynamics(self.dt)


        # # # # # # # # # # # # # # #
        # ---- loss function --------
        # # # # # # # # # # # # # # #

        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self.n_states)
        Delta_u = ca.SX.sym("Delta_u", self.n_controls)

        #
        cost_goal = Delta_s.T @ self.Q @ Delta_s
        cost_u = Delta_u.T @ self.R @ Delta_u

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




        self.mpc_obj = 0      # objective

        # TODO Adding table contraints to the system and their limits
        self.nlp_g = []       # constraint functions
        self.lbg = []         # lower bound of constrait functions, lbg < g
        self.ubg = []         # upper bound of constrait functions, g < ubg

        # self.OPT_variables = ca.reshape(U,(1, self.n_controls * self.N))   # nlp variables

        u_min = np.concatenate([self.lower_limit_joint_pos, self.lower_limit_joint_vel]).tolist()
        u_max = np.concatenate([self.upper_limit_joint_pos, self.upper_limit_joint_vel]).tolist()

        # here I have to put the contraints for table

        # TODO
        # x_bound = ca.inf
        # x_min = [-x_bound for _ in range(self.n_states)]
        # x_max = [+x_bound for _ in range(self.n_states)]

        x_min = [0, -0.5, 0.1]
        x_max = [0.9, 0.5, 0.1]

        g_min = [0 for _ in range(self.n_states)]
        g_max = [0 for _ in range(self.n_states)]

        # This will contain initial states and reference state
        # self.P = ca.SX.sym("P", self.n_states+(self.n_states+3)*self.N+self.n_states)

        self.P = ca.SX.sym("P", self.n_states + self.n_states)

        # This will contain the prediction of the states
        self.X = ca.SX.sym("X", self.n_states, self.N+1)

        # Control actions for all the prediction steps
        self.U = ca.SX.sym("U", self.n_controls, self.N)

        X_next = self.f(self.U,self.X[:,0])

        self.nlp_w += [self.X[:, 0]]
        self.nlp_w0 += self.initial_state
        self.lbw += x_min
        self.ubw += x_max

        # TODO constraints of the table is not coded yet

        # # starting point.
        self.nlp_g += [ self.X[:, 0] - self.P[0:self.n_states]]
        self.lbg += g_min
        self.ubg += g_max

        for k in range(self.N):
            #
            self.nlp_w += [self.U[:, k]]
            self.nlp_w0 += self.initial_control_actions
            self.lbw += u_min
            self.ubw += u_max

            # cost for tracking the goal position
            cost_goal_k, cost_gap_k = 0, 0

            delta_s_k = (self.X[:, k + 1] - self.P[self.n_states:])
            cost_goal_k = f_cost_goal(delta_s_k)

            # penalizing the use of control actions
            if k == 0:
                delta_u_k = self.U[:, k] - self.initial_control_actions
            else:
                # TODO we can add acceleration limit here
                delta_u_k = self.U[:, k] - self.U[:, k - 1]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k

            # New NLP variable for state at end of interval
            self.nlp_w += [self.X[:, k+1]]
            self.nlp_w0 += self.initial_state
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - self.X[:, k+1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {'f': self.mpc_obj,
                    'x': ca.vertcat(*self.nlp_w),
                    'p': self.P,
                    'g': ca.vertcat(*self.nlp_g)}


        ipopt_options = {
            'verbose': False, \
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
            'expand': True
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

        # TODO check the timer

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
        opt_u = sol_x0[self.n_states:self.n_states + self.n_controls]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self.n_states + self.n_controls:2 * (self.n_states + self.n_controls)]) + list(
            sol_x0[self.n_states + self.n_controls:])

        #

        x0_array = np.reshape(sol_x0[:-self.n_states], newshape=(-1, self.n_states + self.n_controls))
        x0_array = x0_array[:3,:]

        # return optimal action, and a sequence of predicted optimal trajectory.
        return opt_u, x0_array









    # Todo use this RK4 for generating next state
    def sys_dynamics(self, dt):
        M = 4  # refinement
        DT = dt / M
        X0 = ca.SX.sym("X", self.n_states)
        U = ca.SX.sym("U", self.n_controls)

        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            # K1[:3] is because the returned object is pos + orientation
            # TODO check this with amarildo
            # taking only the derivative which is for the position, not the orientation

            k1 = DT * self.f(U, X)
            k2 = DT * self.f(U, X + 0.5 * k1[:3])
            k3 = DT * self.f(U, X + 0.5 * k2[:3])
            k4 = DT * self.f(U, X + k3[:3])
            #
            X = X + (k1[:3] + 2 * k2[:3] + 2 * k3[:3] + k4[:3]) / 6
            # Fold
        F = ca.Function('F', [X0, U], [X])
        return F

            # reference must be in xyz format
    # def draw_action(self, , reference):
    #








class MyJacobian(casadi.Callback):
    def __init__(self, mjmodel, mjdata):
        casadi.Callback.__init__(self)
        self.mjmodel = mjmodel
        self.mjdata = mjdata
        self.construct("jacobian", {"enable_fd": True})

    # Number of inputs and outputs
    def get_n_in(self): return 2

    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i==0:
            return ca.Sparsity.dense(3,1)
        if i==1:
            return ca.Sparsity.dense(3,1)
    def get_sparsity_out(self, i):
        if i==0:
            return ca.Sparsity.dense(6, 1)
    # Initialize the object
    def init(self):
        print('initializing object')
    # Evaluate numerically
    def eval(self, arg):
        pos = arg[0]
        vel = arg[1]
        f = jacobian(self.mjmodel, self.mjdata, pos) @ vel
        return [f]

