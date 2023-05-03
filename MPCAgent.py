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
        self.initial_state = forward_kinematics(self.robot_model, self.robot_data, self.initial_robot_pos)[0]

        # initial control action
        self.initial_control_actions = np.concatenate([self.initial_robot_pos, self.initial_robot_vel])


        # # # # # # # # # # # # # # # # # # #
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # #

        eex, eey, eez = ca.SX.sym('eex'), ca.SX.sym('eey'), ca.SX.sym('eez')
        # -- conctenated vector
        self._x = ca.vertcat(eex, eey, eez)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        wpos1, wpos2, wpos3, wvel1, wvel2, wvel3 = ca.MX.sym('wpos1'),ca.MX.sym('wpos2'),ca.MX.sym('wpos3'), \
                                                   ca.MX.sym('wvel1'), ca.MX.sym('wvel2'), ca.MX.sym('wvel3')

        # -- conctenated vector
        self._u = ca.vertcat(wpos1, wpos2, wpos3, wvel1, wvel2, wvel3)

        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #


        """ 
        # Based on their paper, space velocities, x_dot is computed like this: x_dot = Jacobian(joint pos)* joint vel        
        """

        # Define the Jacobian function from Kinematics as a CasADi expression

        myJacobian = MyJacobian(mjmodel=self.robot_model, mjdata=self.robot_data)

        # usage is like below
        # self.x_dot =  myJacobian(self._u[:3]) @ self._u[3:]


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

        # Control actions for all the prediction steps
        self.U = ca.MX.sym("U", self.n_controls, self.N)

        # This will contain initial state and reference state
        self.P = ca.MX.sym("P", self.n_states + self.n_states)

        # This will contain the prediction of the states
        self.X = ca.MX.sym("X", self.n_states, self.N+1)

        self.g = []       # constraint functions

        # TODO Adding table contraints to the system and their limits

        # for i in range(self.N):
        #     self.g =
        self.lbg = []         # lower bound of constrait functions, lbg < g
        self.ubg = []         # upper bound of constrait functions, g < ubg

        # self.OPT_variables = ca.reshape(U,(1, self.n_controls * self.N))   # nlp variables
        self.OPT_variables = []
        self.OPT_variables_initial = self.initial_control_actions.tolist()
        # initial guess of nlp variables
        self.lbx = []         # lower bound of the variables, lbw <= nlp_x
        self.ubx = []         # upper bound of the variables, nlp_x <= ubw




        u_min = np.concatenate([self.lower_limit_joint_pos, self.lower_limit_joint_vel]).tolist()
        u_max = np.concatenate([self.upper_limit_joint_pos, self.upper_limit_joint_vel]).tolist()

        # here I have to put the contraints for table

        # TODO
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self.n_states)]
        x_max = [+x_bound for _ in range(self.n_states)]


        g_min = [0 for _ in range(self.n_states)]
        g_max = [0 for _ in range(self.n_states)]




        # Constraint for starting point
        self.g += [X[:, 0] - P[:self.n_states]]
        self.lbg += g_min
        self.ubg += g_max

        # for k in range()
        # self.X[:, 0] = self.P[:self.n_states]

        # # starting point.




        for k in range(self.N):

            state = self.X[:,k]
            control = self.U[:,k]

            # Using x_dot for calculating next state
            x_dot =  myJacobian(control[:3]) @ control[3:]

            # TODO check this with amarildo
            # taking only the derivative which is for the position, not the orientation
            x_dot = x_dot[:3]

            # next state is calculated based on euler discritization
            next_state = state + self.dt * x_dot
            self.X[:,k+1] = next_state

        opt_state_knowing_opt_control = ca.Function('opt_state', [self.U, self.P], [self.X]);









        self.mpc_obj = 0      # objective

        # Computing cost function
        for k in range(self.N):
            cost_goal_k = 0
            delta_s_k = (self.X[:, k] - self.P[self.n_states:])
            cost_goal_k = f_cost_goal(delta_s_k)

            # penalizing the use of control actions
            if k == 0:
                delta_u_k = self.U[:, k] - self.initial_control_actions
            else:
                # TODO we can add acceleration limit here
                delta_u_k = self.U[:, k] - self.U[:, k-1]

            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k


        # Making OPT variable a vertical vector
        # TODO constraints of the table is not coded yet
        self.OPT_variables = ca.reshape(self.U, self.n_controls * self.N, 1)

        for k in range(self.N):
            self.lbx += u_min
            self.ubx += u_max



        nlp_prob = {'f': self.mpc_obj,
                    'x': self.OPT_variables,
                    'g': self.g,
                    'p': self.P}


        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, ipopt_options);

        args = {
            "lbx": self.lbx,
            "ubx": self.ubx,
            "lbg": self.lbg,
            "ubg": self.ubg
        }

        t0 = 0
        x0 = self.initial_state
        xs = [2, 2, 0.1]  # reference

        xx = []
        tt = []
        xx.append(x0)
        tt.append(t0)

        u0 = np.zeros((self.n_controls, self.N))

        mpciter = 0;
        xx1 = [];
        u_cl = [];

        # if the agent is near enough or # TODO the time reaches
        while (np.linalg.norm((x0 - xs), 2) > 1e-2):
            # set the values of the parameters vector
            args["p"] = [*x0, *xs]
            args["x0"] = np.reshape(u0, (1, self.n_controls * self.N))
            sol = solver(
                x0=args["x0"],
                lbx=args["lbx"],
                ubx=args["ubx"],
                p=args["p"],
                lbg=args["lbg"],
                ubg=args["ubg"])
            u_pred = np.reshape(sol.full(), (self.N, self.n_controls))


            # reference must be in xyz format
    # def draw_action(self, , reference):
    #


    #unchecked
    def sys_dynamics(self, robot_model, robot_data, joint_pos, joint_vel):
        return jacobian(robot_model, robot_data, joint_pos) * joint_vel

    def ff_function(self, U, P):
        X = 0
        return X


if __name__ == '__main__':
    m = MPC

# Forward and inverse kinematics functions
def fk(q):
    # Your forward kinematics implementation
    pass


def ik(x_desired):
    # Your inverse kinematics implementation
    pass


# Define the system dynamics
def robot_dynamics(x, u):
    q = x[:n_joints]
    qdot = x[n_joints:]
    q_next = q + dt * qdot
    qdot_next = qdot + dt * u
    x_next = ca.vertcat(q_next, qdot_next)
    return x_next


# Define the cost function
def cost_function(x, u, x_ref, q_ref):
    x_ee = fk(x[:n_joints])
    cost = ca.mtimes((x_ee - x_ref).T, (x_ee - x_ref)) + ca.mtimes((x[:n_joints] - q_ref).T, (x[:n_joints] - q_ref))
    return cost

def optimizationInit(envinfo):
    # MPC setup

    x = ca.MX.sym('x', n_states)
    u = ca.MX.sym('u', n_controls)
    x_ref = ca.MX.sym('x_ref', 2)
    q_ref = ca.MX.sym('q_ref', n_joints)
    x_next = robot_dynamics(x, u)
    L = cost_function(x, u, x_ref, q_ref)
    F = ca.Function('F', [x, u, x_ref, q_ref], [x_next, L], ['x', 'u', 'x_ref', 'q_ref'], ['x_next', 'L'])

    # Optimization problem
    opti = ca.Opti()
    X = opti.variable(n_states, N + 1)
    U = opti.variable(n_controls, N)
    p = opti.parameter(2)
    Q_ref = opti.parameter(n_joints, N + 1)

    # Objective function
    objective = 0
    for i in range(N):
        x_next, cost = F(X[:, i], U[:, i], p, Q_ref[:, i])
        opti.subject_to(X[:, i + 1] == x_next)
        objective += cost

    opti.minimize(objective)

    # Constraints
    for i in range(N + 1):
        opti.subject_to(opti.bounded(-np.pi, X[0, i], np.pi))  # Joint 1 angle limits
        opti.subject_to(opti.bounded(-np.pi, X[1, i], np.pi))  # Joint 2 angle limits
        opti.subject_to(opti.bounded(-np.pi, X[2, i], np.pi))  # Joint 3 angle limits

    for i in range(N):
        opti.subject_to(opti.bounded(-2 * np.pi, U[0, i], 2 * np.pi))  # Joint 1 velocity limits
        opti.subject_to(opti.bounded(-2 * np.pi, U[1, i], 2 * np.pi))  # Joint 2 velocity limits
        opti.subject_to(opti.bounded(-2 * np.pi, U[2, i], 2 * np.pi))  # Joint 3 velocity limits

    opti.subject_to(X[:, 0] == X[:, 1])  # Initial state constraint

    # Solver setup
    opti.solver('ipopt')
    solver = opti.to_function('solver', [p, X[:, 0], Q_ref], [U[:, 0]], ['p', 'x0', 'Q_ref'], ['u0'])

    # Simulation
    x0 = np.zeros(n_states)  # Initial state
    x_desired = np.array([0.5, 0.5])  # Desired end-effector position
    q_desired = ik(x_desired)  # Desired joint angles

    Q_ref_sim = np.tile(q_desired, (N + 1, 1)).T

    for t in range(int(T / dt)):
        # Update the reference
        x_current = x0[:n_joints]
        x_ref = x_desired
        u0 = solver(x_ref, x0, Q_ref_sim)
        x_next = robot_dynamics(x0, u0)
        x0 = x_next.full().flatten()

        # Do something with the control output, e.g., send itto the robot
        # send_control_to_robot(u0)

    # The code above sets up the MPC optimization problem, solves it at each time step, and updates the initial state for the next iteration. In a real application, you would replace the "send_control_to_robot" placeholder with the actual function to send the control commands to your robot. Additionally, you would need to update the initial state `x0` based on the actual robot state rather than just the simulated state.


class MyJacobian(casadi.Callback):
    def __init__(self, mjmodel, mjdata):
        casadi.Callback.__init__(self)
        self.mjmodel = mjmodel
        self.mjdata = mjdata
        self.construct("jacobian", {"enable_fd": True})

    # Number of inputs and outputs
    def get_n_in(self): return 1

    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i==0:
            return ca.Sparsity.dense(3,1)
    def get_sparsity_out(self, i):
        if i==0:
            return ca.Sparsity.dense(6, 3)
    # Initialize the object
    def init(self):
        print('initializing object')
    # Evaluate numerically
    def eval(self, arg):
        x = arg[0]
        f = jacobian(self.mjmodel, self.mjdata, x)
        return [f]

