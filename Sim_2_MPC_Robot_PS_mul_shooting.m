
% first casadi test for mpc fpr mobile robots
clear all
close all
clc

% CasADi v3.4.5
% addpath('C:\Users\mehre\OneDrive\Desktop\CasADi\casadi-windows-matlabR2016a-v3.4.5')
% CasADi v3.5.5

addpath('F:\Programs\CasadiMatlab\casadi-3.6.2-windows64-matlab2018b')

import casadi.*

connection = tcpclient('localhost', 8080);
fopen(connection)


T = 0.02; %[s]
N = 20; % prediction horizon
rob_diam = 0.3;

r_mallet = 0.04815;
table_width = 1.038;
table_length = 1.948;

x_lower = (-table_length / 2) + r_mallet;
x_upper = 0;

y_lower = (-table_width / 2) + r_mallet;
y_upper = -y_lower;

pos_max = [2.81871, 1.71000, 1.98968];
pos_min = -pos_max;

vel_max = [1.49225651, 1.49225651, 1.98967535];
vel_min = -vel_max;

x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z');
states = [x;y;z]; n_states = length(states);

wpos1 = SX.sym('pos1'); wpos2 = SX.sym('pos2'); wpos3 = SX.sym('pos3');
wvel1 = SX.sym('vel1'); wvel2 = SX.sym('vel2'); wvel3 = SX.sym('vel3');

controls = [wpos1, wpos2, wpos3, wvel1, wvel2, wvel3]; 
n_controls = length(controls);

robot_link_length = [0.55, 0.44, 0.44];

a1 = robot_link_length(1);
a2 = robot_link_length(2);
a3 = robot_link_length(3);


jac = [(-a2 * sin(wpos1 + wpos2)) - (a1 * sin(wpos1)) - (a3 * sin(wpos1 + wpos2 + wpos3)),...
       (-a2 * sin(wpos1 + wpos2)) - (a3 * sin(wpos1 + wpos2 + wpos3)),...
       -a3 * sin(wpos1 + wpos2 + wpos3);...
       (a2 * cos(wpos1 + wpos2)) + (a1 * cos(wpos1)) + (a3 * cos(wpos1 + wpos2 + wpos3)),...
       (a2 * cos(wpos1 + wpos2)) + (a3 * cos(wpos1 + wpos2 + wpos3)),...
       a3 * cos(wpos1 + wpos2 + wpos3)]; % system r.h.s

rhs = jac * controls(4:6)';
rhs = [rhs;0];

f = Function('f',{states,controls},{rhs}); % nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N); % Decision variables (controls)
P = SX.sym('P',n_states + n_states);
% parameters (which include the initial state and the reference state)

X = SX.sym('X',n_states,(N+1));
% A vector that represents the states over the optimization problem.

obj = 0; % Objective function
g = [];  % constraints vector

Q = zeros(3,3); Q(1,1) = 100;Q(2,2) = 100;Q(3,3) = 100; % weighing matrices (states)
R = zeros(6,6); R(1,1) = 1; R(2,2) = 1;R(3,3) = 1;
                R(4,4) = 1; R(5,5) = 1;R(6,6) = 1;% weighing matrices (controls)


st  = X(:,1); % initial state
g = [g;st-P(1:3)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k);
    obj = obj+(st-P(4:6))'*Q*(st-P(4:6)) + con'*R*con; % calculate obj
    st_next = X(:,k+1);
    f_value = f(st,con);
    st_next_euler = st + (T*f_value);
    g = [g;st_next-st_next_euler]; % compute constraints
end
% make the decision variable one column  vector
OPT_variables = [reshape(X,n_states*(N+1),1);reshape(U,n_controls*N,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 2000;
opts.ipopt.print_level =0;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

args = struct;

args.lbg(1:n_states*(N+1)) = 0;  % -1e-20  % Equality constraints
args.ubg(1:n_states*(N+1)) = 0;  % 1e-20   % Equality constraints

args.lbx(1:n_states:n_states*(N+1),1) = x_lower; %state x lower bound
args.ubx(1:n_states:n_states*(N+1),1) = x_upper; %state x upper bound
args.lbx(2:n_states:n_states*(N+1),1) = y_lower; %state y lower bound
args.ubx(2:n_states:n_states*(N+1),1) = y_upper; %state y upper bound
%args.lbx(3:n_states:n_states*(N+1),1) = 0.1; %state z lower bound
%args.ubx(3:n_states:n_states*(N+1),1) = 0.1; %state z upper bound

args.lbx(3:n_states:n_states*(N+1),1) = 0; %state z lower bound
args.ubx(3:n_states:n_states*(N+1),1) = 0; %state z upper bound


args.lbx(3*(N+1)+1:n_controls:3*(N+1)+n_controls*N,1) = pos_min(1); %po1 lower bound
args.ubx(3*(N+1)+1:n_controls:3*(N+1)+n_controls*N,1) = pos_max(1); %pos1 upper bound
args.lbx(3*(N+1)+2:n_controls:3*(N+1)+n_controls*N,1) = pos_min(2); %omega lower bound
args.ubx(3*(N+1)+2:n_controls:3*(N+1)+n_controls*N,1) = pos_max(2); %omega upper bound
args.lbx(3*(N+1)+3:n_controls:3*(N+1)+n_controls*N,1) = pos_min(3); %v lower bound
args.ubx(3*(N+1)+3:n_controls:3*(N+1)+n_controls*N,1) = pos_max(3); %v upper bound
args.lbx(3*(N+1)+4:n_controls:3*(N+1)+n_controls*N,1) = vel_min(1); %omega lower bound
args.ubx(3*(N+1)+4:n_controls:3*(N+1)+n_controls*N,1) = vel_max(1); %omega upper bound
args.lbx(3*(N+1)+5:n_controls:3*(N+1)+n_controls*N,1) = vel_min(2); %v lower bound
args.ubx(3*(N+1)+5:n_controls:3*(N+1)+n_controls*N,1) = vel_max(2); %v upper bound
args.lbx(3*(N+1)+6:n_controls:3*(N+1)+n_controls*N,1) = vel_min(3); %omega lower bound
args.ubx(3*(N+1)+6:n_controls:3*(N+1)+n_controls*N,1) = vel_max(3); %omega upper bound

%----------------------------------------------
% ALL OF THE ABOVE IS JUST A PROBLEM SET UP


% THE SIMULATION LOOP SHOULD START FROM HERE
%-------------------------------------------
t0 = 0;
x0 = [-0.8600679825633534; 2.054985850341584e-05; 0.0];    % initial condition.
xs = [-0.6 ; -0.1 ; 0.0]; % Reference posture.

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,n_controls);        % two control inputs for each robot
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables

sim_tim = 20; % Maximum simulation time

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];

% the main simulaton loop... it works as long as the error is greater
% than 10^-6 and the number of mpc steps is less than its maximum
% value.
tic
while(norm((x0-xs),2) > 1e-2 && mpciter < sim_tim / T && mpciter < 100)
    args.p   = [x0;xs]; % set the values of the parameters vector
    % initial value of the optimization variables
    %args.x0  = [reshape(X0',n_states*(N+1),1);reshape(u0',n_controls*N,1)];
    args.x0  = [reshape(X0',n_states*(N+1),1);reshape(u0',n_controls*N,1)];

    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u = reshape(full(sol.x(n_states*(N+1)+1:end))',n_controls,N)'; % get controls only from the solution
    xx1(:,1:3,mpciter+1)= reshape(full(sol.x(1:3*(N+1)))',3,N+1)'; % get solution TRAJECTORY
    u_cl= [u_cl ; u(1,:)];
    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift(T, t0, x0, u,n_controls, connection);
    xx(:,mpciter+2) = x0;
    X0 = reshape(full(sol.x(1:3*(N+1)))',3,N+1)'; % get solution TRAJECTORY
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    mpciter
    mpciter = mpciter + 1;
end;
toc

ss_error = norm((x0-xs),2)
Draw_MPC_point_stabilization_v1 (t,xx,xx1,u_cl,xs,N,rob_diam)
fclose(t);

