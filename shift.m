function [t0, u0] = shift(T, t0, u,n_control, connection, x_for_send)
%con = u(1,:)';

con = x_for_send;
% send con to python, get next_state which is end-effectors current
% position after we applied con
callMujoco(con, connection);
%next_state = split(next_state);

% Convert each string to a numeric value
%next_state = str2double(next_state);

%[m,n] = size(st);

%disp(next_state)
%next_state_reshaped = reshape(next_state, m, n);

%x0 = next_state_reshaped;

t0 = t0 + T;
u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end