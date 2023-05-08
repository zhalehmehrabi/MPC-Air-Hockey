function [t0, x0, u0] = shift(T, t0, x0, u,n_control)
st = x0;
con = u(1,:)';

% send con to python, get next_state which is end-effectors current
% position after we applied con
next_state = callMujoco(con);

[m,n] = size(st);
next_state_reshaped = reshape(next_state, m, n);

x0 = next_state_reshaped;

t0 = t0 + T;
u0 = [u(n_control:size(u,1),:);u(size(u,1),:)];
end