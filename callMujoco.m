% Open socket connection
function next_state = callMujoco(control_Actions, t)


% Convert the data to a string
data_str = sprintf('%f ', control_Actions);

% Send the data to the socket connection

fwrite(t, data_str);


% Display the response


% Close the socket connection
end