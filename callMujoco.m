% Open socket connection
function next_state = callMujoco(control_Actions)
t = tcpclient('localhost', 8080);
fopen(t)

% Convert the data to a string
data_str = sprintf('%d ', control_Actions);

% Send the data to the socket connection
while true
    fwrite(t, data_str);
    
    % Wait for the response
    response = '';
    while (get(t, 'BytesAvailable') == 0)
        % Wait for data to be available
    end
    while (get(t, 'BytesAvailable') > 0)
        % Read the available data and append it to the response string
        response = [response, char(fread(t, 1))];
    end
    
    % Display the response
    next_state = response;
end
% Close the socket connection
fclose(t);
end