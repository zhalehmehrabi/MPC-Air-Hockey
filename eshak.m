connection = tcpclient('localhost', 8080);
fopen(connection);

while (true)
    y = callMujoco([0,0,0,0,0,0],connection);
    disp(y)
end

