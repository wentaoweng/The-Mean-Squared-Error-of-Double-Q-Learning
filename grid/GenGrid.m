function [N,A,d,R,P,phi,policy,terminal] = GenGrid(n)
    %%In this GridWorld, when the agent arrives (n,n), she will get
    %%transmitted back to (1,1)
    
    dir = [1,0;0,1;-1,0;0,-1]; %four directions
    
    %%label grids
    label = zeros(n,n);
    N = n*n; %Total #of states
    for x = 1 : n
        for y = 1 : n
            label(x,y) = (x-1)*n+y;
        end
    end
    
    N = N + 1; %%terminal (absorbing) state
    
    A = 4; %#of actions
    R = zeros(N,A);
    
    %%Specify whether a state is a terminal state
    terminal = zeros(N,1);
    terminal(N) = 1;
    
    %Transition probability
    P = zeros(N,A,N);
    
    for x = 1 : n
        for y = 1 : n
            %%Construct reward function, and transition matrix
            order = label(x,y);
            
            for j = 1 : A
                if (x == n && y == n)
                    P(order,j,N) = 1; %end of episode
                    R(order,j) = 1;
                else
                    for nj = 1 : A %%Random jump, see the paper for details
                        pr = 0.7 + 0.3 * 1.0 / 4;
                        %pr = 1;
                        if (nj ~= j)
                            pr = 0.3 * 1.0 / 4;
                        %    pr = 0 * 1.0 / 4;
                        end
                        %%(nx,ny): next grid
                        nx = x + dir(nj,1);
                        ny = y + dir(nj,2);
                        %%If out of map, stay at the same position
                        if (nx < 1 || ny < 1 || nx > n || ny > n)
                            nx = x;
                            ny = y;
                        end
                        %%Total transition probability to the next state
                        P(order,j,label(nx,ny)) = P(order,j,label(nx,ny))+pr;
                    end
                    R(order,j) = -1e-3;
                end
            end
        end
    end
    
    for j = 1 : 4 %if the game ends, just starts another episode
        P(N,j,N) = 1.0;
    end
    
    %Uniform behaviroal policy
    policy = ones(N,A);

     for i = 1 : N
        policy(i,:) = policy(i,:) / sum(policy(i,:));
        for j = 1 : A
            s = sum(P(i,j,:));
            P(i,j,:) = P(i,j,:) / s;
        end
     end

    %Use tabular(look-up table) as function approximations
    %Feature matrix is a diagonal matrix
    d = (N-1) * A; %terminal state has no value function
    phi = eye(N * A,d);

end