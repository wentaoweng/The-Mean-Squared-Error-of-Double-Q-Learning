function [N,A,d,R,P,phi,policy] = GenBaird(type) %Return a Baird's instance
    N=6; %#of states
    A = 2; %# of actions
    R = zeros(N,A); % Reward Function
    
    %%type=0, Zero Reward
    %%type=1, Small Random Reward
    %%type=2, Large Random Reward
    
    for i = 1 : N
        for j = 1 : A
            switch (type)
                case 0
                    R(i,j) = 0;
                case 1
                    R(i,j) = (rand(1)-0.5)*0.1;
                case 2
                    R(i,j) = (rand(1)-0.5)*10;
            end
        end
    end
    
    %%%Specify Transition probability 
    P = zeros(N,A,N);
    for i = 1 : N
        for j = 1 : N-1
            P(i,1,j) = 1 / (N - 1);
        end
        P(i,2,N) = 1;
    end
    
    %%adopt uniform policy. Do normalization.
    policy = ones(N,A);
    for i = 1 : N
        policy(i,:) = policy(i,:) / sum(policy(i,:));
        for j = 1 : A
            s = sum(P(i,j,:));
            P(i,j,:) = P(i,j,:) / s;
        end
    end
    
    %%construct feature matrix, see the paper for details
    d = 12;
    phi = zeros(N*A,d);
    for i = 1 : N
        label = (i - 1)* A + 1;
        if (i ~= 1)
            phi(label,label) = 1;
        else
            phi(label,12) = 1;
        end
        if (i ~= N)
            phi(label+1,label+1) = 2;
            phi(label+1,1) = 1;
        else
            phi(label+1,label+1) = 1;
            phi(label+1,1) = 2;
        end
    end
    

end