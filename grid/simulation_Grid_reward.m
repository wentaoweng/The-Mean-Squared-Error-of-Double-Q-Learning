rng(5); %%fix random seed, for reproduction

%Construct a Grid instance
%N,A,d: #of states & actions & dimension of approximations
%R,P: Reward function & Transition probability
%phi: approximation basis: must be a tabular in our experiment
%policy: behaviroal policy
%Input to GenGrid: Size of the Grid
[N,A,d,R,P,phi,policy,terminal] = GenGrid(3);

cnt = 0;

is_double = true;
is_twiced = true;
is_averaged = true;

ti = 100000; %Number of samples per test
expt = 100; %#of tests, where the mean-squared error is averaged

gamma = 0.9; %set discount factor

%Arrays to save #of samples to achieve the destination
round = zeros(expt,1);

%%Use iteration to solve a projected Bellman Equation to get \theta^*
theta_star = zeros(d,1);
V = zeros(N,1); %Value function of each state
prb = zeros(N,1);

%%%Identify the optimal estimation through iteration
for t = 1:100000
    lst = theta_star;
    epsilon = 1000 / (t + 10000);
    for i = 1 : N
        for j = 1 : A
                for p = 1 : N           
                    %Find the optimal action of state p,based on estimator \theta^*
                    act = find_Max(N,A,p,phi,theta_star);
                    %Update the value function of state p
                    V(p) = phi((p-1)*A+act,:)*theta_star;
                    %Probability to go to state p if we use action j at
                    %state i
                    prb(p) = P(i,j,p);
                end
                tar = R(i,j) + gamma * prb' * V;
                cur = (i - 1) * A + j;
                %Update \theta^* through Bellman equation
                theta_star = theta_star+epsilon*phi(cur,:)'*(tar - phi(cur,:)*theta_star);
        end
    end
    %If the difference to the last estimator is small enough, terminate the iteration
    if (max(abs(theta_star - lst)) < 1e-10) break; end
end

optimal_reward = ObtainReward(N,A,P,R,phi,theta_star);

%%Do Q-learning or Double Q-learning
for test = 1 : expt
    %%Initial value of estimators, independently sampled from [0,2].
    theta = 2*rand(d,1);
    mtheta = 2*rand(d,2);
    
    round(test,1) = 100001;
    %Train the algorithm
    i = 1;
    s = 1;
    while (i <= ti)
       epsilon = 1000 / (i + 10000); %Set step size
       
       if (is_twiced == true)
           epsilon = epsilon * 2;
       end
      
       a = randsample(A,1,true,policy(s,:)); %random exploration
       nx = randsample(N,1,true,P(s,a,:)); %Sample next state
       if (terminal(s) > 0) %If the current state is the absorbing state, we start the next episode manually
           nx = 1;
       end
            
       cur = (s - 1) * A + a;
               
       % only update for non-terminal state
       if (terminal(s) < 1)    
                %%Do Double Q-learning update. First find the optimal action of
                %%the next state, then do TD-update.
                if (is_double == false) %Do Q-Learning
                    act = find_Max(N,A,nx,phi,mtheta(:,1));

                    val =  (R(s,a) + gamma * phi((nx-1)*A+act,:) * mtheta(:,1)) - phi(cur,:) * mtheta(:,1);

                    mtheta(:,1) = mtheta(:,1)+epsilon * val * phi(cur,:)';
                    
                else
                    r = randsample(2,1,true);
                    
                    act = find_Max(N,A,nx,phi,mtheta(:,r));

                    val =  (R(s,a) + gamma * phi((nx-1)*A+act,:) * mtheta(:,3-r)) - phi(cur,:) * mtheta(:,r);

                    mtheta(:,r) = mtheta(:,r)+epsilon * val * phi(cur,:)';
                end
       end
       %%%Evaluate the current estimator
       esti = mtheta(:,1);
       if (is_averaged)
           esti = (esti + mtheta(:,2)) / 2;
       end
       
       if (mod(i,100) == 0)
           currReward = ObtainReward(N,A,P,R,phi,esti);

           if (currReward > 0.95 * optimal_reward)
               i, currReward
               round(test,1) = i;
               break;
           end

           if (mod(i,100) == 0)
               test, i
           end
       end
       s = nx;
       i = i + 1;
    end
end

save('grid3-round-D-Q-twice-averaged.txt','round','-ascii')

function Act = find_Max(N,A,s,phi,theta)%Find the best action at state s based on estimator \theta
    val = -10000; %Current maximum value
    Act = 1; %Current best action
    for i = 1 : A
        cr = phi((s - 1) * A + i,:) * theta; %%Calculate the Q-function of state s and action i
        if (cr > val) %%Update the optimal action
            Act = i;
            val = cr;
        end
    end
end

function rew = ObtainReward(N,A,P,R,phi,theta)%Find the mean reward based on theta
    rew = 0;
    for t = 1 : 100
        s = 1;
        for i = 1 : 30 %go for 30 steps
            a = find_Max(N,A,s,phi,theta);
            rew = rew + R(s,a);
            s = randsample(N,1,true,P(s,a,:)); %Sample next state
            if (s == N)
                break; %Go to a terminal state
            end
        end 
    end
    rew = rew / 10;
end