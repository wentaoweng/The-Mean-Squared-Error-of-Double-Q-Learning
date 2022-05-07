rng(5); %%fix random seed, for reproduction

%Construct a Grid instance
%N,A,d: #of states & actions & dimension of approximations
%R,P: Reward function & Transition probability
%phi: approximation basis: must be a tabular in our experiment
%policy: behaviroal policy
%Input to GenGrid: Size of the Grid
[N,A,d,R,P,phi,policy,terminal] = GenGrid(3);

cnt = 0;

path = zeros(5000000,2);
pathn = zeros(5000000,1);

ti = 100000; %Number of samples per test
expt = 100; %#of tests, where the mean-squared error is averaged

gamma = 0.9; %set discount factor

%Arrays to save mean-squared error & its standard deviations for different
%algorithms: single: Q-learning; double: Double Q-learning
%double_d: Double Q-learning with twice the learning rate
%avg_d: Double Q-learning with twice the learning rate with averaging
err_single = zeros(ti,1);
err_avg_d = zeros(ti,1);
err_double_d = zeros(ti,1);
err_double = zeros(ti,1);
std_err_single = zeros(ti,1);
std_err_avg_d = zeros(ti,1);
std_err_double_d = zeros(ti,1);
std_err_double = zeros(ti,1);

%%Q*

%%Use iteration to solve a projected Bellman Equation to get \theta^*
theta_star = zeros(d,1);
V = zeros(N,1); %Value function of each state
prb = zeros(N,1);

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

%%Do Q-learning or Double Q-learning
for test = 1 : expt
    %%Initial value of estimators, independently sampled from [0,2].
    theta = 2*rand(d,1);
    mtheta = 2*rand(d,2);
    
    backup = mtheta;
    
    %Generate a sample path of length ti. Start from state 1.
    s = 1;
    for i=1:ti    
        a = randsample(A,1,true,policy(s,:)); %Choose action
        nx = randsample(N,1,true,P(s,a,:)); %Sample next state
        if (terminal(s) > 0) %If the current state is the absorbing state, we start the next episode manually
            nx = 1;
        end
        path(i,1) = s;
        path(i,2) = a;
        pathn(i) = nx;
        s = nx;
    end
    
    %Do Double Q-learning with the same learning rate 
    i = 1;
    while (i <= ti)
       epsilon =  1000 / (i + 10000); %Set step size
       
       for j = 1 : 2
           
            r=randsample(2,1);
            
            s = path(i,1);
            a = path(i,2);
            nx = pathn(i);
            
            cur = (s - 1) * A + a;
            %%Ctar is the other estimator. 
            tar = zeros(d,1);
            for p = 1 : k
               tar = tar + mtheta(:,p);
            end
            ctar = (tar-mtheta(:,r));
             
            %%calculate mean squared-error of \theta^A. 
            sqr_loss = norm(mtheta(:,1)-theta_star,'fro')^2;
            err_double(i) = err_double(i) + (sqr_loss);
            std_err_double(i) = std_err_double(i) + (sqr_loss)^2;
               
            % only update for non-terminal state because it has no value
            % function
            if (terminal(s) < 1)
                
                %%Do Double Q-learning update. First find the optimal action of
                %%the next state, then do TD-update.
                act = find_Max(N,A,nx,phi,mtheta(:,r));

                val =  (R(s,a) + gamma * phi((nx-1)*A+act,:) * ctar) - phi(cur,:) * mtheta(:,r);

                mtheta(:,r) = mtheta(:,r)+epsilon * val * phi(cur,:)';
            end
            
            i = i + 1;
            if (i > ti) break;end
        end
    end
    
    %%%Double Q-learning with twice learning rates, the process is the same
    %%%as the last case, but with a different learning rate
    %%%Use the same initial value
    mtheta = backup;
    i = 1;
    while (i <= ti)
       epsilon =  1000 / (i + 10000); %Set step size
       
       for j = 1 : 2
            r=randsample(2,1);
            s = path(i,1);
            a = path(i,2);
            nx = pathn(i);
            
            cur = (s - 1) * A + a;
            tar = zeros(d,1);
            for p = 1 : k
               tar = tar + mtheta(:,p);
            end
            ctar = (tar-mtheta(:,r)) / (k-1);
             
            esti = (mtheta(:,1)+mtheta(:,2)) / 2;
            sqr_loss = norm(mtheta(:,1)-theta_star,'fro')^2;
            err_double_d(i) = err_double_d(i) + (sqr_loss);
            std_err_double_d(i) = std_err_double_d(i) + (sqr_loss)^2;
            
            %%calculate the mean-squared error of the averaged estimator
            sqr_loss = norm(esti - theta_star,'fro')^2;
            err_avg_d(i) = err_avg_d(i) + (sqr_loss);
            std_err_avg_d(i) = std_err_avg_d(i)+(sqr_loss)^2;
            
            % only update for non-terminal state
            if (terminal(s) < 1)
                act = find_Max(N,A,nx,phi,mtheta(:,r));

                val =  (R(s,a) + gamma * phi((nx-1)*A+act,:) * ctar) - phi(cur,:) * mtheta(:,r);

                %%Update using twice the step size
                mtheta(:,r) = mtheta(:,r)+ 2 * epsilon * val * phi(cur,:)';
            end
            
            i = i + 1;
            s = nx;
            if (i > ti) break;end
        end
    end
    
    %Q-learning updates
    for i=1: ti
       epsilon =  1000 /(i + 10000); %Set step size
        
        s = path(i,1);
        a = path(i,2);

            
        nx = pathn(i);
        cur = (s - 1) * A + a;
        %Calculate mean-squared error
        err = norm(theta - theta_star,'fro')^2;
        err_single(i) = err_single(i) + (err);
        std_err_single(i) = std_err_single(i) + (err)^2;
        
        act = find_Max(N,A,nx,phi,theta);
        
        % only update for non-terminal state
        if (terminal(s) < 1)
            val = (R(s,a) + gamma * phi((nx-1)*A+act,:) * theta - phi(cur,:) * theta);
            theta = theta + epsilon * (val * phi(cur,:)');
        end
        s = nx;
    end
end

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