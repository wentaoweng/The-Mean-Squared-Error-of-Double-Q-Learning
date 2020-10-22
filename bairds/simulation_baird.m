rng(5); %%fix random seed, for reproduction

%Construct a Baird's instance
%N,A,d: #of states & actions & dimension of approximations
%R,P: Reward function & Transition probability
%phi: approximation basis
%policy: behaviroal policy
%Input to GenBaird: type=0, Zero Reward; type=1, Small Random Reward; type=2, Large Random Reward
[N,A,d,R,P,phi,policy] = GenBaird(2); 

cnt = 0;
%Sample Paths
path = zeros(5000000,2);
pathn = zeros(5000000,1);

ti = 20000;  %Number of samples per test
expt = 100; %#of tests, where the mean-squared error is averaged

gamma = 0.8; %set discount factor

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
std_err_avg = zeros(ti,1);
std_err_double_d = zeros(ti,1);
std_err_double = zeros(ti,1);


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
                %Calcuate Target value function
                tar = R(i,j) + gamma * prb' * V;
                %label of the pair of state and action (i,j)
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
        path(i,1) = s; %Save current state
        path(i,2) = a; %Save current action
        pathn(i) = nx; %Save next state
        s = nx; %Markov transition
    end
    
    %Do Double Q-learning with the same learning rate 
    i = 1;
    while (i <= ti)
       epsilon =  1000 / (i + 10000); %Set step size
       
       for j = 1 : 2
            r=randsample(2,1); %%Randomly update one estimator
            
            s = path(i,1);
            a = path(i,2); 
            nx = pathn(i);
            
            cur = (s - 1) * A + a;
            
            %%Ctar is the other estimator. 
            tar = zeros(d,1);
            for p = 1 : 2
               tar = tar + mtheta(:,p);
            end
            ctar = (tar-mtheta(:,r));
             
            
            %%calculate mean squared-error of \theta^A. 
            sqr_loss = norm(mtheta(:,1)-theta_star,'fro')^2;
            err_double(i) = err_double(i) + (sqr_loss);
            std_err_double(i) = std_err_double(i) + (sqr_loss)^2;
            
            %%Do Double Q-learning update. First find the optimal action of
            %%the next state, then do TD-update.
            act = find_Max(N,A,nx,phi,mtheta(:,r));
            %%Target value
            val =  (R(s,a) + gamma * phi((nx-1)*A+act,:) * ctar) - phi(cur,:) * mtheta(:,r);
            %%Update current estimator
            mtheta(:,r) = mtheta(:,r)+epsilon * val * phi(cur,:)';
            %%Print the mean-squared error every 10000 samples
            if (mod(i,10000) == 0)    
                fprintf('%d %e\n',i, sqr_loss);
            end
            %%Go to next state
            i = i + 1;
            s = nx;
            %%If finish one sample path, break
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
            for p = 1 : 2
               tar = tar + mtheta(:,p);
            end
            ctar = (tar-mtheta(:,r));
             
            sqr_loss = norm(mtheta(:,1) - theta_star,'fro')^2;
            err_double_d(i) = err_double_d(i) + (sqr_loss);
            std_err_double_d(i) = std_err_double_d(i) + (sqr_loss)^2;
            
            %%calculate the mean-squared error of the averaged estimator
            esti = (mtheta(:,1)+mtheta(:,2)) / 2;
            sqr_loss = norm(esti - theta_star,'fro')^2;
            err_avg_d(i) = err_avg_d(i) + (sqr_loss);
            std_err_avg_d(i) = std_err_avg_d(i)+(sqr_loss)^2;
            
            act = find_Max(N,A,nx,phi,mtheta(:,r));

            val =  (R(s,a) + gamma * phi((nx-1)*A+act,:) * ctar) - phi(cur,:) * mtheta(:,r);
            
            %%Update using twice the step size
            mtheta(:,r) = mtheta(:,r)+ 2 * epsilon * val * phi(cur,:)';
                
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
        
        %Q-learning update
        val = (R(s,a) + gamma * phi((nx-1)*A+act,:) * theta - phi(cur,:) * theta);
        theta = theta + epsilon * (val * phi(cur,:)');
    end
end

%%%Calcuate standard deviation, and save results to files

err_double = err_double / expt;
save('Baird=type2-errDouble.txt','err_double','-ascii')
std_err_double = (std_err_double - expt * (err_double .^ 2)) / (expt - 1);
std_err_double = std_err_double .^0.5 / expt^0.5;
save('Baird=type2-stderrDouble.txt','std_err_double','-ascii')

err_double_d = err_double_d / expt;
save('Baird=type2-errDouble_d.txt','err_double_d','-ascii')
std_err_double_d = (std_err_double_d - expt * (err_double_d .^ 2)) / (expt - 1);
std_err_double_d = std_err_double_d .^0.5 / expt^0.5;
save('Baird=type2-stderrDouble_d.txt','std_err_double_d','-ascii')

err_single = err_single / expt;
save('Baird=type2-errsingle.txt','err_single','-ascii')
std_err_single = (std_err_single - expt * (err_single .^ 2)) / (expt - 1);
std_err_single = std_err_single .^0.5 / expt^0.5;
save('Baird=type2-stderrsingle.txt','std_err_single','-ascii')

err_avg_d = err_avg_d / expt;
save('Baird=type2-erravg_d.txt','err_avg_d','-ascii')
std_err_avg_d = (std_err_avg_d - expt * (err_avg_d .^ 2)) / (expt - 1);
std_err_avg_d = std_err_avg_d .^0.5 / expt^0.5;
save('Baird=type2-stderravg_d.txt','std_err_avg_d','-ascii')

function Act = find_Max(N,A,s,phi,theta) %Find the best action at state s based on estimator \theta
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