function [fn,qn,q0,fmean,mqn,gam,psi,stats] = TimeWarp(f,t,lambda,option)
%
% Code from https://github.com/jdtuck/fdasrvf_MATLAB
%
% input:
% f (M,N): matrix defining N functions of M samples
% y : response vector of length N
% t : time vector of length M
% lambda: regularization parameter
%
% default options
% option.parallel = 0; % turns offs MATLAB parallel processing (need
% parallel processing toolbox)
% option.closepool = 0; % determines wether to close matlabpool
% option.smooth = 0; % smooth data using standard box filter
% option.sparam = 25; % number of times to run filter
% option.showplot = 1; % turns on and off plotting
% option.method = 'DP'; % optimization method (DP, DP2, SIMUL, RBFGS)
% option.w = 0.0; % BFGS weight
%
% output
% fn: aligned functions
% qn: aligned srvfs
% q0: original srvfs
% fmean: function mean
% mqn: mean srvf
% gam: warping functions
% psi: srvf of gam
% stats: structure of statistics of alignment
addpath(genpath('DP'))

if nargin < 3
    lambda = 0;
    option.parallel = 0;
    option.closepool = 0;
    option.smooth = 0;
    option.sparam = 25;
    option.showplot = 0;
    option.showplot1 = 1;
    option.method = 'DP';
    option.w = 0.0';
elseif nargin < 4
    option.parallel = 0;
    option.closepool = 0;
    option.smooth = 0;
    option.sparam = 25;
    option.showplot = 0;
    option.showplot1 = 0;
    option.method = 'DP';
    option.w = 0.0';
end

% time warping on a set of functions
if option.parallel == 1
    if isempty(gcp('nocreate'))
        % prompt user for number threads to use
        nThreads = input('Enter number of threads to use: ');
        if nThreads > 1
            parpool(nThreads);
        elseif nThreads > 12 % check if the maximum allowable number of threads is exceeded
            while (nThreads > 12) % wait until user figures it out
                fprintf('Maximum number of threads allowed is 12\n Enter a number between 1 and 12\n');
                nThreads = input('Enter number of threads to use: ');
            end
            if nThreads > 1
                parpool(nThreads);
            end
        end
    end
end
%% Parameters

fprintf('\n lambda = %5.3f \n', lambda);

binsize = mean(diff(t));
[M, N] = size(f);
f0 = f;

if option.smooth == 1
    f = smooth_data(f, option.sparam);
end

if option.showplot == 1
    figure('WindowStyle','docked')
    clf;
    plot(t, f, 'linewidth', 1);
    title('Original data', 'fontsize', 16);
    pause(0.1);
end

%% Compute the q-function of the plot
q = f_to_srvf_new(f,t);

difftime=t-binsize/2;

integrand=q.*abs(q);
fo=f(M,:);
ff=cumsum(integrand);
%ff=cumtrapz(t,integrand);

for ii=1:N
    frec(:,ii)=(fo(ii))+ff(:,ii);
end

% Check reconstitution
if option.showplot==1
    figure('WindowStyle','docked');

    subplot(1,2,1);
    plot(t,f);
    title('Data')

    subplot(1,2,2);
    plot(difftime,frec);
    title('Reconstituted')
end

%% Set initial using the original f space
fprintf('\nInitializing...\n');
mnq = mean(q,2);
dqq = sqrt(sum((q - mnq*ones(1,N)).^2,1));
[~, min_ind] = min(dqq);
mq = q(:,min_ind); mq_i=mq; %save initial value for comparison
mf = f(:,min_ind);

gam = zeros(N,size(q,1));

for k = 1:N
    q_c = q(:,k,1); mq_c = mq;
    gam(k,:) = optimum_reparam_new(mq_c,q_c,t,lambda,option.method,option.w, ...
                               mf(M), fo(1,k,1));               
end

gamI = SqrtMeanInverse(gam);
mf = warp_f_gamma(mf,gamI,t); % just sets these values based on mean gam - is this necessary?
mq = f_to_srvf_new(mf,t);

%% Compute Mean
fprintf('Computing Karcher mean of %d functions in SRVF space...\n',N);
ds = inf;

MaxItr = 100;
q_crit = 0.001;

qun = zeros(1,MaxItr);

figure('WindowStyle','docked');
clf
plot(qun,'b');
title('Convergence plot')

if option.showplot == 1
    figure('WindowStyle','docked');
    %
    subplot(1,3,1);
    plot(t,mf,'k','LineWidth',1); hold on;
    plot(t,f(:,min_ind),'k:');
    plot(t,mean(f(:,:),2),'r:');
    xlabel('Time, t'); ylabel('Amplitude');
    str=sprintf('Mean Function');
    title(str);
    grid on;
    %
    subplot(1,3,2); hold on;
    plot(t,f(:,35),'k','LineWidth',1); hold on;
    xlabel('Time, t'); ylabel('Amplitude');
    str=sprintf('Aligned Function');
    title(str);
    grid on;
    %
    subplot(1,3,3); hold on;
    xpl=[0 1]; ypl=[0 1]; plot(xpl,ypl,'k');
    axis square;
    xlabel('Clock Time'); ylabel('Function Time');
    str=sprintf('Warping Function');
    grid on;
    title(str);
    F(1)=getframe(gcf); 
end

for r = 1:MaxItr
    fprintf('updating step: r=%d\n', r);
    if r == MaxItr
        fprintf('maximal number of iterations is reached. \n');
    end

    % Matching Step
    clear gam gam_dev;
    % use DP to find the optimal warping for each function w.r.t. the mean
    gam = zeros(N,size(q,1));
    gam_dev = zeros(N,size(q,1));
    for k = 1:N
        q_c = q(:,k,1); 
        mq_c = mq(:,r);
        gam(k,:) = optimum_reparam_new(mq_c,q_c,t,lambda,option.method,option.w, ...
                                   mf(M,r), fo(1,k,1));
        gam_dev(k,:) = gradient(gam(k,:), 1/(M-1));
        f_temp(:,k) = interp1(t, f(:,k,1), (t(end)-t(1)).*gam(k,:) + t(1))';
        q_temp(:,k) = f_to_srvf_new(f_temp(:,k),t);
    end
    q(:,:,r+1) = q_temp;
    f(:,:,r+1) = f_temp;
    gam_p(:,:,r+1)=gam;
    
    if option.showplot == 1
    subplot(1,3,1); plot(t,mf(:,r));
    subplot(1,3,2); plot(t,f_temp(:,35));
    subplot(1,3,3); plot((0:M-1)/(M-1),gam(35,:));
    F(r+1)=getframe(gcf);
    end

    ds(r+1) = sum(simps(t, (mq(:,r)*ones(1,N)-q(:,:,r+1)).^2)) + ...
        lambda*sum(simps(t, (1-sqrt(gam_dev')).^2));

    % Minimization Step
    % compute the mean of the matched function
    %mq(:,r+1) = mean(q(:,:,r+1),2);
    mf(:,r+1) = mean(f(:,:,r+1),2);
    mq(:,r+1) = f_to_srvf_new(mf(:,r+1),t);
    test_mf(:,r+1) = srvf_to_f_new(mq(:,r+1),t,mf(M,r+1));
    
    qun(r) = norm(mq(:,r+1)-mq(:,r))/norm(mq(:,r));
    testqun(1,r)=qun(r);
    
    % Plot figure
    clf;
    plot(qun,'b');
    hold on;
    x = 1:1:MaxItr;
    plot(x,q_crit*ones(1,MaxItr),'r-');
    ax = gca;
    ax.YScale = 'log';
    title('Convergence plot')
    
    F(r+1)=getframe(gcf);
    
    fprintf('\n qun = %6.4f \n', qun(r));
    
    if qun(r) < q_crit || r >= MaxItr
        break;
    end
end

if option.showplot == 1
    subplot(1,3,1); plot(t,mf(:,r+1),'r','LineWidth',1);
    subplot(1,3,2); plot(t,f(:,35,r+1),'r','LineWidth',1);
    subplot(1,3,3); plot((0:M-1)/(M-1),gam(35,:),'r','LineWidth',1);
    F(r+1)=getframe(gcf);
end

save('qun.mat','qun');

% last step with centering of gam
r = r+1;
for k = 1:N
    q_c = q(:,k,1); 
    mq_c = mq(:,r); %warp to mean calculated in previous step
    gam(k,:) = optimum_reparam_new(mq_c,q_c,t,lambda,option.method,option.w, ...
                               mf(M,r), f(1,k,1));
    gam_dev(k,:) = gradient(gam(k,:), 1/(M-1));
end
gamI = SqrtMeanInverse(gam); %Algorithm 2 - mean warping function
gamI_dev = gradient(gamI, 1/(M-1));
mq(:,r+1) = interp1(t, mq(:,r), (t(end)-t(1)).*gamI + t(1))'.*sqrt(gamI_dev'); % re-scaling?

%% Save for mapping

testmqc=mq_c;
save testmqc testmqc

testgam=gam;
save testgam testgam;

testqc=q_c;
save testqc testqc;

testgamI=gamI;
save testgamI testgamI;

%%

for k = 1:N
    f(:,k,r+1) = interp1(t, f(:,k,r), (t(end)-t(1)).*gamI + t(1))';
    q(:,k,r+1) = f_to_srvf_new(f(:,k,r+1),t);
    gam(k,:) = interp1(t, gam(k,:), (t(end)-t(1)).*gamI + t(1));
    
end

%% Aligned data & stats
fn = f(:,:,r+1);
qn = q(:,:,r+1);
q0 = q(:,:,1);
mean_f0 = mean(f0, 2);
std_f0 = std(f0, 0, 2);
mean_fn = mean(fn, 2);
std_fn = std(fn, 0, 2);
mqn = mq(:,r+1);
fmean = mean(f0(M,:))+cumsum(mqn.*abs(mqn));
fmean1 = mean(f0(M,:))+cumtrapz(t,mqn.*abs(mqn));

fgam = zeros(M,N);
for ii = 1:N
    fgam(:,ii) = interp1(t, fmean, (t(end)-t(1)).*gam(ii,:) + t(1));
end
var_fgam = var(fgam,[],2);

stats.orig_var = trapz(t,std_f0.^2);
stats.amp_var = trapz(t,std_fn.^2);
stats.phase_var = trapz(t,var_fgam);

gam = gam.';
[~,fy] = gradient(gam,binsize,binsize);
psi = sqrt(fy+eps);

if option.showplot1 == 1
    figure('WindowStyle','docked')
    plot((0:M-1)/(M-1), gam, 'linewidth', 1);
    axis square;
    title('Warping functions', 'fontsize', 16);

    figure('WindowStyle','docked')
    plot(t, fn, 'LineWidth',1);
    title(['Warped data, \lambda = ' num2str(lambda)], 'fontsize', 16);

    figure('WindowStyle','docked')
    plot(t, mean_f0, 'b-', 'linewidth', 1); hold on;
    plot(t, mean_f0+std_f0, 'r-', 'linewidth', 1);
    plot(t, mean_f0-std_f0, 'g-', 'linewidth', 1);
    title('Original data: Mean \pm STD', 'fontsize', 16);

    figure('WindowStyle','docked')
    plot(t, mean_fn, 'b-', 'linewidth', 1); hold on;
    plot(t, mean_fn+std_fn, 'r-', 'linewidth', 1);
    plot(t, mean_fn-std_fn, 'g-', 'linewidth', 1);
    title(['Warped data, \lambda = ' num2str(lambda) ': Mean \pm STD'], 'fontsize', 16);

end

if option.parallel == 1 && option.closepool == 1
    if isempty(gcp('nocreate'))
        delete(gcp('nocreate'))
    end
end
