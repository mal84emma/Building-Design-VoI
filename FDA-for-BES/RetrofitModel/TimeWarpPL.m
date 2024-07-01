function [fn,qn,gam] = TimeWarpPL(f,t,qnPL,mqnPL,fmeanPL,gamPL,lambda,option)
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

if nargin < 7
    lambda = 0;
    option.parallel = 0;
    option.closepool = 0;
    option.smooth = 0;
    option.sparam = 25;
    option.showplot = 0;
    option.method = 'DP';
    option.w = 0.0';
elseif nargin < 8
    option.parallel = 0;
    option.closepool = 0;
    option.smooth = 0;
    option.sparam = 25;
    option.showplot = 0;
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

%option.smooth = 1;
%option.sparam = 10;

if option.smooth == 1
    f = smooth_data(f, option.sparam);
end

if option.showplot == 1
    figure('WindowStyle','docked')
    %figure(1); 
    clf;
    plot(t, f, 'linewidth', 1);
    title('Original data', 'fontsize', 16);
    pause(0.1);
end

%% Compute the q-function of the plot
q = f_to_srvf_new(f,t);

integrand=q.*abs(q);
fo=f(M,:);
ff=cumsum(integrand);
%ff=cumtrapz(t,integrand);

for ii=1:N
    frec(:,ii)=(fo(ii))+ff(:,ii);
end

if option.showplot == 1
    figure('WindowStyle','docked');

    subplot(1,2,1);
    plot(t,f);
    %ylim([-0.2,1.2]);

    subplot(1,2,2);
    plot(t,frec);
    %ylim([-0.2,1.2]);
end
%% Calculate Warping Functions to mean building SRVF

mq = mqnPL; 
mf = fmeanPL;

load testgam;
load testmqc;
load testgamI;

mq_c=testmqc;

clear gam gam_dev;

    
    % use DP to find the optimal warping for each function w.r.t. the mean
    gam = zeros(N,size(q,1));
    gam_dev = zeros(N,size(q,1));
    for k = 1:N
        q_c = q(:,k); 
        gam(k,:) = OptimumReparamPL(mq_c,q_c,t,lambda,option.method,option.w, ...
                                   0, 0);
        gam_dev(k,:) = gradient(gam(k,:), 1/(M-1));
        f_temp(:,k) = interp1(t, f(:,k), (t(end)-t(1)).*gam(k,:) + t(1))';
    end
 
    gamI = testgamI;
    
    for k=1:N
        fn(:,k) = interp1(t, f_temp(:,k), (t(end)-t(1)).*gamI + t(1))';
        qn(:,k) = f_to_srvf_new(fn(:,k),t);
        gam(k,:) = interp1(t, gam(k,:), (t(end)-t(1)).*gamI + t(1));
    end
    
gam=gam.';

if option.parallel == 1 && option.closepool == 1
    if isempty(gcp('nocreate'))
        delete(gcp('nocreate'))
    end
end
