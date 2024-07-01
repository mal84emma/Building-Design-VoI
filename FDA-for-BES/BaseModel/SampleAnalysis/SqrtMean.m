function [mu,gam_mu,psi,vec] = SqrtMean(gam)
%
% Code from https://github.com/jdtuck/fdasrvf_MATLAB
%
 
[n,T] = size(gam); % gam 24 x 505, transposed on entry to routine 
dT = 1/(T-1);
psi = zeros(n,T-1);
for i=1:n
    psi(i,:) = sqrt(diff(gam(i,:))/dT+eps);
    %psi(i,:) = sqrt(diff(gam(i,:))+eps);
end

%Find direction
mnpsi = mean(psi);
dqq = sqrt(sum((psi' - mnpsi'*ones(1,n)).^2,1));
[~, min_ind] = min(dqq);
mu = psi(min_ind,:);
t = 1;
maxiter = 500;
lvm = zeros(1,maxiter);
vec = zeros(n,T-1);
for iter = 1:maxiter
    for i=1:n
        v = psi(i,:) - mu;
        dot1 = simps(linspace(0,1,T-1),mu.*psi(i,:));
        if dot1 > 1
            dot_limited = 1;
        elseif dot1 < -1
            dot_limited = -1;
        else
            dot_limited = dot1;
        end
        len = acos(dot_limited);
        if len > 0.0001
            vec(i,:) = (len/sin(len))*(psi(i,:) - cos(len)*mu);
        else
            vec(i,:) = zeros(1,T-1);
        end
    end
    
    vm = mean(vec);
    lvm(iter) = sqrt(sum(vm.*vm)*dT);
    mu = cos(t*lvm(iter))*mu + (sin(t*lvm(iter))/lvm(iter))*vm;
    
    if lvm(iter) < 1e-7 || iter > maxiter
        fprintf('\n lvm = %10.8f \n', lvm(iter));
        fprintf('\n iter = %3.0f \n', iter);
        break
    end
end

for i=1:n
    phi(i,:) = cumsum([0 psi(i,:).*psi(i,:)/T]);
end

gam_mu = [0 cumsum(mu.*mu)]/T;
gam_mu = (gam_mu-min(gam_mu))/(max(gam_mu)-min(gam_mu));

%% Check does vec transform back to gam?

for i=1:n
    v=vec(i,:);
    vn = norm(v)/sqrt(T);
    psi = cos(vn)*mu + sin(vn)*v/vn;
    gam0 = [0 cumsum(psi.*psi/T)];
    testgam(i,:) = (gam0-gam0(1))/(gam0(end)-gam0(1));
end

%% Plot check if needed

%figure('WindowStyle','docked');

%pt=size(testgam,2)-1;

%plot((0:pt)/(pt), testgam(3,:),'r', 'linewidth', 1);
%hold on;
%plot((0:pt)/(pt), gam(3,:),'b', 'linewidth', 1);


