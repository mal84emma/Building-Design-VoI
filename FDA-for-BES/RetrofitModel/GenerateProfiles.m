function [zWd_out,zWe_out] = GenerateProfiles(zWdcG,zWecG,numsample,Base,Peak,Data,N)


zWdx = zWdcG(:,1:23); zWdy = zWdcG(:,24:48); 
zWex = zWecG(:,1:23); zWey = zWecG(:,24:48);

load WarpToPL_dat;

numx=23; %No of PCs to be included - x
numy=25; %No of PCs to be included - y

binsize = mean(diff(t));

%% Calculate gamma from scores

load mu_PL;
load vec_PL;
load U_PL;

[T,~]=size(U);


vm=mean(vec);

for k=1:numsample % number of samples
    
    zWdxv = zeros(size(vm));
    zWexv = zeros(size(vm));
    
    for i=1:numx % number of principal components
       
        zWdxv = zWdxv + (zWdx(k,i)*U(:,i))';
        zWexv = zWexv + (zWex(k,i)*U(:,i))';
        
    end
     
    zWdxv=zWdxv+vm;
    vn = norm(zWdxv)/sqrt(T);
    psi = cos(vn)*mu + sin(vn)*zWdxv/vn;
    gam0 = [0 cumsum(psi.*psi)]/T;
    zWdxgam(k,:) = (gam0-gam0(1))/(gam0(end)-gam0(1));
    
    zWexv=zWexv+vm;
    vn = norm(zWexv)/sqrt(T);
    psi = cos(vn)*mu + sin(vn)*zWexv/vn;
    gam0 = [0 cumsum(psi.*psi)]/T;
    zWexgam(k,:) = (gam0-gam0(1))/(gam0(end)-gam0(1));
     
end

%% Calculate q from scores

load Uy_PL;
load meanqnew_PL;

[T,~]=size(Uy);

zWdyqs_s = zeros(T,numsample);
zWeyqs_s = zeros(T,numsample);

for i = 1:numsample
    for j=1:numy % number of principal components
        
        zWdyqs_s(:,i) = zWdyqs_s(:,i) + (zWdy(i,j)*Uy(:,j));
        zWeyqs_s(:,i) = zWeyqs_s(:,i) + (zWey(i,j)*Uy(:,j));
        
    end
end

zWdyq_s =  zWdyqs_s + meanqnew';
zWeyq_s = zWeyqs_s + meanqnew';

%%
% compute the correspondence to the original function doAC1
for k = 1:numsample
   zWdyx_s(:,k) = (sign(zWdyq_s(end,k)).*(zWdyq_s(end,k).^2))...
        +cumsum(zWdyq_s(1:end-1,k).*abs(zWdyq_s(1:end-1,k)))*binsize;
   zWeyx_s(:,k) = (sign(zWeyq_s(end,k)).*(zWeyq_s(end,k).^2))...
        +cumsum(zWeyq_s(1:end-1,k).*abs(zWeyq_s(1:end-1,k)))*binsize;
   
end


%% Combine x and y variability

zWdymx = max(zWdyx_s(:,1:numsample));
zWeymx = max(zWeyx_s(:,1:numsample));

% compute the psi-function

[M, ~] = size(f);

% combine x-variability and y-variability
clear x_c;
for k = 1:numsample
       
    zWdyx_c(:,k) = interp1((0:M-1)/(M-1), zWdyx_s(:,k), invertGamma(zWdxgam(k,:)')');
    zWeyx_c(:,k) = interp1((0:M-1)/(M-1), zWeyx_s(:,k), invertGamma(zWexgam(k,:)')');
    
end

save GeneratedProfilesWd zWdyx_c 
save GeneratedProfilesWe zWeyx_c 

%% Convert back to correct magnitude

LoadRange = Peak - Base;

zWd_s = zWdyx_c.*LoadRange + Base;
zWe_s = zWeyx_c.*LoadRange + Base;

%% With cutoff

cutoff_low = 0; % 20% difference between start/end of day

zWd_cutoff_high = 2*zWd_s(1,:); % for symmetry
zWe_cutoff_high = 2*zWe_s(1,:);

for k = 1:numsample
   
    zWd_compare_low = min(zWd_s(:,k));
    zWd_compare_high = zWd_s(end,k);
    zWd_ind(k,1) = (zWd_compare_low < cutoff_low)|(zWd_compare_high > zWd_cutoff_high(1,k));
    %
    zWe_compare_low = min(zWe_s(:,k));
    zWe_compare_high = zWe_s(end,k);
    zWe_ind(k,1) = (zWe_compare_low < cutoff_low)|(zWe_compare_high > zWe_cutoff_high(1,k));
    %
end

zWdcheck = [zWd_ind zWd_s'];
zWdchecksort = sortrows(zWdcheck,1);
if max(zWdchecksort(:,1)) > 0
    NzWd = find(zWdchecksort(:,1)>0);
    zWd_10 = zWdchecksort(1:(NzWd-1),:);
else
    zWd_10 = zWdchecksort;
end
zWd_out = zWd_10(:,2:25)';

%
zWecheck = [zWe_ind zWe_s'];
zWechecksort = sortrows(zWecheck,1);
if max(zWechecksort(:,1)) > 0
    NzWe = find(zWechecksort(:,1)>0);
    zWe_10 = zWechecksort(1:(NzWe-1),:);
else
    zWe_10 = zWechecksort;
end
zWe_out = zWe_10(:,2:25)';

save GeneratedDemand zWd_out zWe_out

%% Plot comparison

% Weekday
ZWd=Data(1:N,3:26);

% Weekend
ZWe=Data(N+1:end,3:26);


figure('WindowStyle','docked');
t = 1:1:24;

subplot(1,2,1)
p1 = plot(t,ZWd,'k:');
hold on;
plotQ(ZWd','r');
plotQD(zWd_out,'b');
t1 = (ZWd - Base)/LoadRange;
t2 = (zWd_out - Base)/LoadRange;
kld_n = gau_kl(mean(t1),std(t1),mean(t2'),std(t2'));
str=sprintf('Weekday: K-L %1.1f',kld_n);
title(str,'FontSize',14)
yl = ylim;
%
subplot(1,2,2)
p1 = plot(t,ZWe,'k:');
hold on;
plotQ(ZWe','r');
plotQD(zWe_out,'b');
t1 = (ZWe - Base)/LoadRange;
t2 = (zWe_out - Base)/LoadRange;
kld_n = gau_kl(mean(t1),std(t1),mean(t2'),std(t2'));
str=sprintf('Weekend: K-L %1.1f',kld_n);
title(str,'FontSize',14)
ylim(yl);
%
