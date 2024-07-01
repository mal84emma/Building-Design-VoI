clear; %close all;

load SampleScores;

Z1x = C1cG(:,1:23); Z1y = C1cG(:,24:48);
Z2x = C2cG(:,1:23); Z2y = C2cG(:,24:48);
Z3x = C3cG(:,1:23); Z3y = C3cG(:,24:48);
Z4x = C4cG(:,1:23); Z4y = C4cG(:,24:48);
Z5x = C5cG(:,1:23); Z5y = C5cG(:,24:48);

load warp_dat;

numx=23; %No of PCs to be included - x
numy=25; %No of PCs to be included - y

numsample=size(C1cG,1); %
binsize = mean(diff(t));

%% Calculate gamma from scores

load mu;
load vec;
load U;

[T,~]=size(U);

vm=mean(vec);

for k=1:numsample % number of samples
    
    Z1xv = zeros(size(vm));
    Z2xv = zeros(size(vm));
    Z3xv = zeros(size(vm));
    Z4xv = zeros(size(vm));
    Z5xv = zeros(size(vm));
    
    for i=1:numx % number of principal components
       
        Z1xv = Z1xv + (Z1x(k,i)*U(:,i))';
        Z2xv = Z2xv + (Z2x(k,i)*U(:,i))';
        Z3xv = Z3xv + (Z3x(k,i)*U(:,i))';
        Z4xv = Z4xv + (Z4x(k,i)*U(:,i))';
        Z5xv = Z5xv + (Z5x(k,i)*U(:,i))';
        
    end
     
    Z1xv=Z1xv+vm;
    vn = norm(Z1xv)/sqrt(T);
    psi = cos(vn)*mu + sin(vn)*Z1xv/vn;
    gam0 = [0 cumsum(psi.*psi)]/T;
    Z1xgam(k,:) = (gam0-gam0(1))/(gam0(end)-gam0(1));
    
    Z2xv=Z2xv+vm;
    vn = norm(Z2xv)/sqrt(T);
    psi = cos(vn)*mu + sin(vn)*Z2xv/vn;
    gam0 = [0 cumsum(psi.*psi)]/T;
    Z2xgam(k,:) = (gam0-gam0(1))/(gam0(end)-gam0(1));
    
    Z3xv=Z3xv+vm;
    vn = norm(Z3xv)/sqrt(T);
    psi = cos(vn)*mu + sin(vn)*Z3xv/vn;
    gam0 = [0 cumsum(psi.*psi)]/T;
    Z3xgam(k,:) = (gam0-gam0(1))/(gam0(end)-gam0(1));
    
    Z4xv=Z4xv+vm;
    vn = norm(Z4xv)/sqrt(T);
    psi = cos(vn)*mu + sin(vn)*Z4xv/vn;
    gam0 = [0 cumsum(psi.*psi)]/T;
    Z4xgam(k,:) = (gam0-gam0(1))/(gam0(end)-gam0(1));
    
    Z5xv=Z5xv+vm;
    vn = norm(Z5xv)/sqrt(T);
    psi = cos(vn)*mu + sin(vn)*Z5xv/vn;
    gam0 = [0 cumsum(psi.*psi)]/T;
    Z5xgam(k,:) = (gam0-gam0(1))/(gam0(end)-gam0(1));
     
end

%% Calculate q from scores

load Uy;
load meanqnew;

[T,~]=size(Uy);

Z1yqs_s = zeros(T,numsample);
Z2yqs_s = zeros(T,numsample);
Z3yqs_s = zeros(T,numsample);
Z4yqs_s = zeros(T,numsample);
Z5yqs_s = zeros(T,numsample);

for i = 1:numsample
    for j=1:numy % number of principal components
        
        Z1yqs_s(:,i) = Z1yqs_s(:,i) + (Z1y(i,j)*Uy(:,j));
        Z2yqs_s(:,i) = Z2yqs_s(:,i) + (Z2y(i,j)*Uy(:,j));
        Z3yqs_s(:,i) = Z3yqs_s(:,i) + (Z3y(i,j)*Uy(:,j));
        Z4yqs_s(:,i) = Z4yqs_s(:,i) + (Z4y(i,j)*Uy(:,j));
        Z5yqs_s(:,i) = Z5yqs_s(:,i) + (Z5y(i,j)*Uy(:,j));
        
    end
end

Z1yq_s = Z1yqs_s + meanqnew';
Z2yq_s = Z2yqs_s + meanqnew';
Z3yq_s = Z3yqs_s + meanqnew';
Z4yq_s = Z4yqs_s + meanqnew';
Z5yq_s = Z5yqs_s + meanqnew';

%%
% compute the correspondence to the original function doAC1
for k = 1:numsample
   Z1yx_s(:,k) = (sign(Z1yq_s(end,k)).*(Z1yq_s(end,k).^2))...
        +cumsum(Z1yq_s(1:end-1,k).*abs(Z1yq_s(1:end-1,k)))*binsize;
   Z2yx_s(:,k) = (sign(Z2yq_s(end,k)).*(Z2yq_s(end,k).^2))...
        +cumsum(Z2yq_s(1:end-1,k).*abs(Z2yq_s(1:end-1,k)))*binsize;
   Z3yx_s(:,k) = (sign(Z3yq_s(end,k)).*(Z3yq_s(end,k).^2))...
        +cumsum(Z3yq_s(1:end-1,k).*abs(Z3yq_s(1:end-1,k)))*binsize;
   Z4yx_s(:,k) = (sign(Z4yq_s(end,k)).*(Z4yq_s(end,k).^2))...
        +cumsum(Z4yq_s(1:end-1,k).*abs(Z4yq_s(1:end-1,k)))*binsize;
   Z5yx_s(:,k) = (sign(Z5yq_s(end,k)).*(Z5yq_s(end,k).^2))...
        +cumsum(Z5yq_s(1:end-1,k).*abs(Z5yq_s(1:end-1,k)))*binsize;
     
end


%% Combine x and y variability

% compute the psi-function

[M, N] = size(f);

% combine x-variability and y-variability

for k = 1:numsample
       
    Z1yx_c(:,k) = interp1((0:M-1)/(M-1), Z1yx_s(:,k), invertGamma(Z1xgam(k,:)')');
    Z2yx_c(:,k) = interp1((0:M-1)/(M-1), Z2yx_s(:,k), invertGamma(Z2xgam(k,:)')');
    Z3yx_c(:,k) = interp1((0:M-1)/(M-1), Z3yx_s(:,k), invertGamma(Z3xgam(k,:)')');
    Z4yx_c(:,k) = interp1((0:M-1)/(M-1), Z4yx_s(:,k), invertGamma(Z4xgam(k,:)')');
    Z5yx_c(:,k) = interp1((0:M-1)/(M-1), Z5yx_s(:,k), invertGamma(Z5xgam(k,:)')');
    
end

save GeneratedProfiles Z1yx_c Z2yx_c Z3yx_c Z4yx_c Z5yx_c 

%% Convert back to correct magnitude

load TestData;
load BaseLoads; 
load PeakLoads; 

LoadRange = PeakLoads - BaseLoads;

Z1_s = Z1yx_c.*LoadRange(1) + BaseLoads(1);
Z2_s = Z2yx_c.*LoadRange(2) + BaseLoads(2);
Z3_s = Z3yx_c.*LoadRange(3) + BaseLoads(3);
Z4_s = Z4yx_c.*LoadRange(4) + BaseLoads(4);
Z5_s = Z5yx_c.*LoadRange(5) + BaseLoads(5);

%% With cutoff

cutoff_low = 0; 

Z1_cutoff_high = 2*Z1_s(1,:);
Z2_cutoff_high = 2*Z2_s(1,:);
Z3_cutoff_high = 2*Z3_s(1,:);
Z4_cutoff_high = 2*Z4_s(1,:);
Z5_cutoff_high = 2*Z5_s(1,:);

for k = 1:numsample
   
    Z1_compare_low = min(Z1_s(:,k));
    Z1_compare_high = Z1_s(end,k);
    Z1_ind(k,1) = (Z1_compare_low < cutoff_low)|(Z1_compare_high > Z1_cutoff_high(1,k));
    
    Z2_compare = min(Z2_s(:,k));
    Z2_compare_high = Z2_s(end,k);
    Z2_ind(k,1) = (Z2_compare < cutoff_low)|(Z2_compare_high > Z2_cutoff_high(1,k));
    
    Z3_compare = min(Z3_s(:,k));
    Z3_compare_high = Z3_s(end,k);
    Z3_ind(k,1) = (Z3_compare < cutoff_low)|(Z3_compare_high > Z3_cutoff_high(1,k));
    
    Z4_compare = min(Z4_s(:,k));
    Z4_compare_high = Z4_s(end,k);
    Z4_ind(k,1) = (Z4_compare < cutoff_low)|(Z4_compare_high > Z4_cutoff_high(1,k));
    
    Z5_compare = min(Z5_s(:,k));
    Z5_compare_high = Z5_s(end,k);
    Z5_ind(k,1) = (Z5_compare < cutoff_low)|(Z5_compare_high > Z5_cutoff_high(1,k));
    
end

Z1check = [Z1_ind Z1_s'];
Z1checksort = sortrows(Z1check,1);
if max(Z1checksort(:,1)) > 0
    NEZ1 = find(Z1checksort(:,1)>0);
    Z1_10 = Z1checksort(1:(NEZ1-1),:);
else
    Z1_10 = Z1checksort;
end
Z1_plot = Z1_10(:,2:25)';

Z2check = [Z2_ind Z2_s'];
Z2checksort = sortrows(Z2check,1);
if max(Z2checksort(:,1)) > 0
    NEZ2 = find(Z2checksort(:,1)>0);
    Z2_10 = Z2checksort(1:(NEZ2(1)-1),:);
else
    Z2_10 = Z2checksort;
end
Z2_plot = Z2_10(:,2:25)';

Z3check = [Z3_ind Z3_s'];
Z3checksort = sortrows(Z3check,1);
if max(Z3checksort(:,1)) > 0
    NEZ3 = find(Z3checksort(:,1)>0);
    Z3_10 = Z3checksort(1:(NEZ3-1),:);
else
    Z3_10 = Z3checksort;
end
Z3_plot = Z3_10(:,2:25)';

Z4check = [Z4_ind Z4_s'];
Z4checksort = sortrows(Z4check,1);
if max(Z4checksort(:,1)) > 0
    NEZ4 = find(Z4checksort(:,1)>0);
    Z4_10 = Z4checksort(1:(NEZ4-1),:);
 else
    Z4_10 = Z4checksort;
end   
Z4_plot = Z4_10(:,2:25)';
    
Z5check = [Z5_ind Z5_s'];
Z5checksort = sortrows(Z5check,1);
if max(Z5checksort(:,1)) > 0
    NEZ5 = find(Z5checksort(:,1)>0);
    Z5_10 = Z5checksort(1:(NEZ5-1),:);
    else
    Z5_10 = Z5checksort;
end
Z5_plot = Z5_10(:,2:25)';

save GeneratedDemand Z1_plot Z2_plot Z3_plot Z4_plot Z5_plot;

%% Plot quantiles

figure('WindowStyle','docked');

numsample = 1000;
NZ = 261;
tt=1:1:24;
tt2=[tt';flipud(tt')];
q1=0.95;
q2=0.05;

subplot(1,5,1);
curve1=quantile(Z1_plot(:,1:numsample),q1,2);
curve2=quantile(Z1_plot(:,1:numsample),q2,2);
Z1SimMean=mean(Z1_plot(:,1:numsample),2);
inBetween=[curve1;flipud(curve2)];
curve3=quantile(Z1(1:NZ,3:26)',q1,2);
curve4=quantile(Z1(1:NZ,3:26)',q2,2);
DataMean=mean(Z1(1:NZ,3:26)',2);
inBetween2=[curve3;flipud(curve4)];
p21=plot(tt,curve1,'b');
hold on;
pSMean=plot(tt,Z1SimMean,'b','LineWidth',2);
p22=plot(tt,curve2,'b');
fill(tt2,inBetween,'b','FaceAlpha',0.3);
p23=plot(tt,curve3,'r');
p24=plot(tt,curve4,'r');
pDMean=plot(tt,DataMean,'r','LineWidth',2);
fill(tt2,inBetween2,'r','FaceAlpha',0.3);
str=sprintf('Z1');
xlim([0,24]);
ylim([0,20]);
ax=gca;
ax.XTick = [0 6 12 18 24];
title(str,'FontSize',14);

subplot(1,5,2);
curve1=quantile(Z2_plot(:,1:numsample),q1,2);
curve2=quantile(Z2_plot(:,1:numsample),q2,2);
Z2SimMean=mean(Z2_plot(:,1:numsample),2);
inBetween=[curve1;flipud(curve2)];
curve3=quantile(Z2(1:NZ,3:26)',q1,2);
curve4=quantile(Z2(1:NZ,3:26)',q2,2);
DataMean=mean(Z2(1:NZ,3:26)',2);
inBetween2=[curve3;flipud(curve4)];
p21=plot(tt,curve1,'b');
hold on;
pSMean=plot(tt,Z2SimMean,'b','LineWidth',2);
p22=plot(tt,curve2,'b');
fill(tt2,inBetween,'b','FaceAlpha',0.3);
p23=plot(tt,curve3,'r');
p24=plot(tt,curve4,'r');
pDMean=plot(tt,DataMean,'r','LineWidth',2);
fill(tt2,inBetween2,'r','FaceAlpha',0.3);
str=sprintf('Z2');
xlim([0,24]);
ylim([0,20]);
ax=gca;
ax.XTick = [0 6 12 18 24];
title(str,'FontSize',14);

subplot(1,5,3);
curve1=quantile(Z3_plot(:,1:numsample),q1,2);
curve2=quantile(Z3_plot(:,1:numsample),q2,2);
Z3SimMean=mean(Z3_plot(:,1:numsample),2);
inBetween=[curve1;flipud(curve2)];
curve3=quantile(Z3(1:NZ,3:26)',q1,2);
curve4=quantile(Z3(1:NZ,3:26)',q2,2);
DataMean=mean(Z3(1:NZ,3:26)',2);
inBetween2=[curve3;flipud(curve4)];
p21=plot(tt,curve1,'b');
hold on;
pSMean=plot(tt,Z3SimMean,'b','LineWidth',2);
p22=plot(tt,curve2,'b');
fill(tt2,inBetween,'b','FaceAlpha',0.3);
p23=plot(tt,curve3,'r');
p24=plot(tt,curve4,'r');
pDMean=plot(tt,DataMean,'r','LineWidth',2);
fill(tt2,inBetween2,'r','FaceAlpha',0.3);
str=sprintf('Z3');
xlim([0,24]);
ylim([0,20]);
ax=gca;
ax.XTick = [0 6 12 18 24];
title(str,'FontSize',14);

subplot(1,5,4);
curve1=quantile(Z4_plot(:,1:numsample),q1,2);
curve2=quantile(Z4_plot(:,1:numsample),q2,2);
Z4SimMean=mean(Z4_plot(:,1:numsample),2);
inBetween=[curve1;flipud(curve2)];
curve3=quantile(Z4(1:NZ,3:26)',q1,2);
curve4=quantile(Z4(1:NZ,3:26)',q2,2);
DataMean=mean(Z4(1:NZ,3:26)',2);
inBetween2=[curve3;flipud(curve4)];
p21=plot(tt,curve1,'b');
hold on;
pSMean=plot(tt,Z4SimMean,'b','LineWidth',2);
p22=plot(tt,curve2,'b');
fill(tt2,inBetween,'b','FaceAlpha',0.3);
p23=plot(tt,curve3,'r');
p24=plot(tt,curve4,'r');
pDMean=plot(tt,DataMean,'r','LineWidth',2);
fill(tt2,inBetween2,'r','FaceAlpha',0.3);
str=sprintf('Z4');
xlim([0,24]);
ylim([0,100]);
ax=gca;
ax.XTick = [0 6 12 18 24];
title(str,'FontSize',14);

subplot(1,5,5);
curve1=quantile(Z5_plot(:,1:numsample),q1,2);
curve2=quantile(Z5_plot(:,1:numsample),q2,2);
Z5SimMean=mean(Z5_plot(:,1:numsample),2);
inBetween=[curve1;flipud(curve2)];
curve3=quantile(Z5(1:NZ,3:26)',q1,2);
curve4=quantile(Z5(1:NZ,3:26)',q2,2);
DataMean=mean(Z5(1:NZ,3:26)',2);
inBetween2=[curve3;flipud(curve4)];
p21=plot(tt,curve1,'b');
hold on;
pSMean=plot(tt,Z5SimMean,'b','LineWidth',2);
p22=plot(tt,curve2,'b');
pf1 = fill(tt2,inBetween,'b','FaceAlpha',0.3);
p23=plot(tt,curve3,'r');
p24=plot(tt,curve4,'r');
pDMean=plot(tt,DataMean,'r','LineWidth',2);
pf2 = fill(tt2,inBetween2,'r','FaceAlpha',0.3);
str=sprintf('Z5');
xlim([0,24]);
ylim([0,20]);
ax=gca;
ax.XTick = [0 6 12 18 24];
title(str,'FontSize',14);

legend([pDMean(1,1),pSMean(1,1),pf1(1,1),pf2(1,1)],{'Data Mean','Simulation Mean','Data 90% CL','Simulation 90% CL'});

csvwrite('Z1_plot.csv',Z1_plot(:,1:numsample)');
csvwrite('Z2_plot.csv',Z2_plot(:,1:numsample)');
csvwrite('Z3_plot.csv',Z3_plot(:,1:numsample)');
csvwrite('Z4_plot.csv',Z4_plot(:,1:numsample)');
csvwrite('Z5_plot.csv',Z5_plot(:,1:numsample)');

%% Write out scores

Z1checks = [Z1_ind C1cG];
Z1checksort = sortrows(Z1checks,1);
if max(Z1checksort(:,1)) > 0
    NEZ1 = find(Z1checksort(:,1)>0);
    Z1_10s = Z1checksort(1:(NEZ1-1),:);
else
    Z1_10s = Z1checksort;
end
Z1_plot_s = Z1_10s(:,2:49)';

Z2checks = [Z2_ind C2cG];
Z2checksort = sortrows(Z2checks,1);
if max(Z2checksort(:,1)) > 0
    NEZ2 = find(Z2checksort(:,1)>0);
    Z2_10s = Z2checksort(1:(NEZ2(1)-1),:);
else
    Z2_10s = Z2checksort;
end
Z2_plot_s = Z2_10s(:,2:49)';

Z3checks = [Z3_ind C3cG];
Z3checksort = sortrows(Z3checks,1);
if max(Z3checksort(:,1)) > 0
    NEZ3 = find(Z3checksort(:,1)>0);
    Z3_10s = Z3checksort(1:(NEZ3-1),:);
else
    Z3_10s = Z3checksort;
end
Z3_plot_s = Z3_10s(:,2:49)';

Z4checks = [Z4_ind C4cG];
Z4checksort = sortrows(Z4checks,1);
if max(Z4checksort(:,1)) > 0
    NEZ4 = find(Z4checksort(:,1)>0);
    Z4_10s = Z4checksort(1:(NEZ4-1),:);
 else
    Z4_10s = Z4checksort;
end   
Z4_plot_s = Z4_10s(:,2:49)';
    
Z5checks = [Z5_ind C5cG];
Z5checksort = sortrows(Z5checks,1);
if max(Z5checksort(:,1)) > 0
    NEZ5 = find(Z5checksort(:,1)>0);
    Z5_10s = Z5checksort(1:(NEZ5-1),:);
    else
    Z5_10s = Z5checksort;
end
Z5_plot_s = Z5_10s(:,2:25)';

save scores_cutoff Z1_plot_s Z2_plot_s Z3_plot_s Z4_plot_s Z5_plot_s;
