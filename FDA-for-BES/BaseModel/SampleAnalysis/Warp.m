%% Align test data

clear; close all;

load TestDataN; 

TestData = [Z1N; Z2N; Z3N; Z4N; Z5N];

f=TestData(:,3:26)';
t=(1:1:24)';

% Align the data

[fn,qn,q0,fmean,mqn,gam,~,~] = TimeWarp(f,t,0.0);

[M, N] = size(f);

%% Reconstitute f from fn and gam:

for i=1:N
x_cr(:,i) = interp1((0:M-1)/(M-1), fn(:,i), invertGamma(gam(:,i)')');
end

save warp_dat f t fn qn q0 fmean mqn gam x_cr

%% Plot

figure('WindowStyle','docked');

subplot(2,2,1);
p1 = plot(t,f(:,1),'k','LineWidth',1); hold on;
p2 = plot(t,fn(:,1),'b:','LineWidth',1);
p3 = plot(t,x_cr(:,1),'r','LineWidth',1);
ylim([-0.1,1.1]);
legend([p1(1,1),p2(1,1),p3(1,1)],{'Data','Aligned Data','Re-constituted Data'},'Location','Northwest');
title('Sample 1')

subplot(2,2,2);
plot(t,f(:,5),'k','LineWidth',1); hold on;
plot(t,fn(:,5),'b:','LineWidth',1);
plot(t,x_cr(:,5),'r','LineWidth',1);
ylim([-0.1,1.1]);
title('Sample 2')

subplot(2,2,3);
plot(t,f(:,8),'k','LineWidth',1); hold on;
plot(t,fn(:,8),'b:','LineWidth',1);
plot(t,x_cr(:,8),'r','LineWidth',1);
ylim([-0.1,1.1]);
title('Sample 3')

subplot(2,2,4);
plot(t,f(:,14),'k','LineWidth',1); hold on;
plot(t,fn(:,14),'b:','LineWidth',1);
plot(t,x_cr(:,14),'r','LineWidth',1);
ylim([-0.1,1.1]);
title('Sample 4')


