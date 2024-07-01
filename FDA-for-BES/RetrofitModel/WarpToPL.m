function [] = WarpToPL(TestData)

% Warp new data to calculated mean 
%
load warp_dat_PL; % f t fn qn q0 fmean mqn gam x_cr calculated for Plug Loads

fPL=f;
fnPL=fn;
qnPL=qn;
fmeanPL=fmean;
mqnPL=mqn;
gamPL=gam';
x_cr_PL=x_cr;

f = TestData(:,3:26)';
t = (1:24)';

% Align the data
[fn,qn,gam] = TimeWarpPL(f,t,qnPL,mqnPL,fmeanPL,gamPL,0.0);

[M, N] = size(f);

integrand=qn.*abs(qn);
ff=cumsum(integrand);
fno=fn(end,:);

for ii=1:N
    frec(:,ii)=fno(ii)+ff(:,ii);
end

%% Reconstitute f from fn and gam:

for i=1:N
x_cr(:,i) = interp1((0:M-1)/(M-1), fn(:,i), invertGamma(gam(:,i)')');
end

save WarpToPL_dat f t fn qn q0 fmean mqn gam x_cr

%% Plot

figure('WindowStyle','docked');

subplot(2,2,1);
plot(t,f(:,1),'k','LineWidth',1); hold on;
plot(t,fn(:,1),'b:','LineWidth',1);
plot(t,x_cr(:,1),'r','LineWidth',1);
legend({'Data','Aligned','Reconstituted'},'Location','Northwest');
ylim([-0.1,1.5]);
title('Sample 1')

subplot(2,2,2);
plot(t,f(:,3),'k','LineWidth',1); hold on;
plot(t,fn(:,3),'b:','LineWidth',1);
plot(t,x_cr(:,3),'r','LineWidth',1);
ylim([-0.1,1.5]);
title('Sample 2')

subplot(2,2,3);
plot(t,f(:,5),'k','LineWidth',1); hold on;
plot(t,fn(:,5),'b:','LineWidth',1);
plot(t,x_cr(:,5),'r','LineWidth',1);
ylim([-0.1,1.5]);
title('Sample 3')

subplot(2,2,4);
plot(t,f(:,6),'k','LineWidth',1); hold on;
plot(t,fn(:,6),'b:','LineWidth',1);
plot(t,x_cr(:,6),'r','LineWidth',1);
ylim([-0.1,1.5]);
title('Sample 4')

