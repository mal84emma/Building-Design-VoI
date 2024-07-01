
clear; %close all;

ZoneScores;

%% Number of PCs

numx = 23;
numy = 25;

%% Select weekdays

Z1Wds=[xZ1Wd(:,1:numx) yZ1Wd(:,1:numy)];
Z2Wds=[xZ2Wd(:,1:numx) yZ2Wd(:,1:numy)];
Z3Wds=[xZ3Wd(:,1:numx) yZ3Wd(:,1:numy)];
Z4Wds=[xZ4Wd(:,1:numx) yZ4Wd(:,1:numy)];
Z5Wds=[xZ5Wd(:,1:numx) yZ5Wd(:,1:numy)];

%% Transform scores;  
for ii=1:numx+numy
C1(:,ii)=ksdensity(Z1Wds(:,ii),Z1Wds(:,ii),'function','cdf','Bandwidth',0.02);
C2(:,ii)=ksdensity(Z2Wds(:,ii),Z2Wds(:,ii),'function','cdf','Bandwidth',0.02);
C3(:,ii)=ksdensity(Z3Wds(:,ii),Z3Wds(:,ii),'function','cdf','Bandwidth',0.02);
C4(:,ii)=ksdensity(Z4Wds(:,ii),Z4Wds(:,ii),'function','cdf','Bandwidth',0.02);
C5(:,ii)=ksdensity(Z5Wds(:,ii),Z5Wds(:,ii),'function','cdf','Bandwidth',0.02);
end

%% Fit Gaussian Copula to transformed data

[C1yRho]=copulafit('Gaussian',C1);
[C2yRho]=copulafit('Gaussian',C2);
[C3yRho]=copulafit('Gaussian',C3);
[C4yRho]=copulafit('Gaussian',C4);
[C5yRho]=copulafit('Gaussian',C5);

%% Sample from copula

numsample = 2000;

C1rG=copularnd('Gaussian',C1yRho,numsample);
C2rG=copularnd('Gaussian',C2yRho,numsample);
C3rG=copularnd('Gaussian',C3yRho,numsample);
C4rG=copularnd('Gaussian',C4yRho,numsample);
C5rG=copularnd('Gaussian',C5yRho,numsample);

%% Transform Back

for ii=1:numx+numy
C1cG(:,ii)=ksdensity(Z1Wds(:,ii),C1rG(:,ii),'function','icdf','Bandwidth',0.02);
C2cG(:,ii)=ksdensity(Z2Wds(:,ii),C2rG(:,ii),'function','icdf','Bandwidth',0.02);
C3cG(:,ii)=ksdensity(Z3Wds(:,ii),C3rG(:,ii),'function','icdf','Bandwidth',0.02);
C4cG(:,ii)=ksdensity(Z4Wds(:,ii),C4rG(:,ii),'function','icdf','Bandwidth',0.02);
C5cG(:,ii)=ksdensity(Z5Wds(:,ii),C5rG(:,ii),'function','icdf','Bandwidth',0.02);
end

save SampleScores C1cG C2cG C3cG C4cG C5cG


%% Plot
figure('WindowStyle','docked');
subplot(1,2,1)
scatter(C1cG(:,1),C1cG(:,2),'gx')
hold on;
scatter(C2cG(:,1),C2cG(:,2),'ro')
scatter(C3cG(:,1),C3cG(:,2),'b+')
scatter(C4cG(:,1),C4cG(:,2),'cs')
scatter(C5cG(:,1),C5cG(:,2),'k.')
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
xlabel('PC X1');
ylabel('PC X2');
title('Phase Scores PC X1/PC X2')


subplot(1,2,2)
scatter(C1cG(:,24),C1cG(:,25),'gx')
hold on;
scatter(C2cG(:,24),C2cG(:,25),'ro')
scatter(C3cG(:,24),C3cG(:,25),'b+')
scatter(C4cG(:,24),C4cG(:,25),'cs')
scatter(C5cG(:,24),C5cG(:,25),'k.')
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
xlabel('PC Y1');
ylabel('PC Y2');
title('Amplitude Scores PC Y1/PC Y2')


