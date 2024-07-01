clear; close all;

t = 1:1:24;

%%
load('Z1HSortP.mat')
N = find(Z1HSortP(:,2)>5,1) - 1;
[Z1PD, Z1PNMean, Z1PNMedian, Z1Base, Z1Peak] = Normalise(Z1HSortP,N);
save Z1PDZ1PD;
save Z1PNMean Z1PNMean;
save Z1PNMedian Z1PNMedian;

figure('WindowStyle','docked');
subplot(2,3,1)
plot(t,Z1PD(:,3:26),'k')
title('Z1 Diversity')
subplot(2,3,2);
plot(t,Z1PNMean(:,3:26),'r')
title('Z1 Mean')
subplot(2,3,3);
plot(t,Z1PNMedian(:,3:26),'b')
title('Z1 Median')
subplot(2,2,3);
histogram(Z1Peak,'FaceColor','b','Normalization','Probability','binwidth',0.5);
title(['Peak, Mean ',num2str(mean(Z1Peak),2),' ,Median ',num2str(median(Z1Peak),2)])
subplot(2,2,4);
histogram(Z1Base,'FaceColor','b','Normalization','Probability','binwidth',0.25);
title(['Base, Mean ',num2str(mean(Z1Base),2),' ,Median ',num2str(median(Z1Base),2)])

%%
load('Z2HSortP.mat')
N = find(Z2HSortP(:,2)>5,1) - 1;
[Z2PD, Z2PNMean, Z2PNMedian, Z2Base, Z2Peak] = Normalise(Z2HSortP,N);
save Z2PD Z2PD;
save Z2PNMean Z2PNMean;
save Z2PNMedian Z2PNMedian;

figure('WindowStyle','docked');
subplot(2,3,1)
plot(t,Z2PD(:,3:26),'k')
title('Z2 Diversity')
subplot(2,3,2);
plot(t,Z2PNMean(:,3:26),'r')
title('Z2 Mean')
subplot(2,3,3);
plot(t,Z2PNMedian(:,3:26),'b')
title('Z2 Median')
subplot(2,2,3);
histogram(Z2Peak,'FaceColor','b','Normalization','Probability','binwidth',0.5);
title(['Peak, Mean ',num2str(mean(Z2Peak),2),' ,Median ',num2str(median(Z2Peak),2)])
subplot(2,2,4);
histogram(Z2Base,'FaceColor','b','Normalization','Probability','binwidth',0.25);
title(['Base, Mean ',num2str(mean(Z2Base),2),' ,Median ',num2str(median(Z2Base),2)])

%%
load('Z3HSortP.mat')
N = find(Z3HSortP(:,2)>5,1) - 1;
[Z3PD, Z3PNMean,Z3PNMedian, Z3Base, Z3Peak] = Normalise(Z3HSortP,N);
save Z3PD Z3PD;
save Z3PNMean Z3PNMean;
save Z3PNMedian Z3PNMedian;

figure('WindowStyle','docked');
subplot(2,3,1)
plot(t,Z3PD(:,3:26),'k')
title('Z3 Diversity')
subplot(2,3,2);
plot(t,Z3PNMean(:,3:26),'r')
title('Z3 Mean')
subplot(2,3,3);
plot(t,Z3PNMedian(:,3:26),'b')
title('Z3 Median')
subplot(2,2,3);
histogram(Z3Peak,'FaceColor','b','Normalization','Probability','binwidth',0.5);
title(['Peak, Mean ',num2str(mean(Z3Peak),2),' ,Median ',num2str(median(Z3Peak),2)])
subplot(2,2,4);
histogram(Z3Base,'FaceColor','b','Normalization','Probability','binwidth',0.25);
title(['Base, Mean ',num2str(mean(Z3Base),2),' ,Median ',num2str(median(Z3Base),2)])

%%
load('Z4HSortP.mat')
N = find(Z4HSortP(:,2)>5,1) - 1;
[Z4PD, Z4PNMean, Z4PNMedian, Z4Base, Z4Peak] = Normalise(Z4HSortP,N);
save Z4PD Z4PD;
save Z4PNMean Z4PNMean;
save Z4PNMedian Z4PNMedian;

figure('WindowStyle','docked');
subplot(2,3,1)
plot(t,Z4PD(:,3:26),'k')
title('Z4 Diversity')
subplot(2,3,2);
plot(t,Z4PNMean(:,3:26),'r')
title('Z4 Mean')
subplot(2,3,3);
plot(t,Z4PNMedian(:,3:26),'b')
title('Z4 Median')
subplot(2,2,3);
histogram(Z4Peak,'FaceColor','b','Normalization','Probability','binwidth',0.5);
title(['Peak, Mean ',num2str(mean(Z4Peak),2),' ,Median ',num2str(median(Z4Peak),2)])
subplot(2,2,4);
histogram(Z4Base,'FaceColor','b','Normalization','Probability','binwidth',0.25);
title(['Base, Mean ',num2str(mean(Z4Base),2),' ,Median ',num2str(median(Z4Base),2)])

%%
load('Z5HSortP.mat')
N = find(Z5HSortP(:,2)>5,1) - 1;
[Z5PD, Z5PNMean, Z5PNMedian, Z5Base, Z5Peak] = Normalise(Z5HSortP,N);
save Z5PD Z5PD;
save Z5PNMean Z5PNMean;
save Z5PNMedian Z5PNMedian;

figure('WindowStyle','docked');
subplot(2,3,1)
plot(t,Z5PD(:,3:26),'k')
title('Z5 Diversity')
subplot(2,3,2);
plot(t,Z5PNMean(:,3:26),'r')
title('Z5 Mean')
subplot(2,3,3);
plot(t,Z5PNMedian(:,3:26),'b')
title('Z5 Median')
subplot(2,2,3);
histogram(Z5Peak,'FaceColor','b','Normalization','Probability','binwidth',0.5);
title(['Peak, Mean ',num2str(mean(Z5Peak),2),' ,Median ',num2str(median(Z5Peak),2)])
subplot(2,2,4);
histogram(Z5Base,'FaceColor','b','Normalization','Probability','binwidth',0.25);
title(['Base, Mean ',num2str(mean(Z5Base),2),' ,Median ',num2str(median(Z5Base),2)])

%% 

Z1 = Z1HSortP;
Z2 = Z2HSortP;
Z3 = Z3HSortP;
Z4 = Z4HSortP;
Z5 = Z5HSortP;

save TestData Z1 Z2 Z3 Z4 Z5

BaseLoads = [median(Z1Base) median(Z2Base) median(Z3Base) median(Z4Base) median(Z5Base)];
save BaseLoads BaseLoads

PeakLoads = [median(Z1Peak) median(Z2Peak) median(Z3Peak) median(Z4Peak) median(Z5Peak)];
save PeakLoads PeakLoads

Z1N = Z1PNMedian;
Z2N = Z2PNMedian;
Z3N = Z3PNMedian;
Z4N = Z4PNMedian;
Z5N = Z5PNMedian;

save TestDataN Z1N Z2N Z3N Z4N Z5N

