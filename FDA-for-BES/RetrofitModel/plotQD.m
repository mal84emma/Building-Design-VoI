function [p1,p2,pDMean,pf1] = plotQD(Data1,c1)

tt=1:1:24;
tt2=[tt';flipud(tt')];
q1=0.95;
q2=0.05;

curve1=quantile(Data1,q1,2);
curve2=quantile(Data1,q2,2);
Data1Mean=mean(Data1,2);
inBetween1=[curve1;flipud(curve2)];

p1=plot(tt,curve1,'Color',c1);
hold on;
pDMean=plot(tt,Data1Mean,'--','Color',c1,'LineWidth',2);
p2=plot(tt,curve2,'Color',c1);
pf1=fill(tt2,inBetween1,c1,'FaceAlpha',0.3);

xlim([0,24]);

ax=gca;
ax.XTick = [0 6 12 18 24];
