function [] = PlotKPIs(DataI,AnnualDemand)

Data = DataI(:,3:26)';

AD = reshape(AnnualDemand.Test2019,24,365);

ADmin = min(AD);
ADmax = max(AD);
ADDT = sum(AD);
ADPT = AD==ADmax;

PeakTime=0;

for i=1:365
    for j=1:24
        if AD(j,i)==max(AD(:,i))
            PeakTime(j,i)=j;
        else
            PeakTime(j,i)=0;
        end
    end 
end
ADPTH=(max(PeakTime)+1);


Datamin = min(Data);
Datamax = max(Data);
DataDT = sum(Data);
DataPT = Data==Datamax;

PeakTime=0;

for i=1:365
    for j=1:24
        if Data(j,i)==max(Data(:,i))
            PeakTime(j,i)=j;
        else
            PeakTime(j,i)=0;
        end
    end 
end
DataPTH=(max(PeakTime)+1);

plotBase = [Datamin,ADmin];
plotPeak = [Datamax,ADmax];
plotDT = [DataDT,ADDT];
plotPTH = [DataPTH,ADPTH];


grp = [ones(365,1); 2*ones(365,1)];

figure('WindowStyle','docked');

subplot(1,4,1)
boxplot(plotBase,grp,'Labels',{'Data','Simulation'})
title('Base Load')
ax = gca;
ax.FontSize = 14;
ylabel('Hourly Base Load (Wh/m^2)')
subplot(1,4,2)
boxplot(plotPeak,grp,'Labels',{'Data','Simulation'})
title('Peak Load')
ax = gca;
ax.FontSize = 14;
ylabel('Hourly Peak Load (Wh/m^2)')
subplot(1,4,4)
boxplot(plotDT,grp,'Labels',{'Data','Simulation'})
title('Daily Total')
ax = gca;
ax.FontSize = 14;
ylabel('Daily Total Demand (Wh/m^2)')
subplot(1,4,3)
boxplot(plotPTH,grp,'Labels',{'Data','Simulation'})
title('Peak Time')
ax = gca;
ax.FontSize = 14;
ylabel('Hour')
