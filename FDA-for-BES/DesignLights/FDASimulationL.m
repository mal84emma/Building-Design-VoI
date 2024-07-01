function[Test2019,Test2020,SampleWeekdays,SampleWeekends] = FDASimulationL(Base,Range,Cluster)

rng('shuffle'); 

Peak = Base + Range;

NCluster = 5;

id1 = randi(1000,1,265);
id2 = randi(1000,1,114);

DataWd = Model(NCluster,Cluster,Base,Range,1000);
DataWe = ModelWe(NCluster,Cluster,Base,Range,1000);

Weekday = DataWd(:,id1);
Weekend = DataWe(:,id2);

Week = zeros(24,7,53);

for ii = 1:53
    for jj = 1:5
        Week(:,jj,ii) = Weekday(:,(ii-1)*5+jj);
    end
    for jj = 1:2
        Week(:,jj+5,ii) = Weekend(:,(ii-1)*2+jj);
    end
    
end

Data = reshape(Week,24,7*53);

%% Specify year => start day

stdayL = 3; % 2020, 1/1 == Wednesday
YearL = Data(:,stdayL:stdayL+365);

stday = 2; % 2019, 1/1 = Tuesday
Year = Data(:,stday:stday+364);

%% Bank holidays

% 2020 - leap year
YearL(:,1) = Weekend(:,107); % 1/1/2020
YearL(:,101) = Weekend(:,108); % 10/4/2020
YearL(:,104) = Weekend(:,109); % 13/4/2020
YearL(:,129) = Weekend(:,110); % 8/5/2020
YearL(:,146) = Weekend(:,111); % 25/5/2020
YearL(:,244) = Weekend(:,112); % 31/8/2020
YearL(:,360) = Weekend(:,113); % 25/12/2020
YearL(:,362) = Weekend(:,114); % 27/12/2020

% 2019
Year(:,1) = Weekend(:,107); % 1/1/2020
Year(:,109) = Weekend(:,108); % 19/4/2020
Year(:,112) = Weekend(:,109); % 22/4/2020
Year(:,126) = Weekend(:,110); % 6/5/2020
Year(:,147) = Weekend(:,111); % 27/5/2020
Year(:,238) = Weekend(:,112); % 26/8/2020
Year(:,358) = Weekend(:,113); % 25/12/2020
Year(:,359) = Weekend(:,114); % 26/12/2020

%% Smooth at midnight

for ii = 1:365
    YearL(24,ii) = (YearL(23,ii)+YearL(1,ii+1))/2;
end

Test2020 = reshape(YearL,24*366,1);

for ii = 1:364
    Year(24,ii) = (Year(23,ii)+Year(1,ii+1))/2;
end

Test2019 = reshape(Year,24*365,1);

%% Plot
figure('WindowStyle','docked');

subplot(2,2,1)

for ii = 1:52
t = 1:1:7*24;
plot(t,Test2019((ii-1)*7*24+1:ii*7*24,1));
hold on;
end
ax = gca;
ax.XTick = [12,36,60,84,108,132,156];
xlim([0,168])
ax.XTickLabel = [1 2 3 4 5 6 7];
xlabel('Day')
ylabel('Power demand (Wh/m^2)')

str=sprintf('Weekly Sample Data'); % %1.0f Clusters, Cluster %1.0f, \n Base Load %1.1f, Peak Load %1.1f',NCluster,Cluster,Base,Peak);
title(str);

%% Time Table

t2020_1 = datetime(2020,1,1,1,0,0);
t2020_2 = datetime(2020,12,31,24,0,0);

Dates = t2020_1:hours(1):t2020_2;

%figure('WindowStyle','docked');
%plot(Dates,Test2020)

t2019_1 = datetime(2019,1,1,1,0,0);
t2019_2 = datetime(2019,12,31,24,0,0);

Dates = t2019_1:hours(1):t2019_2;

subplot(2,2,2)
plot(Dates,Test2019)
title('Annual Sample Data');

%% Qplots

SampleWeek = reshape(Year(:,7:363),24,7,51);
SampleWeekdays = reshape(SampleWeek(:,1:5,:),24,255);
SampleWeekends = reshape(SampleWeek(:,6:7,:),24,102);

subplot(2,2,3)
t = 1:1:24;
plot(t,SampleWeekdays,'k:');
hold on;
plotQ(SampleWeekdays,'r');
yl = ylim;
ylim(yl);
title('Weekday')
ylabel('Power demand (Wh/m^2)')
xlabel('Hour')
subplot(2,2,4)
t = 1:1:24;
plot(t,SampleWeekends,'k:');
hold on;
plotQ(SampleWeekends,'b');
ylim(yl);
title('Weekend')
xlabel('Hour')

end
