
clear; %close all;

load scoresx;
load scoresy;

%% y Scores

yZ1=scoresy(1:365,:);
yZ2=scoresy(366:730,:);
yZ3=scoresy(731:1095,:);
yZ4=scoresy(1096:1460,:);
yZ5=scoresy(1461:1825,:);

%% x Scores

xZ1=scoresx(1:365,:);
xZ2=scoresx(366:730,:);
xZ3=scoresx(731:1095,:);
xZ4=scoresx(1096:1460,:);
xZ5=scoresx(1461:1825,:);

%% Weekday scores

% y
yZ1Wd=yZ1(1:261,:);
yZ2Wd=yZ2(1:261,:);
yZ3Wd=yZ3(1:261,:);
yZ4Wd=yZ4(1:261,:);
yZ5Wd=yZ5(1:261,:);

% x
xZ1Wd=xZ1(1:261,:);
xZ2Wd=xZ2(1:261,:);
xZ3Wd=xZ3(1:261,:);
xZ4Wd=xZ4(1:261,:);
xZ5Wd=xZ5(1:261,:);

%% Weekend scores

% y
yZ1We=yZ1(262:end,:);
yZ2We=yZ2(262:end,:);
yZ3We=yZ3(262:end,:);
yZ4We=yZ4(262:end,:);
yZ5We=yZ5(262:end,:);

% x
xZ1We=xZ1(262:end,:);
xZ2We=xZ2(262:end,:);
xZ3We=xZ3(262:end,:);
xZ4We=xZ4(262:end,:);
xZ5We=xZ5(262:end,:);



