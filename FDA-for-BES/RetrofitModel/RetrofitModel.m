%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Routine for projecting monitored data onto database of PCs and         %
% generating sample data for input into BES                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% Load test data
fprintf('Set Up Data \n');

load TestDataN % Normalised Data - 5 zones included
load TestData % Raw Data for the 5 test zones
load BaseLoads % Base Loads for the 5 test zones
load PeakLoads % Peak Loads for the 5 test zones

% Select zone of interest

Data = Z1; 
DataN = Z1N;

% Select Base and Peak loads for zone of interest

Base = BaseLoads(1);
Peak = PeakLoads(1);

% Identify weekdays

N = find(DataN(:,2)>5,1)-1; % Find Weekdays

%% Align the new data with the mean profile calculated for the database of samples
fprintf('Align to mean function \n'); 

tic
WarpToPL(DataN);
toc

% Generates a file WarpToPL_dat.mat which contains the output required in
% subsequent processing

%% Project aligned data and warping functions onto PCs and extract scores
fprintf('Map scores \n');

tic
[MappedScoresx,MappedScoresy] = MappedScores(DataN);
toc

%% Fit copula to mapped scores and generate sample scores
fprintf('Generate sample scores \n');

nsample = 3000; % High number used to ensure variability of results

tic
[SampleWd,SampleWe] = CopulaSample(MappedScoresx,MappedScoresy,N,nsample);
toc

%% Use sample scores and PCs to generate sample weekday and weekend demand
fprintf('Generate daily sample data \n');

tic
[zWd_out,zWe_out] = GenerateProfiles(SampleWd,SampleWe,nsample,Base,Peak,Data,N);
toc

%% Generate annual demand profiles for use in BES (regular and leap year)
fprintf('Generate annual sample data \n');

tic
[AnnualDemand,AnnualDemandL] = GenerateAnnualDemand(zWd_out,zWe_out);
toc

% Also written to *.csv files

%% Plot comparison of KPIs between sample data and generated samples
fprintf('Plot KPIs \n');

PlotKPIs(Data,AnnualDemand)








