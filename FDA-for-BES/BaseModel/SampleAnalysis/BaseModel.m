%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Routine for aligning data and extracting PCs and scores.  Scores are   %
% then sampled and the sample scores are combined with the PCs to        %
% produce sample data                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% Align data
fprintf('Align Data \n');

tic
Warp
toc

%% Calculate PCs
fprintf('Calculate PCs and Scores \n')

tic
CalcPCs
toc

%% Sample Scores
fprintf('Sample Scores \n')

tic
CopulaSample
toc

%% Generate Profiles
fprintf('Generate Profiles \n')

tic
GenerateProfiles
toc
