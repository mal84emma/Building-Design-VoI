function [zWdcG,zWecG] = CopulaSample(MappedScoresx,MappedScoresy,N,numsample)

xWd = MappedScoresx(1:N,:);
yWd = MappedScoresy(1:N,:);

xWe = MappedScoresx(N+1:end,:);
yWe = MappedScoresy(N+1:end,:);

% Weekdays

ZWd = [xWd yWd];

% Transform Data
for ii=1:48
zWd(:,ii)=ksdensity(ZWd(:,ii),ZWd(:,ii),'function','cdf','Bandwidth',0.04);
end

% Fit Gaussian Copula to transformed data

[zWdyRho]=copulafit('Gaussian',zWd);

% Sample from copula
zWdrG=copularnd('Gaussian',zWdyRho,numsample);

% Transform Back
for ii=1:48
zWdcG(:,ii)=ksdensity(ZWd(:,ii),zWdrG(:,ii),'function','icdf','Bandwidth',0.04);
end

save zWdcG zWdcG


%% Select weekends

ZWe = [xWe yWe];

% Transform Data
for ii=1:48
zWe(:,ii)=ksdensity(ZWe(:,ii),ZWe(:,ii),'function','cdf','Bandwidth',0.04);
end

% Fit Gaussian Copula to transformed data

[zWeyRho]=copulafit('Gaussian',zWe);

% Sample from copula
zWerG=copularnd('Gaussian',zWeyRho,numsample);

% Transform Back
for ii=1:48
zWecG(:,ii)=ksdensity(ZWe(:,ii),zWerG(:,ii),'function','icdf','Bandwidth',0.04);
end

save zWecG zWecG 

















