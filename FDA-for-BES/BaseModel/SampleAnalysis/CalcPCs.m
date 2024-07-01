%% Extract Principal Components and scores for amplitude and phase (warping) functions

clear; %close all;

option.showplot = 0; % Turn this on (1) to show check plots

load warp_dat

[M, N] = size(f);

%% Check qn

integrand=qn.*abs(qn);
fno=fn(M,:);
ff=cumsum(integrand);
%ff=cumtrapz(t,integrand);

for ii=1:N
    frec(:,ii)=fno(1,ii)+ff(:,ii);
end

if option.showplot == 1
    figure('WindowStyle','docked');

    subplot(1,2,1);
    plot(t,fn);
    title('Aligned Amplitude Functions')
    %ylim([-0.1,1.1]);

    subplot(1,2,2);
    plot(t,frec);
    title('Re-constituted Aligned Amplitude Functions')
    %ylim([-0.1,1.1]);

end
%% Separated and Warped Data
% sampling from the estimated model

x0 = f;
x = fn; % aligned function
binsize = mean(diff(t));

% compute mean and covariance in q-domain
q_new = qn; % aligned srvf
mq_new = mean(qn,2);
m_new = sign(fn(end,:)).*sqrt(abs(fn(end,:)));  % scaled version
%m_new = sign(fn(round(length(t)/2),:)).*sqrt(abs(fn(round(length(t)/2),:)));  % scaled version
% NB if using cumtrpazmid, you need to use fn(round(length(t)/2), if using
% cumtrapz, use fn(end,:) - but cumtrapz gives wide spread at end of day

mqn2 = [mq_new; mean(m_new)];
qnew = [q_new;m_new]';

meanqnew=mean(qnew,1);

save('meanqnew.mat','meanqnew');


%% Calculate meany

integrand=meanqnew(1:M).*abs(meanqnew(1:M));
fo=sign(meanqnew(end)).*(meanqnew(end).^2);
ff = cumsum(integrand);

meany = ff + fo;

if option.showplot == 1
    figure('WindowStyle','docked');
    plot(t,fn);
    hold on;
    plot(t,meany,'k','LineWidth',2);
    title('Mean function')
end

%% Calculate mean and principal components for amplitude

qnew_centred = qnew - meanqnew;
    
Cc = cov(qnew_centred);

save('qnew.mat','qnew');
save('Cov_Centred.mat','Cc');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%  Use same approach as for warping functions i.e take SVD of Covariance
%%%  matrix and calculate scores from [q_new;m_new]'*U

[Uy,Sy,Vy] = svd(Cc); % y for vertical
Sigy=diag(Sy);
propY=Sigy/sum(Sigy);
sumpropY=cumsum(propY);
scoresy = qnew_centred*Uy;
stdS = sqrt(Sigy);

save('Uy.mat','Uy'); % Principal components in columns
save('Sigy.mat','Sigy');
save('propY.mat','propY');
save('sumpropY.mat','sumpropY');
save('scoresy.mat','scoresy');
save Vy Vy;
save Sy Sy;

recon1=zeros(M+1,N);
for i=1:N
    for j=1:M+1
        recon1(:,i)=recon1(:,i)+(scoresy(i,j)*Uy(:,j));
    end
end

recon2 = recon1 + meanqnew';

recon = recon2';

for k = 1:N
    x_s_recon(:,k) = (sign(recon2(end,k)).*(recon2(end,k).^2))...
        +cumsum(recon2(1:end-1,k).*abs(recon2(1:end-1,k)));
end

if option.showplot ==1
    figure('WindowStyle','docked')
    plot(t,fn(:,5),'k','LineWidth',2);
    hold on;
    plot(t,x_s_recon(:,5),'r:','LineWidth',2);
    title('Example re-constituted sample')
end
%%
if option.showplot ==1
    figure('WindowStyle','docked')
    plot(t,meanqnew(1,1:M),'k','LineWidth',2);
    hold on;
    plot(t,Uy(1:end-1,1),'r','LineWidth',1);
    plot(t,Uy(1:end-1,2),'b','LineWidth',1);
    plot(t,Uy(1:end-1,3),'g','LineWidth',1);
    plot(t,Uy(1:end-1,4),'m','LineWidth',1);
    plot(t,Uy(1:end-1,5),'c','LineWidth',1);
    title('Mean and first 5 Uy terms')
end

    for i=1:24
    PC(:,i)=(sign(Uy(end,i)).*(Uy(end,i).^2))+cumsum(Uy(1:end-1,i).*abs(Uy(1:end-1,i)));
    end

    figure('WindowStyle','docked')
    plot(meany,'k','LineWidth',2); 
    hold on; 
    plot(PC(:,1),'r','LineWidth',1); 
    plot(PC(:,2),'b','LineWidth',1); 
    plot(PC(:,3),'g','LineWidth',1); 
    plot(PC(:,4),'m','LineWidth',1); 
    plot(PC(:,5),'c','LineWidth',1); 
    title('Mean and first 5 amplitude PCs', 'fontsize', 14);
    legend('mean','PC 1','PC 2','PC 3','PC 4','PC 5');


%% Calculate meany + 1SD*principal components

qs_sp = zeros(M+1,M+1);
qs_sn = zeros(M+1,M+1);

%stdsy=std(scoresy,1);
stdsy = 1;

    for i=1:M+1 % number of principal components
        
        qs_sp(i,:) = (1*stdsy*Uy(:,i))';
        qs_sn(i,:) = (-1*stdsy*Uy(:,i))';  
        
    end
 
for k=1:M+1
    for i=1:M+1
        qs_tp(k,i)=qs_sp(k,i)+meanqnew(1,i);
        qs_tn(k,i)=qs_sn(k,i)+meanqnew(1,i);
    end
end

q_sp = qs_tp';
q_sn = qs_tn';

% compute the correspondence to the original function domain
for k = 1:M+1
    x_sp(:,k) = (sign(q_sp(M+1,k)).*(q_sp(M+1,k).^2))...
        +cumsum(q_sp(1:M,k).*abs(q_sp(1:M,k)));
    x_sn(:,k) = (sign(q_sn(M+1,k)).*(q_sn(M+1,k).^2))...
        +cumsum(q_sn(1:M,k).*abs(q_sn(1:M,k)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate PCs for warping functions

[mu,gam_mu,psi,vec] = SqrtMean(gam.');

save('mu.mat','mu');
save('vec.mat','vec'); % Data

vm=mean(vec);

for i=1:N
    for j=1:M-1
        vec_centred(i,j)=vec(i,j)-vm(1,j);
    end
end

K = cov(vec_centred); % Covariance vector
%K=cov(vec);
[U,S,V] = svd(K);
Sig = diag(S); % Variance on diagonal of S

save('U.mat','U'); % Principal components in rows
save('Sig.mat','Sig');
save V V;

propX=Sig/sum(Sig);
sumpropX=cumsum(propX);

save('propX.mat','propX');
save('sumpropX.mat','sumpropX');

scoresx=vec_centred*U;
save('scoresx.mat','scoresx');
   
%% Plot X PCs

[n,~] = size(gam);

T = n;

for i=1:n-1
    
    v=U(:,i)';
    vn=norm(v)/sqrt(T);
    psi=cos(vn)*mu+sin(vn)*v/vn;
    gam0=[0 cumsum(psi.*psi)]/T;
    PCx(i,:)=(gam0-gam0(1))/(gam0(end)-gam0(1));

end

xx = [0 0; 1 1];

figure('WindowStyle','docked')
plot(xx(:,1),xx(:,2),'k','linewidth',2);
hold on;
plot((0:M-1)/(M-1), PCx(1,:),'r', 'linewidth', 1);
plot((0:M-1)/(M-1), PCx(2,:),'b', 'linewidth', 1);
plot((0:M-1)/(M-1), PCx(3,:),'g', 'linewidth', 1);
plot((0:M-1)/(M-1), PCx(4,:),'m', 'linewidth', 1);
plot((0:M-1)/(M-1), PCx(5,:),'c', 'linewidth', 1);

axis square;
title('Mean and first 5 phase PCs', 'fontsize', 14);
legend({'mean','PC 1','PC 2','PC 3','PC 4','PC 5'},'Location','Northwest');


%  Calculate +/- std on PCxs
%stdsx=std(scoresx);
stdsx = 1;

mnx0=[0 cumsum(mu.*mu)]/T; % Alternatively calculate from vm using the eqns below
meanx=(mnx0-mnx0(1))/(mnx0(end)-mnx0(1)); % Gives the same result

for i=1:n-1
    
    v=stdsx*U(:,i)';
    vn=norm(v)/sqrt(T);
    psi=cos(vn)*mu+sin(vn)*v/vn;
    gam0=[0 cumsum(psi.*psi)]/T;
    PCxsdp1(i,:)=(gam0-gam0(1))/(gam0(end)-gam0(1));

end

for i=1:n-1
    
    v=1*stdsx*U(:,i)';
    vn=norm(v)/sqrt(T);
    psi=cos(vn)*mu+sin(vn)*v/vn;
    
    amu=cos(vn)*mu;
    
    gam0p2=[0 cumsum(psi.*psi)]/T;
    gammu=[0 cumsum(amu.*amu)]/T;
    gam0n2=gam0p2-2*gammu;
    PCxsdp2(i,:)=(gam0p2-gam0p2(1))/(gam0p2(M)-gam0p2(1));
    meanx2(i,:)=(gammu-gammu(1))/(gammu(M)-gammu(1));
    PCxsdn2(i,:)=(gam0n2-gam0n2(1))/(gam0n2(M)-gam0n2(1));
    

end

for k = 1:n-1
    x_cp(:,k) = interp1((0:M-1)/(M-1), meany, invertGamma(PCxsdp2(k,:)')');
    x_cn(:,k) = interp1((0:M-1)/(M-1), meany, invertGamma(PCxsdn2(k,:)')');
end

%% Plot Cumulative proportion of variability

figure('WindowStyle','docked')
plot(sumpropY,'b-o','LineWidth',2);
hold on;
plot(sumpropX,'r-x','LineWidth',2);
xlabel('PC')
ylabel('Cumulative Proportion')
title('Cumulative Proportion of Variability')
legend({'Y','X'},'Location','Northwest');


