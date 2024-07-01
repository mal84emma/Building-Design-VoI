function [MappedScoresx,MappedScoresy] = MappedScores(TestData)

f = TestData(:,3:26)';
t = (1:24)';

load warp_dat_PL; % f t fn qn q0 fmean mqn gam x_cr calculated for Gates Building

mqnPL=mqn;
fnPL=fn;
qnPL=qn;

load WarpToPL_dat % Data from WarpToPL

[M, N] = size(f);

%% Separated and Warped Data
% sampling from the estimated model

x0 = f;
x = fn; % aligned function
binsize = mean(diff(t));

% compute mean and covariance in q-domain
q_new = qnPL; % aligned srvf
mq_new = mqnPL; % mean srvf

m_new_PL = sign(fnPL(24,:)).*sqrt(abs(fnPL(24,:)));  % scaled version

qnew = [q_new;m_new_PL]';
meanqnew=mean(qnew,1);

%% Calculate meany

integrand=meanqnew(1:end-1).*abs(meanqnew(1:end-1));
fo=sign(meanqnew(end)).*(meanqnew(end).^2);
ff=cumsum(integrand);

meany=fo+ff;

q_new = qn; % aligned srvf
mq_new = mqn; % mean srvf

m_new = sign(fn(24,:)).*sqrt(abs(fn(24,:)));  

qnew = [q_new;m_new]';


%% Centre qnew
for i=1:N
    for j=1:M+1
        qnew_centred(i,j)=qnew(i,j)- meanqnew(1,j);
    end
end

%  Calculate scores from [q_new;m_new]'*U

load Uy_PL;
load Vy_PL;

MappedScoresy=qnew_centred*Uy;
save MappedScoresy MappedScoresy;

Cc_mapped = cov(qnew_centred);

Sy_new = Uy'*Cc_mapped*Vy;
Sigy_new=diag(Sy_new);
propY_new=Sigy_new/sum(Sigy_new);
save propY_new propY_new

% Check approach

scorecomp=zeros(25,25);

n=25; %No of PCs to be included
d=N; % Data items to be examined

for jj=1:d
    for i=1:n
        scc(i,:)=MappedScoresy(jj,i)*Uy(:,i)';
        scorecomp(i+1,:)=scorecomp(i,:)+scc(i,:);
    end  

    qr=zeros(25,25);

    qr(1,:)=meanqnew(1,:);

    for i=1:n
        qr(i+1,:)=qr(1,:)+scorecomp(i+1,:);
    end

    qrt=qr';
    
    for k = 1:n+1
        y_sr(:,k,jj) = (sign(qrt(end,k)).*(qrt(end,k).^2))...
            +cumsum(qrt(1:end-1,k).*abs(qrt(1:end-1,k)));
    end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate scores for warping functions

load mu_PL;
load vec_PL;

muPL=mu;
vecPL=vec;

[mu,gam_mu,psi,vec] = SqrtMeanMap(gam.',muPL);

vm=mean(vecPL);

for i=1:N
    for j=1:M-1
        vec_centred(i,j)=vec(i,j)-vm(1,j);
    end
end

load U_PL;
load V_PL;

MappedScoresx=vec_centred*U;

K_mapped = cov(vec_centred);

S_new = U'*K_mapped*V;
Sig_new=diag(S_new);
propX_new=Sig_new/sum(Sig_new);

save MappedScoresx MappedScoresx;
save propX_new propX_new

