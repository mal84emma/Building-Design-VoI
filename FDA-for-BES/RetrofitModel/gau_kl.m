function kl = gau_kl(pm,pv,qm,qv)

dpv = prod(pv);
dqv = prod(qv);

iqv = qv.^-1;

diff = qm - pm;

kl = 0.5*(log(dqv/dpv) + iqv*pv' + diff.*iqv*diff' - size(pm,2));
