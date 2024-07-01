function f = srvf_to_f_new(q,time,fo)
integrand = q.*abs(q);
f = cumsum(integrand);
for i = 1:length(fo)
    f(:,i) = fo(i) + f(:,i);
end