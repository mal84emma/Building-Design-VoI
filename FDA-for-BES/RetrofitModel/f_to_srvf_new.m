function q = f_to_srvf_new(f,time)

binsize = mean(diff(time));
[M, N] = size(f);

difftime=time-binsize/2;
ttime=(1:0.25:24)';

fy = zeros(M,N);

for ii = 1:N
    
    y = [f(M,ii); f(1:M,ii)];
    ydiff = diff(y);
    %fo(ii) = y(2)-ydiff(1);
    fy(:,ii) = ydiff(1:length(time))/binsize;
    %yy(:,ii)=y;
    %yydiff(:,ii)=ydiff;

end
q = fy./sqrt(abs(fy)+eps);


%figure('WindowStyle','docked');

%plot(time,yy(:,1));
%hold on
%plot(difftime,fy(:,1));
%plot(difftime,q(:,1));
%hold on;

%integrand=q.*abs(q);
%ff=cumtrapz(difftime,integrand);

%for ii=1:N
%    frec(:,ii)=fo(ii)+ff(:,ii);
%end

%plot(time,frec(:,1));
