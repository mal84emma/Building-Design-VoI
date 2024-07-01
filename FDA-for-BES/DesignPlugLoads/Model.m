function[DataPlot] = Model(NCluster,Cluster,Base,Range,nsamples)

load GeneratedProfiles12

C12 = Sampleyx_c;

if NCluster == 12
    Div = C12(:,((Cluster-1)*2000+1):Cluster*2000);
end

load GeneratedProfiles7

C7 = Sampleyx_c;

if NCluster == 7
    Div = C7(:,((Cluster-1)*2000+1):Cluster*2000);
end

load GeneratedProfiles4

C4 = Sampleyx_c;

if NCluster == 4
    Div = C4(:,((Cluster-1)*2000+1):Cluster*2000);
end

cutoff_low = 0;

Data = Div.*Range + Base;

Data_cutoff_high = 2*Data(1,:);

for k = 1:2000
   
    Data_compare_low = min(Data(:,k));
    Data_compare_high = Data(end,k);
    Data_ind(k,1) = (Data_compare_low < cutoff_low)|(Data_compare_high > Data_cutoff_high(1,k));

end

Datacheck = [Data_ind Data'];
Datachecksort = sortrows(Datacheck,1);
if max(Datachecksort(:,1)) > 0
    ND = find(Datachecksort(:,1)>0);
    D_10 = Datachecksort(1:(ND-1),:);
else
    D_10 = Datachecksort;
end

%rng(0,'twister'); % Leave in for repeatable results

id1 = randi([1 size(D_10,1)],1,nsamples);

DataPlot = D_10(id1,2:25)';


