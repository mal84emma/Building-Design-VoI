function[DataPlot] = Model(NCluster,Cluster,Base,Range,nsamples)

load GeneratedProfiles_C10

C10 = [C1Gen C2Gen C3Gen C4Gen C5Gen C6Gen C7Gen C8Gen C9Gen C10Gen];

if NCluster == 10
    Div = C10(:,((Cluster-1)*3000+1):Cluster*3000);
end

load GeneratedProfiles_C5

C5 = [C51Gen C52Gen C53Gen C54Gen C55Gen];

if NCluster == 5
    Div = C5(:,((Cluster-1)*3000+1):Cluster*3000);
end

load GeneratedProfiles_C3

C3 = [C31Gen C32Gen C33Gen];

if NCluster == 3
    Div = C3(:,((Cluster-1)*3000+1):Cluster*3000);
end

cutoff_low = 0;

Data = Div.*Range + Base;

Data_cutoff_high = 2*Data(1,:);

for k = 1:3000
   
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

rng(0,'twister');

id1 = randi([1 size(D_10,1)],1,nsamples);

DataPlot = D_10(id1,2:25)';


