function [d, Meand, Mediand, Base, Peak] = Normalise(TestData,N)

Data = TestData(:,3:26)';

Range = max(Data) - min(Data);
Base = min(Data);
div = (Data - Base)./Range;

Peak = max(Data(:,1:N));
MeanPeak = mean(Peak);
MedianPeak = median(Peak);

Base = min(Data(:,1:N));
MeanBase = mean(Base);
MedianBase = median(Base);

MeanRange = MeanPeak - MeanBase;
MedianRange = MedianPeak - MedianBase;

Meandiv = (Data - MeanBase)./MeanRange;
Mediandiv = (Data - MedianBase)./MedianRange;

d = [TestData(:,1:2) div'];
Meand = [TestData(:,1:2) Meandiv'];
Mediand = [TestData(:,1:2) Mediandiv'];

end


