# FDA-for-BES
***Functional Data Analysis approach for generation of occupant-related internal loads for Building Energy Simulation***

## Introduction
This suite of tools enables a user to generate occupant-related stochastic internal loads for input into a building energy simulation package such as EnergyPlus.  The tools have been developed using a functional data analysis approach with monitored electricity consumption data taken from the University of Cambridge building stock.  The functional data analysis, in particular the separation of the data into phase and amplitude components and extraction of the Principal Components (PCs) and scores, has extensively been based on the repository https://github.com/jdtuck/fdasrvf_MATLAB edited for use with hourly electricity consumption data.

In this approach each set of 24 hourly values representing one day constitutes a data sample.  Given a set of data samples, functional Principal Component Analysis (fPCA) is used to derive a set of *functional* Principal Components (PCs) - each one a function of time - and a set of coefficients or *scores* on the PCs that control how much each PC contributes to each data sample.  Each data sample, <img src="https://render.githubusercontent.com/render/math?math=\phi(t)"> can therefore be expressed simply as a sum of a mean function, <img src="https://render.githubusercontent.com/render/math?math=\mu(t)"> and the weighted summation over *i* functional PCs, <img src="https://render.githubusercontent.com/render/math?math=\nu_i(t)">, with scores <img src="https://render.githubusercontent.com/render/math?math=\alpha_i">, i.e.

<img src="https://render.githubusercontent.com/render/math?math=\phi(t) = \mu(t) + \Sigma_i \alpha_i \nu_i(t)">

It is useful in the analysis of transient energy demand to separate the data into its phase and amplitude, i.e. the timing component and the magnitude of the demand, as both are quantities of interest. In order to do this, the data are aligned to a common mean using a dynamic programming algorithm (Tucker et al, 2013), giving a set of warping functions that align each data sample to the mean and a set of aligned amplitude functions. fPCA is then performed on the warping and amplitude functions separately, yielding separate PCs for phase and amplitude together with a unique set of scores on the PCs for each data sample. As the PCs are same across all data samples it is the PC scores that distinguish one data sample from another.  

The FDA model thus comprises the phase and amplitude PCs together with the scores that can be used in conjunction with the PCs to regenerate the original data. If we want to generate new sample data it suffices to take a random sample of the scores and use the sampled scores with the PCs to generate sample data. It is necessary to fit a multivariate probability distribution to the scores to ensure correlations are captured - here we use a copula as the scores are not normally distributed.  

The approach is described in detail in the papers detailed below.

## The models
Three types of model are included. These are;

1. Design models for 
  a) Plug Loads, and 
  b) Lighting. 
The design models are based on a clustering analysis of the scores which identifies the more similar zones. These models are intended to be used in the case where no monitored data are available.
2. Retrofit model: the retrofit model comprises a mean function and a set of principal components derived from a large dataset consisting of 54 building zones from four buildings.  This model can be used when limited monitored data are available. The code aligns the new data to the mean function then projects the aligned data and warping functions onto the phase and amplitude PCs and extracts the mapped scores.  These are then used as the basis for a probability distribution of the scores from which samples are drawn and used with the PCs to generate sample data.   
3. Base model: the base model includes the code for alignment of a dataset, extraction of principal components and scores, and generation of sample data. A test dataset comprising data for five different building zones has been included.

### Design model: plug loads
The model includes weekday and weekend normalised sample profiles for 4 different clusters of zones representing an increase in the expected variability in demand from 1 (least variable) to 4 (most variable).  The input required is the expected median hourly base load and load range, together with the expected variability of the data.

The MATLAB command to run the model is 

`[Test2019,Test2020,SampleWeekdays,SampleWeekends] = FDASimulationP(Base,Range,Cluster)`

where Base and Range are the median hourly base load and load range and Cluster is a value from 1 to 4 indicating the expected variability. The outputs Test2019 and Test2020 are the sample annual data for a non-leap year (2019) and a leap year (2020), and SampleWeekdays and SampleWeekends are the daily samples that have been used to build the annual demand profile.

The following figure shows sample output for Cluster 1 generated using a Base load of 2.5 Wh/m<sup>2</sup>/h and a Load Range of 8 Wh/m<sup>2</sup>/h.

<img src="img/Sample1.png" alt="drawing" width="500"/>


### Design model: lighting
The model includes weekday and weekend normalised sample lighting profiles for 5 different clusters.  These 5 clusters correspond to different types of lighting control in different building zones and the suggested choice of variability is:

1. predominantly manual operation with some areas controlled by lighting sensor,
2. manual operation,
3. occupancy sensor - high occupancy,
4. occupancy sensor - low occupancy,
5. occupancy sensor, occupancy 7 days a week.

The MATLAB command to run the model is 

`[Test2019,Test2020,SampleWeekdays,SampleWeekends] = FDASimulationL(Base,Range,Cluster)`

where Base and Range are the median hourly base load and load range and Cluster is a value from 1 to 5 according to the above list. The outputs Test2019 and Test2020 are the sample annual data for a non-leap year (2019) and a leap year (2020), and SampleWeekdays and SampleWeekends are the daily samples that have been used to build the annual demand profile.

### Retrofit model
If monitored electricity consumption are available the data can be projected onto the standard set of PCs derived from a large training dataset.  The first step is to format the data so that it is hourly, normalised by area and re-normalised according to the weekday median base load and load range.  It is also necessary to include an identifier signifying whether the data are weekday or weekend - the code requires an additional column to be inserted into the MATLAB file at the start of the file with numbers 1 to 7 signifying the day of the week, where 1 is a Monday and 7 is a Sunday.  The data should also be sorted by the day of the week, so all the Mondays first, then the Tuesdays etc.  

Sample data are provided in file **TestData.mat** with the normalised data in **TestDataN.mat**. these files contain data for 5 different building zones, Z1 to Z5, with 365 days of data for each zone.  The data have been ordered as described above.  Also included are files **BaseLoads.mat** and **PeakLoads.mat** which contain the median weekday base loads and peak loads used in the normalisation.

The file **RetrofitModel.m** illustrates how to project the data for a single zone onto the PCs and generate annual data samples. In the file it is necessary to select the zone of interest and ensure that the correct base load and peak load are selected.  Then the MATLAB files are run in the following order:

1. **WarpToPL.m** aligns the data with the training data mean function, this process requires the dynamic programming algorithm in directory **DP** which must either be in the same directory as the run files or the path must be added to MATLAB,
2. **MappedScores.m** projects the aligned data onto the PCs and extracts the scores,
3. **CopulaSample.m** fits a copula to the scores and generate random score samples for the weekday and weekend separately,
4. **GenerateProfiles.m** takes as input the sample scores together with the median base load and load range identified during the normalisation of the data; this routine generates sample weekday/weekend data and removes extreme samples i.e samples with negative values or samples with end of day demand more than twice the start of day demand, 
5. **GenerateAnnualDemand.m** reads in the generated profiles, selects a random sample and generates annual samples including bank holidays according to the 2019 or 2020 dates (non-leap or leap year),
6. **PlotKPIs.m** plots the hourly base load, hourly peak load, daily total demand and the timing of the daily peak for the generated samples compared against the monitored data.

The generated sample appears in MATLAB file **AnnualDemand.mat** and is also saved to file **AnnualDemand.csv** for ease of input into a building energy simulation (**AnnualDemandL.mat** for the leap year data).

The following figure shows sample output in terms of the KPIs for Zone 1.

<img src="img/RetrofitSample1.png" alt="drawing" width="700"/>

### Base model
This is the model used to generate PCs and scores directly from the training data.  As for the retrofit model, the data must be formatted and normalised and the sample data **TestData.mat** and **TestDataN.mat** are used as an example. The base model calculates the PCs and scores for the 5 zones together, these are similar to but different from the PCs calculated for the large training dataset provided for the Retrofit model.

The file **BaseModel.m** illustrates how to align the data to a common mean and to calculate PCs and scores. The example uses just the weekday data and generates sample weekday data for the 5 test zones.  The MATLAB files are run in the following order:

1. **Warp.m** aligns the data to a common mean function, this process requires the dynamic programming algorithm in directory **DP** which must either be in the same directory as the run files or the path must be added to MATLAB,
2. **CalcPCs.m** calculates the PCs for phase and amplitude and the scores that can be used in conjunction with the PCs to re-generate each of the data samples,,
3. **CopulaSample.m** fits a copula to the scores and generates random score samples for the weekday samples,
4. **GenerateProfiles.m** takes as input the sample scores together with the median base load and load range identified during the normalisation of the data; this routine generates sample weekday/weekend data and removes extreme samples i.e samples with negative values or samples with end of day demand more than twice the start of day demand. 

The mean and 90% confidence limits of the generated samples are plotted against the test data for comparison.  

## References
1) The best introduction to FDA is at https://www.psych.mcgill.ca/misc/fda/

**Useful books:**

2) J. Ramsay and B. Silverman, *Functional Data Analysis*, Springer, New York, second edition, 2005
3) A. Srivastava and E. Klassen, *Functional and Shape Data Analysis*,  Springer, New York, 2016

**Useful papers on phase/amplitude separation:**

*An interesting introduction,*

4) J. S. Marron, J. O. Ramsay, L. M. Sangalli, and A. Srivastava,  *Functional Data Analysis of Amplitude and Phase Variation*. Statistical Science, 30(4):468–484, Nov. 2015. ISSN0883-4237. doi: 10.1214/15-STS524. URL http://projecteuclid.org/euclid.ss/1449670854

*The detail of the methodology used for phase and amplitude separation,*

5) J. D. Tucker, W. Wu, and A. Srivastava, *Generative models for functional data using phase and amplitude separation*. Computational Statistics & Data Analysis, 61:50–66, May 2013.ISSN 01679473. doi: 10.1016/j.csda.2012.12.001. URL http://linkinghub.elsevier.com/retrieve/pii/S0167947312004227.
6) GitHub repository https://github.com/jdtuck/fdasrvf_MATLAB

**Application to the context of building internal loads:**

6) R.M. Ward, R. Choudhary, Y. Heo, and J. Aston,  *A data-centric bottom up model for generation of stochastic internal load profiles based on space-use type.* Journal of Building Performance Simulation, 12(5), 620 – 636, 2019. https://doi.org/10.1080/19401493.2019.1583287
7) R. Ward, C. Sze Yin Wong, A. Chong, R. Choudhary, and S. Ramasamy,  *A study on the transferability of computational models of building electricity load patterns across climatic zones.* Energy and Buildings 237, 110826, 2021.  https://doi.org/10.1016/j.enbuild.2021.110826
8) R.M. Ward, R. Choudhary, *A tool for generation of stochastic occupant-based internal loads using a functional data analysis approach to re-define ‘activity’.* Journal of Building Performance Simulation 14(3), 303-327, 2021. https://doi.org/10.1080/19401493.2021.1919209








