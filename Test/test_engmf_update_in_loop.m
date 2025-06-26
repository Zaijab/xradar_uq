clear
clc
%%
close all
%%
runKDE = true;
runGUE = false;
runEnGMM = true;
screenshotPlots = false;
compileToVideo = false;

% Gravitational constant
mu = 0.012150584269940; 

nSamples = 100;
displaySamples = 100;

% Specify initial mean for object CRTBP non-dimensionalized coordinates
mu_initial = [1.021339954388544 -0.000000045869005 -0.181619950369762 0.000000617839352 -0.101759879771430 0.000001049698173]'; % mean.rot.t0;  % mean of initial state (in rotating frame)
P_initial = 1.0e-08 *[ ...
   0.067741479217036  -0.000029214433641   0.000292500436172   0.000343197998120  -0.000801894296500  -0.000076851751508
  -0.000029214433641   0.067949657828148  -0.000045655889447   0.000112485276059   0.002893878948354  -0.000038999497288
   0.000292500436172  -0.000045655889447   0.067754170807105  -0.000931574297640   0.000434803811832   0.000042975146838
   0.000343197998120   0.000112485276059  -0.000931574297640   0.950650788374193   0.004879599683572   0.000839738344685
  -0.000801894296500   0.002893878948354   0.000434803811832   0.004879599683572   0.955575624017479  -0.002913896437441
  -0.000076851751508  -0.000038999497288   0.000042975146838   0.000839738344685  -0.002913896437441   0.954675354567578];

ICs = mvnrnd(mu_initial, P_initial, nSamples);

measurementInterval = 0.005;
frameInterval = 0.001;

% set up initial time and final time (non dimensional)
t0 = 0;      
tf = 0.75103;

% Initialize Condition is Gaussian
nX     = 6;  
w0     = {1};
m0     = {mu_initial};
P0     = {P_initial};         
dH = 1.455021851351014; 
[t, Y] = PropagateSamples(ICs, t0, frameInterval, measurementInterval);

% Measurement Covariance - R
R = diag([(0.5 * pi / 180)^2, (0.5 * pi / 180)^2]);
% Measurements num_samples 
Z = angles_only(Y(:,:,end), mu, R, tre);





