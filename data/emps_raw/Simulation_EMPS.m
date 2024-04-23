%% CLEAR
close all
clear all
clc

%% DATA
% File to load
load('DATA_EMPS')
% Variables are:
% qm = motor position (measured through the motor encoder)
% qg = the reference position
% t = time
% vir = motor voltage (output of the controller)

%% Parameters of Butterworth filter
% Nyquist frequency
fec = 1000;
pas = 1/fec;
fnyq = fec/2;
% pas means sampling time (1 ms)

% Butterworth parameters
freq_fil = 5*20.0;
nfilt = 4;
ob = freq_fil/fnyq;
[b,a] = butter(nfilt,ob);

%% For Simulation
% Dynamic parameters
M1  = 95.1089;
Fv1 = 203.5034;
Fc1 = 20.3935;
OF1 = -3.1648;

% Maximum voltage (saturation)
viradm = 10;
% For filtering the position
filterq1 = [0.5000 0.5000];
% Sampling
tec = t(10)-t(9);
% Duration
tf = t(end);

%% Simulation of EMPS
disp(' ')
disp('Simulation of EMSP with DATA_EMPS.mat')

tic; 
options=simset('Solver','ode45','RelTol','auto','AbsTol','auto',...
  'Refine',1,'MaxStep',tec/10,'ZeroCross','off');
sim('Simulink_EMPS_Rigide',tf,options);
tps = toc;
h = floor(tps/3600);
m = floor((tps-h*3600)/60);
s = tps-h*3600-m*60;
disp(' ')
disp(sprintf('Simulation duration: %gh,%gm,%gs.',h,m,s))
clear tps h m

%% For comparison
% Simulated data
Force1_s = gtau*vir_s;
q_s   = q1_vect_s(:,1);
dq_s  = q1_vect_s(:,2);
ddq_s = q1_vect_s(:,3);
% Measured data
Force1 = gtau*vir;
q_f    = filtfilt(b,a,qm);
dq_f   = diffcent(q_f,pas);
ddq_f  = diffcent(dq_f,pas);

%% Plot results
figure,
subplot(221),
plot(t,q_f,'b'),hold on
plot(t,q_s,'r'),grid
ylabel(' Position '),xlabel(' t(s) ')
subplot(222),
plot(t,dq_f,'b'),hold on
plot(t,dq_s,'r'),grid
ylabel(' Velocity '),xlabel(' t(s) ')
subplot(223),
plot(t,ddq_f,'b'),hold on
plot(t,ddq_s,'r'),grid
ylabel(' Acceleration '),xlabel(' t(s) ')
subplot(224),
plot(t,Force1,'b'),hold on
plot(t,Force1_s,'r'),grid
ylabel(' Force '),xlabel(' t(s) ')

%% Compute relative errors
% Remove edge effects because of diffcent function
N_pts = length(q_f);
n_border = 50;
% Measured data
q_f = q_f(n_border:N_pts,1);
dq_f = dq_f(n_border:N_pts,1);
ddq_f = ddq_f(n_border:N_pts,1);
Force1 = Force1(n_border:N_pts,1);
% Simulated data
q_s = q_s(n_border:N_pts,1);
dq_s = dq_s(n_border:N_pts,1);
ddq_s = ddq_s(n_border:N_pts,1);
Force1_s = Force1_s(n_border:N_pts,1);
% Relative errors
rel_err_q   = 100*norm(q_f - q_s)/norm(q_f);
rel_err_dq  = 100*norm(dq_f - dq_s)/norm(dq_f);
rel_err_ddq = 100*norm(ddq_f - ddq_s)/norm(ddq_f);
rel_err_u   = 100*norm(Force1 - Force1_s)/norm(Force1);

%% Display results
disp(' ')
disp('Relative errors with EMPS_DATA.mat : ')
disp(['Position     : ',num2str(rel_err_q),'%'])
disp(['Velocity     : ',num2str(rel_err_dq),'%'])
disp(['Acceleration : ',num2str(rel_err_ddq),'%'])
disp(['Force        : ',num2str(rel_err_u),'%'])

