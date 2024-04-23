
%% INTRO
% This m file runs the cross-test vlidation.
% Run Scrip_IDIM_LS m file before running it in order to have the IDIM LS
% estimates.
%

disp(' ')
disp(' Validating the IDIM-LS estimates with cross-test validation ')
disp(' ')

%% DATA
% File to load to perform cross-test validation
load('DATA_EMPS_PULSES')
% Variables are:
% qm = motor position (measured through the motor encoder)
% qg = the reference position
% t = time
% vir = motor voltage (output of the controller)
% pulses_N = pulses applied to EMPS

%% Construction of the vector measurements
% Motor force
Force1 = gtau*vir;

%% Data filtering
% q_f is qm filtered by Butterworth
q_f = filtfilt(b,a,qm);
% Diffcent = central differentiation
% Velocity
dq_f = diffcent(q_f,pas);
% Acceleration
% diffcent = central differentiation
ddq_f = diffcent(dq_f,pas);
% Remove edge effects because of diffcent function
N_pts = length(q_f);
n_border = 50;
q_f = q_f(n_border:N_pts,1);
dq_f = dq_f(n_border:N_pts,1);
ddq_f = ddq_f(n_border:N_pts,1);
Force1 = Force1(n_border:N_pts,1);
td = t(n_border:N_pts,1);
q_ref = qg(n_border:N_pts,1);

%% Construction of the observation matrix
% Raw observation matrix
X_0 = [ddq_f dq_f sign(dq_f) ones(size(q_f))];
% QR factorization of the observation matrix
[qx,rx] = qr(X_0,0);
% absolute values of the diagonal elements of R
abs_diag_rx = abs(diag(rx));

%% Decimation process
% Decimate columns of the raw observation matrix
[nligne ncolonne] = size(X_0);
X_IDM_0 = [];
for col = 1 : ncolonne
    Xi = X_0(:,col);
    Xi = decimate(Xi,ndecim);
    X_IDM_0 = [X_IDM_0 Xi];
    clear Xi
end
% Decimate the vector of measurements
y_IDM_0 = decimate(Force1,ndecim);
% Decimate t
t_decim = decimate(td,ndecim);

%% Reconstructing the force with the IDIM-LS estimates + statistics
% Estimated force
y_LS = X_IDM_0*Beta_LS(:,1);
% IDIM-LS error
error_LS = y_IDM_0 - y_LS;
rel_err_LS = 100*norm(error_LS)/norm(y_IDM_0);
% std deviations of IDIM-LS estimates
Beta_LS(:,2) = std(error_LS)*sqrt(diag(inv(X_IDM_0'*X_IDM_0)));
Beta_LS(:,3) = 100*Beta_LS(:,2)./abs(Beta_LS(:,1));
% norm of IDIM-LS error and IDIM-LS relative error
R_LS(1,1) = norm(error_LS);
R_LS(2,1) = 100*norm(error_LS)/norm(y_IDM_0);
% standard deviation of IDIM-LS error
R_LS(3,1) = std(error_LS);

%% Plot and display results
% plot
figure,
plot(t_decim,y_IDM_0,'b','LineWidth',2),hold on,
plot(t_decim,y_LS,'--r','LineWidth',3),hold on,
plot(t_decim,error_LS,'-.k','LineWidth',1),grid
title(' Cross test validation ')
ylabel(' N '),xlabel(' Time (s)')
legend(' Measurements ',' Reconstructed force ',' Error ')

% display absolute value of diagonal elements of R
disp(' ')
disp('Absolute value of the diagonal element of R')
for lineX = 1 : nligneX
    disp(['Parameter ',params(lineX,:),' : ',nu2stab(abs_diag_rx(lineX))])
end
disp(['Condition number of the observation matrix: ',num2str(cond(X_IDM_0))])

% display features of error
disp(' ')
disp(['Error norm : ',num2str(R_LS(1,1))])
disp(['Relative error (%): ',num2str(R_LS(2,1))])
disp(['Deviation of error: ',num2str(R_LS(3,1))])
disp(' ')
