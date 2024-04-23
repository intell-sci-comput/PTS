README

This .zip file contains multiple files. Each of these files are briefly explained below.

0. Dataset document: EMPS_Description.pdf
	Document describing the system, the dataset, and an approach commonly used in robotics (IDIM-LS) where the system parameters are estimated through the estimation of an inverse model of the system.
1. Estimation dataset: DATA_EMPS.mat --> input Force (vir), output Position (qm). Also the reference position is included since the measurements were obtained in closed loop (qg).
2. Validation dataset: DATA_EMPS_PULSES.mat --> input Force (vir), output Position (qm). Also the reference position, and applied force pulses are included since the measurements were obtained in closed loop (qg and pulses_N).
3. Script_IDIM_LS.m and Script IDIM_LS_Asym_Fric.m
	These matlab scripts estimate an inverse model of the system (from position, velocity and acceleration to force).
4. Script_IDIM_LS_CVT.m
	This matlab script allows one to perform a cross-validation of the obtained inverse model on the validation dataset (simulating the inverse model, target value = force).
5. Simulation_EMPS.m and Simulation_EMPS_Validation.m
	These matlab scripts simulate the forward model in the known feedback loop (from reference position (and force pulses) to force, position, velocity and acceleration) using the parameters obtained by IDIM_LS. This simulation is performed using the simulink files  Simulink_EMPS_Rigide.slx, Simulink_EMPS_Rigide_Pulse.slx
6. Supporting matlab and simulink files (nu2stab.m, diffcent.m, Simulink_EMPS_Rigide.slx, Simulink_EMPS_Rigide_Pulse.slx)