%% FDIA_IEEE14_Detection_Localization.m
% Complex standalone MATLAB R2024a simulation for false data injection attack
% detection and localization in the IEEE 14-bus power system.
%
% Key features:
%   1. No MATPOWER dependency and no external files.
%   2. IEEE 14-bus bus/generator/branch data embedded internally.
%   3. Explicit Ybus construction, including transformer tap ratios.
%   4. Newton-Raphson AC load flow from scratch.
%   5. Synthetic noisy SCADA/PMU-style voltage, angle, P, and Q measurements.
%   6. Coordinated FDIA on selected buses during a known attack window.
%   7. Physics-informed residual score for attack detection.
%   8. Bus-level residual score for attack localization.
%   9. Command-window metrics and publication-style plots.
%
% Academic use only: this script is a synthetic benchmark demonstration on a
% public IEEE test system, intended for research on power-system resilience.

clear; clc; close all;
rng(42, 'twister');

%% ------------------------- User Configuration -------------------------
cfg.baseFrequencyHz       = 50;
cfg.nSamples              = 144;        % 24 h with 10-minute resolution
cfg.attackStartSample     = 60;
cfg.attackEndSample       = 95;
cfg.attackBuses           = [5 9 14];    % coordinated multi-bus FDIA
cfg.topK                  = numel(cfg.attackBuses);

% Measurement noise levels in per-unit/radian quantities.
cfg.sigmaV                = 0.0020;             % voltage magnitude noise, pu
cfg.sigmaTheta            = deg2rad(0.05);      % angle noise, rad
cfg.sigmaP                = 0.0200;             % active injection noise, pu
cfg.sigmaQ                = 0.0200;             % reactive injection noise, pu

% FDIA magnitude. The attack intentionally corrupts voltage/angle and partly
% compensates P/Q to imitate a semi-stealthy coordinated manipulation.
cfg.attackVoltageBiasPU   = 0.018;
cfg.attackVoltageRampPU   = 0.012;
cfg.attackAngleBiasDeg    = 1.25;
cfg.attackAngleRampDeg    = 0.75;
cfg.attackStealthFraction = 0.65;       % 0 = obvious, 1 = more AC-consistent
cfg.attackInjectionBiasPU = 0.015;

% Newton-Raphson options.
opts.maxIter              = 30;
opts.tolerance            = 1e-10;
opts.verbose              = false;

%% ----------------------- Load IEEE 14-Bus System ----------------------
[baseMVA, bus, branch, gen] = loadIEEE14Case();
Ybus = buildYbus(baseMVA, bus, branch);

nBus = size(bus, 1);
nT   = cfg.nSamples;
timeHour = linspace(0, 24, nT);
attackActive = false(1, nT);
attackActive(cfg.attackStartSample:cfg.attackEndSample) = true;
attackStartHour = timeHour(cfg.attackStartSample);
attackEndHour   = timeHour(cfg.attackEndSample);

%% ---------------------- Dynamic Load-Flow Simulation ------------------
% Daily loading pattern: morning/evening peaks plus a mild stochastic term.
loadScale = 1.00 ...
    + 0.12*sin(2*pi*(timeHour - 7)/24) ...
    + 0.06*sin(4*pi*(timeHour - 18)/24) ...
    + 0.015*randn(1, nT);
loadScale = max(loadScale, 0.72);
loadScale = min(loadScale, 1.30);

cleanVm      = zeros(nBus, nT);
cleanVa      = zeros(nBus, nT);
cleanP       = zeros(nBus, nT);
cleanQ       = zeros(nBus, nT);
iterCount    = zeros(1, nT);
finalMismatch = zeros(1, nT);
exampleMismatchHistory = [];

for k = 1:nT
    [pf, mismatchHistory] = solveNRPowerFlow(baseMVA, bus, gen, Ybus, loadScale(k), opts);

    if ~pf.converged
        error('Newton-Raphson did not converge at sample %d. Final mismatch = %.3e', ...
            k, pf.finalMismatch);
    end

    cleanVm(:, k) = pf.Vm;
    cleanVa(:, k) = pf.Va;
    cleanP(:, k)  = pf.Pinj;
    cleanQ(:, k)  = pf.Qinj;
    iterCount(k)  = pf.iterations;
    finalMismatch(k) = pf.finalMismatch;

    if k == round(nT/2)
        exampleMismatchHistory = mismatchHistory;
    end
end

%% -------------------------- Measurement Layer -------------------------
% Synthetic measurement vector at every time sample:
%   z(k) = [Vm; theta; P_inj; Q_inj] + measurement noise.
measVm = cleanVm + cfg.sigmaV     * randn(nBus, nT);
measVa = cleanVa + cfg.sigmaTheta * randn(nBus, nT);
measP  = cleanP  + cfg.sigmaP     * randn(nBus, nT);
measQ  = cleanQ  + cfg.sigmaQ     * randn(nBus, nT);

fdiaVm = measVm;
fdiaVa = measVa;
fdiaP  = measP;
fdiaQ  = measQ;

%% ----------------------------- FDIA Model -----------------------------
% Coordinated attack: buses 5, 9, and 14 are modified during the attack
% interval. P/Q measurements are partly modified toward the AC-consistent
% model created by the attacked voltage/angle state. This produces a more
% difficult case than a simple random spike attack.
for k = 1:nT
    if attackActive(k)
        alpha = (k - cfg.attackStartSample) / max(1, cfg.attackEndSample - cfg.attackStartSample);
        alpha = min(max(alpha, 0), 1);

        target = cfg.attackBuses(:);
        deltaV = cfg.attackVoltageBiasPU + cfg.attackVoltageRampPU * alpha;
        deltaA = deg2rad(cfg.attackAngleBiasDeg + cfg.attackAngleRampDeg * alpha);

        fdiaVm(target, k) = fdiaVm(target, k) + deltaV;
        fdiaVa(target, k) = fdiaVa(target, k) + deltaA;

        [PmodelCleanMeas, QmodelCleanMeas] = calcPowerInjection(Ybus, measVm(:, k), measVa(:, k));
        [PmodelAttacked,  QmodelAttacked]  = calcPowerInjection(Ybus, fdiaVm(:, k), fdiaVa(:, k));

        dP = PmodelAttacked(target) - PmodelCleanMeas(target);
        dQ = QmodelAttacked(target) - QmodelCleanMeas(target);

        fdiaP(target, k) = fdiaP(target, k) ...
            + cfg.attackStealthFraction * dP ...
            + cfg.attackInjectionBiasPU * (1 + 0.25*alpha);

        fdiaQ(target, k) = fdiaQ(target, k) ...
            + cfg.attackStealthFraction * dQ ...
            - 0.75 * cfg.attackInjectionBiasPU * (1 + 0.20*alpha);
    end
end

%% ---------------------- Detection and Localization --------------------
% Physics-informed consistency residual:
%   rP_i(k) = [Pmeas_i(k) - P_AC_i(Vmeas, theta_meas)] / sigmaP
%   rQ_i(k) = [Qmeas_i(k) - Q_AC_i(Vmeas, theta_meas)] / sigmaQ
%   localScore_i(k) = sqrt((rP_i^2 + rQ_i^2)/2)
%   globalScore(k) = RMS_i(localScore_i(k))
localScore = zeros(nBus, nT);
globalScore = zeros(1, nT);
residualP = zeros(nBus, nT);
residualQ = zeros(nBus, nT);

for k = 1:nT
    [Pmodel, Qmodel] = calcPowerInjection(Ybus, fdiaVm(:, k), fdiaVa(:, k));

    residualP(:, k) = (fdiaP(:, k) - Pmodel) / cfg.sigmaP;
    residualQ(:, k) = (fdiaQ(:, k) - Qmodel) / cfg.sigmaQ;

    localScore(:, k) = sqrt(0.5 * (residualP(:, k).^2 + residualQ(:, k).^2));
    globalScore(k) = sqrt(mean(localScore(:, k).^2));
end

trainingSamples = 1:floor(0.30*nT);   % normal samples before attack
threshold = robustThreshold(globalScore(trainingSamples), 6.0);
detected = globalScore > threshold;

metrics = computeBinaryMetrics(attackActive, detected);

% Aggregate bus localization scores during detected attack samples. If the
% detector misses all attack samples, fall back to the true attack interval
% so that the localization diagnostic remains interpretable.
localizationWindow = detected & attackActive;
if ~any(localizationWindow)
    localizationWindow = attackActive;
end

aggregateScore = mean(localScore(:, localizationWindow), 2);
[~, ranking] = sort(aggregateScore, 'descend');
predictedBuses = sort(ranking(1:cfg.topK));

trueBusMask = false(nBus, 1);
predBusMask = false(nBus, 1);
trueBusMask(cfg.attackBuses) = true;
predBusMask(predictedBuses) = true;

locTP = sum(trueBusMask & predBusMask);
locFP = sum(~trueBusMask & predBusMask);
locFN = sum(trueBusMask & ~predBusMask);
localizationPrecision = locTP / max(locTP + locFP, 1);
localizationRecall    = locTP / max(locTP + locFN, 1);
localizationF1        = 2 * localizationPrecision * localizationRecall / ...
                        max(localizationPrecision + localizationRecall, 1e-12);

firstDetectionIndex = find(detected & attackActive, 1, 'first');
if isempty(firstDetectionIndex)
    detectionDelaySamples = NaN;
    detectionDelayHours = NaN;
else
    detectionDelaySamples = firstDetectionIndex - cfg.attackStartSample;
    detectionDelayHours = timeHour(firstDetectionIndex) - attackStartHour;
end

%% ------------------------- Command-Window Output ----------------------
fprintf('\n============================================================\n');
fprintf(' IEEE 14-Bus FDIA Detection and Localization Summary\n');
fprintf('============================================================\n');
fprintf('Base MVA                         : %.1f\n', baseMVA);
fprintf('Number of buses                   : %d\n', nBus);
fprintf('Number of samples                 : %d\n', nT);
fprintf('Attack window                     : samples %d to %d, hours %.2f to %.2f\n', ...
    cfg.attackStartSample, cfg.attackEndSample, attackStartHour, attackEndHour);
fprintf('True attacked buses               : %s\n', mat2str(cfg.attackBuses));
fprintf('Predicted top-%d localized buses   : %s\n', cfg.topK, mat2str(predictedBuses(:).'));
fprintf('Detection threshold               : %.4f\n', threshold);
fprintf('First detection delay             : %g samples, %.3f hours\n', ...
    detectionDelaySamples, detectionDelayHours);
fprintf('Mean NR iteration count           : %.2f\n', mean(iterCount));
fprintf('Worst final NR mismatch           : %.3e pu\n', max(finalMismatch));

metricName = {'TP'; 'TN'; 'FP'; 'FN'; 'Accuracy'; 'Precision'; 'Recall'; ...
              'F1'; 'FPR'; 'FNR'; 'LocalizationPrecision'; ...
              'LocalizationRecall'; 'LocalizationF1'};
metricValue = [metrics.TP; metrics.TN; metrics.FP; metrics.FN; ...
               metrics.accuracy; metrics.precision; metrics.recall; ...
               metrics.F1; metrics.FPR; metrics.FNR; ...
               localizationPrecision; localizationRecall; localizationF1];
summaryTable = table(metricName, metricValue, 'VariableNames', {'Metric', 'Value'});
disp(summaryTable);

fprintf('Top bus localization ranking by aggregate score:\n');
for r = 1:min(8, nBus)
    fprintf('  Rank %2d: Bus %2d | score = %.4f\n', r, ranking(r), aggregateScore(ranking(r)));
end

%% ------------------------------- Plots --------------------------------
makePlots(timeHour, loadScale, iterCount, exampleMismatchHistory, ...
    cleanVm, fdiaVm, globalScore, threshold, localScore, aggregateScore, ...
    cfg, attackStartHour, attackEndHour, predictedBuses);

%% =========================== Local Functions ===========================
function [baseMVA, bus, branch, gen] = loadIEEE14Case()
% Internal IEEE 14-bus test case. Numerical data are the standard IEEE 14-bus
% benchmark values commonly distributed with power-flow test cases.
%
% bus columns:
%   1 bus_i | 2 type: 1 PQ, 2 PV, 3 slack | 3 Pd MW | 4 Qd MVAr |
%   5 Gs | 6 Bs | 7 Vm0 pu | 8 Va0 deg
%
% gen columns:
%   1 bus_i | 2 Pg MW | 3 Qg MVAr | 4 Vg pu | 5 status
%
% branch columns:
%   1 from | 2 to | 3 r pu | 4 x pu | 5 b pu | 6 tap | 7 shift deg | 8 status

baseMVA = 100;

bus = [
     1 3   0.0   0.0 0 0 1.060   0.00;
     2 2  21.7  12.7 0 0 1.045  -4.98;
     3 2  94.2  19.0 0 0 1.010 -12.72;
     4 1  47.8  -3.9 0 0 1.019 -10.33;
     5 1   7.6   1.6 0 0 1.020  -8.78;
     6 2  11.2   7.5 0 0 1.070 -14.22;
     7 1   0.0   0.0 0 0 1.062 -13.37;
     8 2   0.0   0.0 0 0 1.090 -13.36;
     9 1  29.5  16.6 0 0 1.056 -14.94;
    10 1   9.0   5.8 0 0 1.051 -15.10;
    11 1   3.5   1.8 0 0 1.057 -14.79;
    12 1   6.1   1.6 0 0 1.055 -15.07;
    13 1  13.5   5.8 0 0 1.050 -15.16;
    14 1  14.9   5.0 0 0 1.036 -16.04];

gen = [
     1 232.4 -16.9 1.060 1;
     2  40.0  42.4 1.045 1;
     3   0.0  23.4 1.010 1;
     6   0.0  12.2 1.070 1;
     8   0.0  17.4 1.090 1];

branch = [
     1  2 0.01938 0.05917 0.0528 0     0 1;
     1  5 0.05403 0.22304 0.0492 0     0 1;
     2  3 0.04699 0.19797 0.0438 0     0 1;
     2  4 0.05811 0.17632 0.0340 0     0 1;
     2  5 0.05695 0.17388 0.0346 0     0 1;
     3  4 0.06701 0.17103 0.0128 0     0 1;
     4  5 0.01335 0.04211 0.0000 0     0 1;
     4  7 0.00000 0.20912 0.0000 0.978 0 1;
     4  9 0.00000 0.55618 0.0000 0.969 0 1;
     5  6 0.00000 0.25202 0.0000 0.932 0 1;
     6 11 0.09498 0.19890 0.0000 0     0 1;
     6 12 0.12291 0.25581 0.0000 0     0 1;
     6 13 0.06615 0.13027 0.0000 0     0 1;
     7  8 0.00000 0.17615 0.0000 0     0 1;
     7  9 0.00000 0.11001 0.0000 0     0 1;
     9 10 0.03181 0.08450 0.0000 0     0 1;
     9 14 0.12711 0.27038 0.0000 0     0 1;
    10 11 0.08205 0.19207 0.0000 0     0 1;
    12 13 0.22092 0.19988 0.0000 0     0 1;
    13 14 0.17093 0.34802 0.0000 0     0 1];
end

function Ybus = buildYbus(~, bus, branch)
% Explicit bus admittance matrix construction.

nBus = size(bus, 1);
Ybus = complex(zeros(nBus, nBus));

for ell = 1:size(branch, 1)
    fromBus = branch(ell, 1);
    toBus   = branch(ell, 2);
    r       = branch(ell, 3);
    x       = branch(ell, 4);
    b       = branch(ell, 5);
    tapMag  = branch(ell, 6);
    shiftDeg = branch(ell, 7);
    status  = branch(ell, 8);

    if status == 0
        continue;
    end

    z = complex(r, x);
    if abs(z) < eps
        error('Invalid zero branch impedance at branch %d.', ell);
    end

    ySeries = 1 / z;
    yShunt  = 1i * b / 2;

    if tapMag == 0
        tap = 1.0;
    else
        tap = tapMag * exp(1i * deg2rad(shiftDeg));
    end

    Yff = (ySeries + yShunt) / (tap * conj(tap));
    Yft = -ySeries / conj(tap);
    Ytf = -ySeries / tap;
    Ytt = ySeries + yShunt;

    Ybus(fromBus, fromBus) = Ybus(fromBus, fromBus) + Yff;
    Ybus(fromBus, toBus)   = Ybus(fromBus, toBus)   + Yft;
    Ybus(toBus, fromBus)   = Ybus(toBus, fromBus)   + Ytf;
    Ybus(toBus, toBus)     = Ybus(toBus, toBus)     + Ytt;
end

% Add fixed bus shunts if present in bus columns 5 and 6.
Gs = bus(:, 5);
Bs = bus(:, 6);
Ybus = Ybus + diag(complex(Gs, Bs));
end

function [pf, mismatchHistory] = solveNRPowerFlow(baseMVA, bus0, gen, Ybus, loadScale, opts)
% Newton-Raphson AC power-flow solver in polar coordinates.

bus = bus0;
bus(:, 3) = bus(:, 3) * loadScale;
bus(:, 4) = bus(:, 4) * loadScale;

PQ    = 1;
PV    = 2;
SLACK = 3;

busType = bus(:, 2);
slack = find(busType == SLACK);
pv    = find(busType == PV);
pq    = find(busType == PQ);
pvpq  = [pv; pq];

Vm = bus(:, 7);
Va = deg2rad(bus(:, 8));

activeGen = gen(:, 5) > 0;
for g = find(activeGen).'
    genBus = gen(g, 1);
    Vm(genBus) = gen(g, 4);
end

Pd = bus(:, 3) / baseMVA;
Qd = bus(:, 4) / baseMVA;
Pg = zeros(size(bus, 1), 1);
Qg = zeros(size(bus, 1), 1);

for g = find(activeGen).'
    genBus = gen(g, 1);
    Pg(genBus) = Pg(genBus) + gen(g, 2) / baseMVA;
    Qg(genBus) = Qg(genBus) + gen(g, 3) / baseMVA;
end

Pspec = Pg - Pd;
Qspec = Qg - Qd;

converged = false;
mismatchHistory = zeros(opts.maxIter, 1);
iterations = opts.maxIter;

for iter = 1:opts.maxIter
    [Pcalc, Qcalc] = calcPowerInjection(Ybus, Vm, Va);

    dP = Pspec(pvpq) - Pcalc(pvpq);
    dQ = Qspec(pq)   - Qcalc(pq);
    mismatch = [dP; dQ];

    mismatchNorm = max(abs(mismatch));
    mismatchHistory(iter) = mismatchNorm;

    if opts.verbose
        fprintf('  NR iter %2d | max mismatch = %.3e\n', iter, mismatchNorm);
    end

    if mismatchNorm < opts.tolerance
        converged = true;
        iterations = iter;
        mismatchHistory = mismatchHistory(1:iter);
        break;
    end

    J = buildNRJacobian(Ybus, Vm, Va, pvpq, pq, Pcalc, Qcalc);
    dx = J \ mismatch;

    nAngle = numel(pvpq);
    dTheta = dx(1:nAngle);
    dVm    = dx(nAngle+1:end);

    Va(pvpq) = Va(pvpq) + dTheta;
    Vm(pq)   = Vm(pq)   + dVm;

    % Keep generator voltage setpoints fixed at PV and slack buses.
    for g = find(activeGen).'
        genBus = gen(g, 1);
        if busType(genBus) == PV || busType(genBus) == SLACK
            Vm(genBus) = gen(g, 4);
        end
    end
end

[Pfinal, Qfinal] = calcPowerInjection(Ybus, Vm, Va);

if ~converged
    mismatchHistory = mismatchHistory(:);
    mismatchHistory = mismatchHistory(mismatchHistory > 0);
end

pf.Vm = Vm;
pf.Va = Va;
pf.Pinj = Pfinal;
pf.Qinj = Qfinal;
pf.converged = converged;
pf.iterations = iterations;
pf.finalMismatch = mismatchHistory(end);
pf.slackBus = slack;
end

function J = buildNRJacobian(Ybus, Vm, Va, pvpq, pq, Pcalc, Qcalc)
% Full Newton-Raphson Jacobian submatrices:
%   J = [dP/dtheta  dP/dV;
%        dQ/dtheta  dQ/dV]

nBus = numel(Vm);
G = real(Ybus);
B = imag(Ybus);

H = zeros(nBus, nBus);  % dP/dtheta
N = zeros(nBus, nBus);  % dP/dV
M = zeros(nBus, nBus);  % dQ/dtheta
L = zeros(nBus, nBus);  % dQ/dV

for i = 1:nBus
    for k = 1:nBus
        thetaIK = Va(i) - Va(k);

        if i == k
            H(i, k) = -Qcalc(i) - B(i, i) * Vm(i)^2;
            N(i, k) =  Pcalc(i) / Vm(i) + G(i, i) * Vm(i);
            M(i, k) =  Pcalc(i) - G(i, i) * Vm(i)^2;
            L(i, k) =  Qcalc(i) / Vm(i) - B(i, i) * Vm(i);
        else
            H(i, k) = Vm(i) * Vm(k) * (G(i, k) * sin(thetaIK) - B(i, k) * cos(thetaIK));
            N(i, k) = Vm(i) * (G(i, k) * cos(thetaIK) + B(i, k) * sin(thetaIK));
            M(i, k) = -Vm(i) * Vm(k) * (G(i, k) * cos(thetaIK) + B(i, k) * sin(thetaIK));
            L(i, k) = Vm(i) * (G(i, k) * sin(thetaIK) - B(i, k) * cos(thetaIK));
        end
    end
end

J11 = H(pvpq, pvpq);
J12 = N(pvpq, pq);
J21 = M(pq,   pvpq);
J22 = L(pq,   pq);

J = [J11 J12; J21 J22];
end

function [P, Q] = calcPowerInjection(Ybus, Vm, Va)
% AC nodal complex-power injection from state and Ybus.
V = Vm(:) .* exp(1i * Va(:));
I = Ybus * V;
S = V .* conj(I);
P = real(S);
Q = imag(S);
end

function threshold = robustThreshold(x, kappa)
% Median/MAD threshold without relying on the Statistics Toolbox.
x = x(:);
medx = median(x);
madx = median(abs(x - medx));
robustSigma = 1.4826 * madx;

if robustSigma < 1e-12
    robustSigma = std(x);
end

threshold = medx + kappa * robustSigma;
threshold = max(threshold, 1.10 * max(x));
end

function metrics = computeBinaryMetrics(truth, detected)
truth = logical(truth(:));
detected = logical(detected(:));

TP = sum(truth & detected);
TN = sum(~truth & ~detected);
FP = sum(~truth & detected);
FN = sum(truth & ~detected);

small = 1e-12;
metrics.TP = TP;
metrics.TN = TN;
metrics.FP = FP;
metrics.FN = FN;
metrics.accuracy  = (TP + TN) / max(TP + TN + FP + FN, 1);
metrics.precision = TP / max(TP + FP, 1);
metrics.recall    = TP / max(TP + FN, 1);
metrics.F1        = 2 * metrics.precision * metrics.recall / ...
                    max(metrics.precision + metrics.recall, small);
metrics.FPR       = FP / max(FP + TN, 1);
metrics.FNR       = FN / max(FN + TP, 1);
end

function makePlots(timeHour, loadScale, iterCount, mismatchHistory, ...
    cleanVm, fdiaVm, globalScore, threshold, localScore, aggregateScore, ...
    cfg, attackStartHour, attackEndHour, predictedBuses)

figure('Color', 'w', 'Name', 'Load Profile and NR Convergence');
ax1 = subplot(2, 1, 1);
plot(timeHour, loadScale, 'LineWidth', 1.6);
grid on; xlabel('Time (h)'); ylabel('Load scale');
title('Dynamic Load Profile Used for IEEE 14-Bus Simulation');
addAttackPatch(ax1, attackStartHour, attackEndHour);

ax2 = subplot(2, 1, 2);
semilogy(1:numel(mismatchHistory), mismatchHistory, '-o', 'LineWidth', 1.4);
grid on; xlabel('NR iteration'); ylabel('Max mismatch (pu)');
title(sprintf('Representative Newton-Raphson Convergence | Mean iterations = %.2f', mean(iterCount)));

figure('Color', 'w', 'Name', 'FDIA Global Detection Score');
ax3 = axes;
plot(timeHour, globalScore, 'LineWidth', 1.8); hold on;
yline(threshold, '--', 'Detection threshold', 'LineWidth', 1.4);
grid on; xlabel('Time (h)'); ylabel('Global residual score');
title('Physics-Informed FDIA Detection Score');
addAttackPatch(ax3, attackStartHour, attackEndHour);
legend('Global residual score', 'Detection threshold', 'Location', 'best');

busToShow = cfg.attackBuses(1);
figure('Color', 'w', 'Name', 'Clean vs Attacked Voltage Magnitude');
ax4 = axes;
plot(timeHour, cleanVm(busToShow, :), 'LineWidth', 1.6); hold on;
plot(timeHour, fdiaVm(busToShow, :), '--', 'LineWidth', 1.6);
grid on; xlabel('Time (h)'); ylabel('Voltage magnitude (pu)');
title(sprintf('Voltage Magnitude at Bus %d: Clean vs FDIA-Corrupted', busToShow));
addAttackPatch(ax4, attackStartHour, attackEndHour);
legend('Clean', 'FDIA-corrupted', 'Location', 'best');

figure('Color', 'w', 'Name', 'Bus-Level Localization Heatmap');
imagesc(timeHour, 1:size(localScore, 1), localScore);
set(gca, 'YDir', 'normal'); colorbar;
xlabel('Time (h)'); ylabel('Bus number');
title('Bus-Level Physics Residual Score for FDIA Localization');
hold on;
plot([attackStartHour attackStartHour], [0.5 size(localScore,1)+0.5], 'w--', 'LineWidth', 1.5);
plot([attackEndHour attackEndHour],     [0.5 size(localScore,1)+0.5], 'w--', 'LineWidth', 1.5);
for b = cfg.attackBuses(:).'
    plot(timeHour, b*ones(size(timeHour)), 'w:', 'LineWidth', 1.0);
end

figure('Color', 'w', 'Name', 'Aggregate Bus Localization Scores');
bar(1:numel(aggregateScore), aggregateScore, 0.75); hold on;
plot(cfg.attackBuses, aggregateScore(cfg.attackBuses), 'ro', 'MarkerSize', 8, 'LineWidth', 1.8);
plot(predictedBuses, aggregateScore(predictedBuses), 'kx', 'MarkerSize', 9, 'LineWidth', 1.8);
grid on; xlabel('Bus number'); ylabel('Aggregate localization score');
title('Aggregate Localization Ranking During Detected Attack Interval');
legend('All buses', 'True attacked buses', 'Predicted top-K buses', 'Location', 'best');
end

function addAttackPatch(ax, startHour, endHour)
% Adds a shaded attack interval without changing axis limits.
axes(ax); %#ok<LAXES>
yLimits = ylim(ax);
xPatch = [startHour endHour endHour startHour];
yPatch = [yLimits(1) yLimits(1) yLimits(2) yLimits(2)];
p = patch(ax, xPatch, yPatch, [1.0 0.85 0.85], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.35, 'HandleVisibility', 'off');
if exist('uistack', 'file') == 2
    uistack(p, 'bottom');
end
ylim(ax, yLimits);
end
