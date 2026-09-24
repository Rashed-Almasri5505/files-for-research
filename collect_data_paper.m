%% collect_data_paper.m   (v2)
%  Data collection replicating:
%    M. Sahani, M. D. Siddique, P. Sundararajan, S. K. Panda,
%    "Deep Convolutional Neural Network Based Fault Detection and
%     Diagnosis Method for Three-Phase T-Type Converter", NPEC 2023.
%
%  PAPER SETTINGS USED HERE
%    Vdc          1000 V  (dc source / ESS)
%    Rated power  100 kW
%    Carrier      5 kHz
%    Fundamental  50 Hz
%    Sampling     25 kHz   <-- see note below
%    Window       2 cycles = 40 ms (1 cycle before + 1 cycle after fault)
%    Classes      10  (healthy + OCF in S1..S9)
%    Runs/class   250      -> 2500 runs total
%    Split        40% train / 20% validation / 40% test
%
%  NOTE ON SAMPLING RATE
%    The paper's text says 5 kHz, but its Table I implies a 3000-sample
%    network input (2950 + 51 - 1 = 3000). Three phases x 1000 samples
%    = 3000 requires 25 kHz over a 40 ms window. The companion
%    six-switch paper states 25 kHz. So 25 kHz is used here.
%
%  CHANGE IN v2
%    The Three-Phase Series RLC Load block is specified by POWER, not by
%    R and L. v1 tried to set Rload/Lload on it, which silently did
%    nothing - every v1 run used the same fixed load. v2 sets the active
%    and reactive power instead, which the block does accept.
%
%  BEFORE RUNNING - required model settings
%    Sine Wave1/2/3   Frequency   = 100*pi
%    Sine Wave1/2/3   Amplitude   = mi
%    Repeating Seq x2 Time values = [0 0.5/5000 1/5000]
%    V1               Amplitude   = Vdc
%    3-Ph RLC Load    Configuration            = Y (floating)
%                     Nominal ph-ph voltage Vn = 555
%                     Nominal frequency fn     = 50
%                     "Specify PQ per phase"   = UNTICKED
%                     Active power P           = Pload
%                     Inductive reactive QL    = Qload
%                     Capacitive reactive Qc   = 0
%    To Workspace     Variable    = Iabc, Sample time = 1/25000,
%                                   Save format = Array, limit = inf
%
%  NOT SPECIFIED BY THE PAPER (chosen here - document these in your work)
%    The paper says it varies "modulation index, load parameters,
%    unbalancing of the load in each phase, fundamental and carrier
%    frequency" but gives no ranges, no counts, and no table. The ranges
%    below are my choice and must be reported as such.
%    Load unbalance is deliberately left OFF - see the note at the end.

clear; clc;

mdl = 'new_t_type_three_level';
load_system(mdl);
set_param(mdl,'ReturnWorkspaceOutputs','on');
set_param(mdl,'StopTime','0.2');

% ---- gate ordering of the Subsystem output ports ----------------------
portOrder = [7 1 2 3 8 4 5 6 9];

% ---- fixed settings ---------------------------------------------------
f0    = 50;            % Hz   fundamental
fs    = 25000;         % Hz   sampling  -> 40 us
Ts    = 1/fs;          % s
Vdc0  = 1000;          % V    nominal dc link
Pn    = 100e3;         % W    rated active power

spc   = fs/f0;         % 500 samples per fundamental cycle
N     = 2*spc;         % 1000 samples per phase  (2 cycles)
Lin   = 3*N;           % 3000 -> length of the concatenated DCNN input

classes = 0:9;         % 0 = healthy, 1..9 = OCF in S1..S9
nRuns   = 250;         % paper: 250 signals per class
                       % SET THIS TO 2 FOR A QUICK TEST FIRST

randomiseFaultInstant = true;   % paper does not say; strongly recommended

%% ---- PRE-FLIGHT: are the variables actually reaching the blocks? -----
%  v1 was wasted because they were not. Two runs at different power must
%  give clearly different currents. Costs ~5 seconds, saves ~3 hours.
fprintf('Pre-flight check...\n');
mi = 0.95; Vdc = Vdc0; Qload = 20e3; faultLoc = 0; tFault = 0.10;

Pload = 100e3;  chk1 = max(abs( sim(mdl).Iabc(:) ));
Pload =  50e3;  chk2 = max(abs( sim(mdl).Iabc(:) ));

fprintf('   100 kW -> %.1f A peak\n    50 kW -> %.1f A peak\n', chk1, chk2);
if abs(chk1 - chk2) < 0.10*chk1
    error(['Load power is not reaching the model. Check that the load ' ...
           'block''s Active power field contains Pload, and that ' ...
           '"Specify PQ powers for each phase" is unticked.']);
end
fprintf('   OK - the load responds.\n\n');


qTest = [0.18 0.33];
pfTest = 1./sqrt(1+qTest.^2);
fprintf('   pf range will be %.3f .. %.3f\n', min(pfTest), max(pfTest));
if max(pfTest) > 0.99
    error('Power factor too close to 1 - inductance will vanish.');
end
%% ---- storage ---------------------------------------------------------
nTotal = numel(classes)*nRuns;
X    = zeros(nTotal, Lin);     % each row = [Ia Ib Ic] concatenated
Y    = zeros(nTotal, 1);       % class label 0..9
cond = repmat(struct('faultLoc',0,'tFault',0,'mi',0, ...
                     'Pload',0,'Qload',0,'Vdc',0), nTotal, 1);
idx  = 0;

rng(42);                       % reproducible dataset
t0 = tic;

for c = classes
    for k = 1:nRuns

        % ---- operating condition for this run ----------------------
        faultLoc = c;
        mi       = 0.80 + 0.15*rand;          % 0.80 .. 0.95 (linear)
        Pload    = Pn * (0.70 + 0.60*rand);   % 70 kW .. 130 kW
        Qload = Pload * (0.18 + 0.15*rand);      % pf ~0.93 .. 1.0
        Vdc      = Vdc0 * (0.95 + 0.10*rand); % +/-5 % dc link

        if randomiseFaultInstant
            tFault = 0.10 + rand/f0;          % anywhere in one cycle
        else
            tFault = 0.10;
        end

        % ---- run --------------------------------------------------
        out = sim(mdl);
        I   = out.Iabc;                        % [samples x 3]

        % ---- cut the 2-cycle window -------------------------------
        i0 = round((tFault - 1/f0)/Ts) + 1;    % start 1 cycle early
        if i0 < 1 || i0+N-1 > size(I,1)
            error('Window outside logged data. Check StopTime / tFault.');
        end
        W = I(i0 : i0+N-1, :);                 % [1000 x 3]

        % ---- store ------------------------------------------------
        idx = idx + 1;
        X(idx,:) = [W(:,1); W(:,2); W(:,3)]';  % concatenate -> 3000
        Y(idx)   = c;

        cond(idx).faultLoc = faultLoc;
        cond(idx).tFault   = tFault;
        cond(idx).mi       = mi;
        cond(idx).Pload    = Pload;
        cond(idx).Qload    = Qload;
        cond(idx).Vdc      = Vdc;

        if mod(idx,25)==0
            fprintf('%4d / %4d runs  (%.1f min elapsed)\n', ...
                    idx, nTotal, toc(t0)/60);
        end
    end
end
fprintf('Done in %.1f minutes.\n', toc(t0)/60);

%% ---- 40 / 20 / 40 split, stratified by class -------------------------
%  Every class is split separately so all three sets stay balanced.
trainIdx = []; valIdx = []; testIdx = [];
for c = classes
    r = find(Y==c);
    r = r(randperm(numel(r)));
    n = numel(r);
    nTr = round(0.40*n);
    nVa = round(0.20*n);
    trainIdx = [trainIdx; r(1:nTr)];
    valIdx   = [valIdx;   r(nTr+1 : nTr+nVa)];
    testIdx  = [testIdx;  r(nTr+nVa+1 : end)];
end

Xtrain = X(trainIdx,:);   Ytrain = Y(trainIdx);
Xval   = X(valIdx,:);     Yval   = Y(valIdx);
Xtest  = X(testIdx,:);    Ytest  = Y(testIdx);

fprintf('train %d   val %d   test %d\n', ...
        numel(Ytrain), numel(Yval), numel(Ytest));

save('ttype_dataset_v3.mat', ...
     'X','Y','cond', ...
     'Xtrain','Ytrain','Xval','Yval','Xtest','Ytest', ...
     'f0','fs','N','Lin','classes','portOrder');

%% ---- sanity checks ---------------------------------------------------
%  Run these by hand after the script finishes.
%
%    size(X)                        % nTotal x 3000
%    histcounts(Y, -0.5:1:9.5)      % equal count per class
%    any(~isfinite(X(:)))           % 0
%    [min(X(:)) max(X(:))]          % roughly -200 .. +200 A
%
%    % THE IMPORTANT ONE: did the load really vary?
%    pk = max(abs(X),[],2);  r0 = find(Y==0);
%    scatter([cond(r0).Pload], pk(r0));
%    xlabel('P_{load} (W)'); ylabel('peak current (A)'); grid on
%    % must slope UPWARD. A flat cloud means the load never changed.
%
%    % fault signatures
%    for c = [0 1 2 7]
%        r = find(Y==c); figure;
%        plot(reshape(X(r(1),:), N, 3)); grid on
%        legend('Ia','Ib','Ic'); title(sprintf('class %d', c));
%    end
%    % class 1 (S1): phase A loses its POSITIVE half
%    % class 2 (S2): phase A loses its NEGATIVE half
%    % class 7 (S7): phase A distorted around its zero crossings
%
%    % three-wire check - must be ~0 (order 1e-9), not tens of amps
%    r = find(Y==1); W = reshape(X(r(1),:), N, 3);
%    plot(sum(W,2)); title('Ia + Ib + Ic'); grid on

%% ---- WHY LOAD UNBALANCE IS NOT INCLUDED -----------------------------
%  The paper lists "unbalancing of the load in each phase" as one of the
%  varied conditions. It is left out on purpose.
%
%  An unbalanced load makes the three phase currents unequal - which is
%  also the main symptom of an open-circuit fault. So unbalance can mimic
%  a fault, and a healthy-but-unbalanced run labelled class 0 teaches the
%  network a contradiction.
%
%  To add it safely later:
%    - tick "Specify PQ powers for each phase" on the load block and set
%      Active powers to [Pa Pb Pc],
%    - keep the spread small relative to the asymmetry a real OCF causes,
%    - and apply the SAME spread to healthy and faulted runs, so
%      unbalance itself cannot be used as a shortcut cue.