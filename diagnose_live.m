%% diagnose_live.m
%  End-to-end demonstration: run the converter with a fault, capture the
%  currents, and ask the trained network which switch failed.
%
%  Two modes, because they answer different questions.
%
%  ORACLE WINDOW


clear; clc;

MDL       = 'new_t_type_three_level';
NETFILE   = 'diagnosis_net.mat';
NOISE_AUG = 2;        % % of full scale, added during training.
                      % The sweep showed this costs nothing on clean
                      % data and prevents collapse on noisy data.
FS        = 300;      % A, sensor full scale
STEP      = 5;       % sliding window step, samples (0.8 ms at 25 kHz)

%% ---- the fault to simulate -------------------------------------------
%  Change these. faultLoc = 0 is healthy, 1..9 is an OCF in S1..S9.
faultLoc  = 8;
tFault    = 0.100;
mi        = 0.88;
Pload     = 95e3;
Qload     = Pload*0.25;
Vdc       = 1010;
portOrder = [7 1 2 3 8 4 5 6 9];
f0        = 50;
Ts        = 1/25000;
N         = 1000;                 % 2 cycles at 50 Hz, 25 kHz

%% ---- network: load, or train and save --------------------------------
if isfile(NETFILE)
    load(NETFILE,'net','clsNames');
    fprintf('Loaded %s\n', NETFILE);
else
    fprintf('No saved network. Training one (config A, ~2 min)...\n');
    D = load('ttype_dataset_v3.mat');
    rng(1);
    s  = NOISE_AUG/100*FS;
    Xt = flat(normRec(D.Xtrain + s*randn(size(D.Xtrain))));
    Xv = flat(normRec(D.Xval   + s*randn(size(D.Xval))));
    Tt = categorical(D.Ytrain);  Tv = categorical(D.Yval);
    clsNames = categories(Tt);
    net = trainOne(Xt, Tt, Xv, Tv, [3*N 1 1], numel(D.classes));
    save(NETFILE,'net','clsNames');
    fprintf('Saved %s\n', NETFILE);
end

%% ---- run the converter ------------------------------------------------
load_system(MDL);
set_param(MDL,'ReturnWorkspaceOutputs','on');
set_param(MDL,'StopTime','0.2');
out = sim(MDL);
I   = out.Iabc;                       % 5001 x 3
t   = (0:size(I,1)-1)'*Ts;
fprintf('\nSimulated: faultLoc = %d at t = %.4f s\n', faultLoc, tFault);

%% ---- mode 1: oracle window --------------------------------------------
i0 = round((tFault - 1/f0)/Ts) + 1;
W  = I(i0:i0+N-1, :);
x  = flat(normRec([W(:,1); W(:,2); W(:,3)]'));
sc = minibatchpredict(net, x);
[conf, k] = max(sc);
pred = double(string(clsNames(k)));

fprintf('\n--- oracle window (fault instant known) ---\n');
fprintf('  predicted class %d  (%s)   confidence %.1f %%\n', ...
        pred, switchName(pred), 100*conf);
fprintf('  true class      %d  (%s)\n', faultLoc, switchName(faultLoc));
if pred == faultLoc, fprintf('  CORRECT\n'); else, fprintf('  WRONG\n'); end

fprintf('\n  top three:\n');
[ss, oo] = sort(sc, 'descend');
for r = 1:3
    fprintf('    class %s  %5.1f %%\n', string(clsNames(oo(r))), 100*ss(r));
end

%% ---- mode 2: sliding window -------------------------------------------
%  No knowledge of tFault. Classify every window position in the run.
starts = 1 : STEP : (size(I,1) - N + 1);
nW = numel(starts);
Xs = zeros(3*N, 1, 1, nW, 'single');
for w = 1:nW
    Ww = I(starts(w):starts(w)+N-1, :);
    Xs(:,1,1,w) = single(normRec([Ww(:,1); Ww(:,2); Ww(:,3)]'))';
end
S = minibatchpredict(net, Xs);              % nW x nCls
[confS, kS] = max(S, [], 2);
predS = double(string(clsNames(kS)));
tEnd  = t(starts + N - 1);                  % time at the window's end

fprintf('\n--- sliding window (fault instant unknown) ---\n');

% before the fault enters any window, the answer should be class 0
preFault = tEnd < tFault;
fprintf('  windows entirely before the fault : %3d, called healthy %5.1f %%\n', ...
        sum(preFault), 100*mean(predS(preFault)==0));

% once the window is fully past the fault, the answer should be faultLoc
fullPost = t(starts) > tFault;
if any(fullPost)
    fprintf('  windows entirely after the fault  : %3d, called correctly %5.1f %%\n', ...
            sum(fullPost), 100*mean(predS(fullPost)==faultLoc));
end

% detection latency
if faultLoc > 0
    hit = find(tEnd > tFault & predS == faultLoc, 1);
    if isempty(hit)
        fprintf('  never reached the correct class.\n');
    else
        fprintf('  first correct at  %5.2f ms after the fault\n', ...
                1e3*(tEnd(hit)-tFault));
        % first index from which it stays correct to the end
        okRun = find(tEnd > tFault);
        stay  = okRun(find(arrayfun(@(a) all(predS(a:end)==faultLoc), okRun), 1));
        if ~isempty(stay)
            fprintf('  stays correct from %5.2f ms after the fault\n', ...
                    1e3*(tEnd(stay)-tFault));
        else
            fprintf('  never settles on the correct class.\n');
        end
    end
end

%% ---- figure ------------------------------------------------------------
figure('Color','w');

subplot(3,1,1)
plot(t*1e3, I); grid on; hold on
xline(tFault*1e3,'r--','fault');
ylabel('current (A)'); title(sprintf('OCF in S%d', faultLoc)); xlim([0 200])

subplot(3,1,2)
plot(tEnd*1e3, predS, '.-'); grid on; hold on
yline(faultLoc,'g-','true class');
xline(tFault*1e3,'r--');
ylabel('predicted class'); ylim([-0.5 9.5]); yticks(0:9); xlim([0 200])

subplot(3,1,3)
plot(tEnd*1e3, 100*confS, '.-'); grid on; hold on
xline(tFault*1e3,'r--');
ylabel('confidence (%)'); xlabel('window end time (ms)'); ylim([0 105]); xlim([0 200])

%% ---- reading the sliding result ---------------------------------------
%  Expect three regions:
%    1. Before the fault enters the window - should read class 0.
%    2. A transition while the window straddles the fault. The network
%       never saw these mixtures, so predictions here may be erratic
%       and the confidence may stay high while being wrong. That is
%       worth reporting: high confidence is not evidence of a correct
%       answer on inputs outside the training distribution.
%    3. Once the window is entirely post-fault, it should settle.
%
%  If region 3 is not reliable, the dataset construction is the cause,
%  not the network: no training record ever contained two full cycles
%  of post-fault current. Fixing that means rebuilding the dataset with
%  windows at varied offsets relative to the fault, which is a cheap
%  change to the collection script and a defensible contribution.

%% ---- helpers -----------------------------------------------------------
function s = switchName(c)
    if c == 0, s = 'healthy'; return; end
    names = ["S1 phase A upper","S2 phase A lower","S3 phase B upper", ...
             "S4 phase B lower","S5 phase C upper","S6 phase C lower", ...
             "S7 clamp A","S8 clamp B","S9 clamp C"];
    s = names(c);
end

function Z = normRec(M)
    Z = M ./ max(abs(M),[],2);
end

function out = flat(M)
    out = reshape(single(M)', [size(M,2) 1 1 size(M,1)]);
end

function net = trainOne(Xt, Tt, Xv, Tv, inSize, nCls)
    layers = [
        imageInputLayer(inSize,'Normalization','none')
        convolution2dLayer([51 1],15); reluLayer; batchNormalizationLayer
        maxPooling2dLayer([10 1],'Stride',[2 1],'Padding','same')
        convolution2dLayer([26 1],10); reluLayer; batchNormalizationLayer
        maxPooling2dLayer([10 1],'Stride',[2 1],'Padding','same')
        convolution2dLayer([16 1],5);  reluLayer; batchNormalizationLayer
        maxPooling2dLayer([10 1],'Stride',[2 1],'Padding','same')
        fullyConnectedLayer(500)
        fullyConnectedLayer(100)
        fullyConnectedLayer(nCls)
        softmaxLayer];
    opts = trainingOptions('adam','InitialLearnRate',1e-3,'MaxEpochs',30, ...
        'MiniBatchSize',32,'Shuffle','every-epoch','ValidationData',{Xv,Tv}, ...
        'ValidationFrequency',25,'ValidationPatience',8, ...
        'OutputNetwork','best-validation','Verbose',false,'Plots','none');
    net = trainnet(Xt, Tt, layers, "crossentropy", opts);
end
