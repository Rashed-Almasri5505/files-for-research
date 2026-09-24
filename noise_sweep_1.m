%% noise_sweep.m
%  How far does classification accuracy survive sensor noise?
%
%  The v3 dataset is perfectly clean, and every configuration reached
%  100 %, so the benchmark has no resolving power left. Adding noise
%  lowers the ceiling deliberately and turns a single number into a
%  curve, which can separate methods that a single number cannot.
%
%  -------------------------------------------------------------------
%  NOISE MODEL
%
%  Additive white Gaussian noise, standard deviation fixed relative to
%  the SENSOR's full-scale rating - not to each record's own amplitude.
%
%  Why: current transducers are specified as a percentage of rated
%  current, not of the measured value. A 300 A sensor has the same
%  absolute noise floor whether it reads 120 A or 215 A.
%
%  This also matters for the experiment. Load power varies 70-130 kW,
%  so a fixed absolute noise floor gives light-load records a worse
%  signal-to-noise ratio automatically - realistic, and harder.
%  Percentage-of-record noise would instead give every record the same
%  SNR and would encode load level into the noise amplitude, which is
%  a new leak.
%
%  ORDERING: noise is added to the raw amps BEFORE per-record peak
%  normalisation. Normalising first would change the SNR the model
%  sees and make the sweep meaningless.
%  -------------------------------------------------------------------
%
%  TWO TRAINING MODES, because they answer different questions:
%    clean-trained  - train on perfect data, test on noisy data.
%                     The simulation-to-hardware gap. Harsher.
%    noise-trained  - noise present in training and test.
%                     Is the task solvable at all at this noise level?
%
%  The nearest-centroid baseline is evaluated at every level too, so
%  the CNN's advantage can be tracked as a function of noise rather
%  than quoted at one operating point.
%
%  Uses input configuration B (1000 x 3). It matched A and C exactly
%  on accuracy and trained in about half the time.

clear; clc;

load('ttype_dataset_v3.mat');        % X, Y, Xtrain/val/test, N, classes

FS       = 300;                      % A, assumed sensor full-scale rating
NOISEPCT = [0 0.5 1 2 5 10];         % sigma as % of full scale
SEEDS    = 1:3;
NOISESEED = 7;                       % fixed, so the noise is reproducible

nCls  = numel(classes);
nLvl  = numel(NOISEPCT);
Ttr   = categorical(Ytrain);
Tva   = categorical(Yval);
Tte   = categorical(Ytest);
clsNames = categories(Ttr);

fprintf('sigma at each level (A): ');
fprintf('%.1f  ', NOISEPCT/100*FS);
fprintf('\n\n');

%% ---- pre-generate noise so every method sees the SAME corruption ----
%  Without this, the CNN and the baseline would be scored on different
%  noise draws and the comparison would carry that difference.
rng(NOISESEED);
Ntr = randn(size(Xtrain), 'like', Xtrain);
Nva = randn(size(Xval),   'like', Xval);
Nte = randn(size(Xtest),  'like', Xtest);

accCNN_clean = zeros(nLvl, numel(SEEDS));   % trained clean, tested noisy
accCNN_noisy = zeros(nLvl, numel(SEEDS));   % trained noisy, tested noisy
accBase      = zeros(nLvl, 1);              % baseline, no seed dependence

%% ---- baseline at every noise level -----------------------------------
for li = 1:nLvl
    s = NOISEPCT(li)/100*FS;
    accBase(li) = centroidAcc(Xtrain + s*Ntr, Ytrain, ...
                              Xtest  + s*Nte, Ytest, N, nCls);
    fprintf('baseline   %4.1f %% noise -> %6.2f %%\n', NOISEPCT(li), 100*accBase(li));
end
fprintf('\n');

%% ---- CNN, clean-trained ----------------------------------------------
%  Train once per seed on clean data, then evaluate at every level.
for si = 1:numel(SEEDS)
    rng(SEEDS(si));
    net = trainOne(toChan(normRec(Xtrain), N), Ttr, ...
                   toChan(normRec(Xval), N), Tva, N, nCls);
    for li = 1:nLvl
        s  = NOISEPCT(li)/100*FS;
        Xe = toChan(normRec(Xtest + s*Nte), N);
        accCNN_clean(li,si) = evalNet(net, Xe, Tte, clsNames);
    end
    fprintf('clean-trained seed %d done\n', SEEDS(si));
end
fprintf('\n');

%% ---- CNN, noise-trained ----------------------------------------------
for li = 1:nLvl
    s = NOISEPCT(li)/100*FS;
    Xt = toChan(normRec(Xtrain + s*Ntr), N);
    Xv = toChan(normRec(Xval   + s*Nva), N);
    Xe = toChan(normRec(Xtest  + s*Nte), N);
    for si = 1:numel(SEEDS)
        rng(SEEDS(si));
        net = trainOne(Xt, Ttr, Xv, Tva, N, nCls);
        accCNN_noisy(li,si) = evalNet(net, Xe, Tte, clsNames);
    end
    fprintf('noise-trained %4.1f %% -> %6.2f %% +/- %.2f\n', NOISEPCT(li), ...
            100*mean(accCNN_noisy(li,:)), 100*std(accCNN_noisy(li,:)));
end

%% ---- summary ---------------------------------------------------------
fprintf('\n noise%%   baseline   CNN clean-tr   CNN noise-tr\n');
for li = 1:nLvl
    fprintf('  %4.1f     %6.2f      %6.2f         %6.2f\n', NOISEPCT(li), ...
            100*accBase(li), 100*mean(accCNN_clean(li,:)), ...
            100*mean(accCNN_noisy(li,:)));
end

save('noise_results.mat','NOISEPCT','FS','accBase','accCNN_clean','accCNN_noisy','SEEDS');

%% ---- figure ----------------------------------------------------------
figure('Color','w'); hold on; grid on
errorbar(NOISEPCT, 100*mean(accCNN_noisy,2), 100*std(accCNN_noisy,0,2), ...
         '-o','LineWidth',1.6,'DisplayName','CNN, trained with noise');
errorbar(NOISEPCT, 100*mean(accCNN_clean,2), 100*std(accCNN_clean,0,2), ...
         '-s','LineWidth',1.6,'DisplayName','CNN, trained clean');
plot(NOISEPCT, 100*accBase, '-^','LineWidth',1.6, ...
     'DisplayName','nearest centroid');
yline(10,'--','chance','HandleVisibility','off');
xlabel(sprintf('sensor noise \\sigma (%% of %g A full scale)', FS));
ylabel('test accuracy (%)'); ylim([0 105]); legend('Location','southwest');
title('Accuracy against sensor noise');

%% ---- what to look for ------------------------------------------------
%  1. Where does the CNN first drop below 100 %? That is the point at
%     which this benchmark starts to have resolving power again, and
%     where the A/B/C comparison should be re-run.
%  2. Does the gap to the baseline widen or narrow with noise? Widening
%     strengthens the case for the CNN; narrowing means its advantage
%     was specific to clean simulated data.
%  3. How far apart are the two CNN curves? That distance is the cost
%     of training on clean simulation and deploying on real sensors -
%     a number neither reference paper reports.
%  4. Per-class behaviour is worth checking at the level where accuracy
%     first falls: the clamp classes 7, 8 and 9 should degrade first,
%     since their signature is roughly thirty times smaller than the
%     outer-switch ones.

%% ---- helpers ---------------------------------------------------------
function Z = normRec(M)
    Z = M ./ max(abs(M),[],2);          % one divisor per record
end

function out = toChan(M, N)             % -> 1000 x 1 x 3 x nObs
    n = size(M,1);
    out = zeros(N, 1, 3, n, 'single');
    for i = 1:n
        out(:,1,:,i) = single(reshape(M(i,:), N, 3));
    end
end

function net = trainOne(Xt, Tt, Xv, Tv, N, nCls)
    layers = [
        imageInputLayer([N 1 3],'Normalization','none')
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
    opts = trainingOptions('adam', ...
        'InitialLearnRate',1e-3,'MaxEpochs',30,'MiniBatchSize',32, ...
        'Shuffle','every-epoch','ValidationData',{Xv,Tv}, ...
        'ValidationFrequency',25,'ValidationPatience',8, ...
        'OutputNetwork','best-validation','Verbose',false,'Plots','none');
    net = trainnet(Xt, Tt, layers, "crossentropy", opts);
end

function a = evalNet(net, Xe, Te, clsNames)
    p = scores2label(minibatchpredict(net, Xe), clsNames);
    a = mean(p == Te);
end

function a = centroidAcc(Xtr, Ytr, Xte, Yte, N, nCls)
    Ftr = feats(Xtr, N);  Fte = feats(Xte, N);
    sd  = std(Ftr);  Ftr = Ftr./sd;  Fte = Fte./sd;
    C0  = zeros(nCls, size(Ftr,2));
    u   = unique(Ytr);
    for k = 1:nCls, C0(k,:) = mean(Ftr(Ytr==u(k),:)); end
    D = zeros(size(Fte,1), nCls);
    for k = 1:nCls, D(:,k) = sum((Fte - C0(k,:)).^2, 2); end
    [~,p] = min(D,[],2);
    a = mean(u(p) == Yte);
end

function F = feats(M, N)
    n = size(M,1);  F = zeros(n,9);
    for i = 1:n
        W = reshape(M(i,:), N, 3);  W = W(N/2+1:end,:);
        F(i,:) = [mean(W) max(W) min(W)];
    end
end
