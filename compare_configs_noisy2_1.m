%% compare_configs_noisy2.m
%  Ten-seed run to settle A vs B at 30 % noise.
%
%  The five-seed run gave A - B = +2.34 +/- 2.06 points, t = +2.54.
%  With df = 4 the two-tailed 5 % critical value is 2.776, so that
%  was suggestive but not conclusive (p ~ 0.064). Ten seeds raises
%  df to 9 and lowers the critical value to 2.262.
%
%  CHANGED FROM compare_configs_noisy.m
%    NOISEPCT  = 30      single level, since 20 % could not separate
%                        anything (all |t| < 1.7)
%    SEEDS     = 1:10    was 1:5
%    tcrit     = 2.262   was 2.776 - MUST change with the seed count,
%                        or real differences get reported as ties
%    save file = config_comparison_10seed.mat, so the five-seed
%                results are not overwritten
%
%  CONFIGURATIONS
%    A  3000 x 1, per-record peak normalised   the paper's input
%    B  1000 x 3, per-record peak normalised   phases as channels
%    C  3000 x 1, raw amps                     unnormalised control
%
%  All three see the SAME noise realisation, from a fixed seed.
%
%  NOTE ON INTERPRETING HIGH NOISE
%    Real current transducers sit at 0.5-2 % of full scale, where every
%    configuration reaches 100 %. Thirty percent is an artificial
%    stress level chosen so the benchmark has resolving power. It is
%    far beyond any real sensor and must not be described as a
%    deployment condition.

clear; clc;

load('ttype_dataset_v3.mat');        % X, Y, Xtrain/val/test, N, classes

FS        = 300;                     % A, assumed sensor full-scale rating
NOISEPCT  = 30;
SEEDS     = 1:10;
NOISESEED = 7;                       % same draw as the sweep
CONFIGS   = ["A" "B" "C"];

nCls = numel(classes);
nLvl = numel(NOISEPCT);
nCfg = numel(CONFIGS);
nSd  = numel(SEEDS);

% two-tailed 5 % critical value of t for df = nSd - 1
tTable = containers.Map([2 3 4 5 6 7 8 9 10 12 15 20 30], ...
                        [4.303 3.182 2.776 2.571 2.447 2.365 2.306 ...
                         2.262 2.228 2.179 2.131 2.086 2.042]);
df = nSd - 1;
if isKey(tTable, df), tcrit = tTable(df); else, tcrit = 1.96; end
fprintf('%d seeds, df = %d, two-tailed 5%% critical t = %.3f\n\n', nSd, df, tcrit);

Ttr = categorical(Ytrain);
Tva = categorical(Yval);
Tte = categorical(Ytest);
clsNames = categories(Ttr);

rng(NOISESEED);
Ntr = randn(size(Xtrain), 'like', Xtrain);
Nva = randn(size(Xval),   'like', Xval);
Nte = randn(size(Xtest),  'like', Xtest);

acc  = zeros(nCfg, nSd, nLvl);
tsec = zeros(nCfg, nSd, nLvl);
CM   = zeros(nCls, nCls, nCfg, nLvl);

tAll = tic;
for li = 1:nLvl
    s = NOISEPCT(li)/100*FS;
    fprintf('===== noise %g %% (sigma = %.0f A) =====\n', NOISEPCT(li), s);

    Atr = Xtrain + s*Ntr;   Ava = Xval + s*Nva;   Ate = Xtest + s*Nte;

    d.A.tr = toFlat(normRec(Atr));   d.A.va = toFlat(normRec(Ava));   d.A.te = toFlat(normRec(Ate));
    d.B.tr = toChan(normRec(Atr),N); d.B.va = toChan(normRec(Ava),N); d.B.te = toChan(normRec(Ate),N);
    d.C.tr = toFlat(Atr);            d.C.va = toFlat(Ava);            d.C.te = toFlat(Ate);

    for ci = 1:nCfg
        cfg = CONFIGS(ci);
        inSize = size(d.(cfg).tr, 1:3);
        for si = 1:nSd
            rng(SEEDS(si));
            t0  = tic;
            net = trainOne(d.(cfg).tr, Ttr, d.(cfg).va, Tva, inSize, nCls);
            tsec(ci,si,li) = toc(t0);

            p = scores2label(minibatchpredict(net, d.(cfg).te), clsNames);
            acc(ci,si,li) = mean(p == Tte);
            CM(:,:,ci,li) = CM(:,:,ci,li) + ...
                accumarray([double(Tte) double(p)], 1, [nCls nCls]);
            fprintf('    %s seed %2d  %6.2f %%  (%.0f s)\n', cfg, SEEDS(si), ...
                    100*acc(ci,si,li), tsec(ci,si,li));
        end
        a = 100*acc(ci,:,li);
        fprintf('  %s  %6.2f +/- %4.2f %%   (%.1f s/run)\n\n', cfg, ...
                mean(a), std(a), mean(tsec(ci,:,li)));
    end
end
fprintf('Total %.1f minutes.\n', toc(tAll)/60);

%% ---- paired comparison ------------------------------------------------
%  Seeds are matched across configurations, so the differences are
%  paired. Note the multiple-comparison caveat printed below: three
%  pairwise tests mean the per-test threshold should be stricter than
%  5 % if a family-wise claim is being made.
fprintf('\npaired differences (percentage points), critical t = %.3f\n', tcrit);
pairs = [1 2; 1 3; 2 3];
for li = 1:nLvl
    for k = 1:size(pairs,1)
        i = pairs(k,1); j = pairs(k,2);
        dd = 100*(acc(i,:,li) - acc(j,:,li));
        t  = mean(dd)/(std(dd)/sqrt(nSd));
        if abs(t) > tcrit, verdict = "differ"; else, verdict = "cannot distinguish"; end
        fprintf('  %s - %s : %+6.2f +/- %4.2f   t = %+6.2f   %s\n', ...
                CONFIGS(i), CONFIGS(j), mean(dd), std(dd), t, verdict);
    end
end
fprintf(['  Bonferroni note: three pairwise tests. For a family-wise 5 %% claim\n' ...
         '  the per-test threshold is 1.67 %%, i.e. |t| > %.3f for df = %d.\n'], ...
         tinvApprox(df), df);

%% ---- training cost ----------------------------------------------------
fprintf('\nmean training time per run (s)\n');
for ci = 1:nCfg
    fprintf('  %s : %5.1f\n', CONFIGS(ci), mean(tsec(ci,:,:), 'all'));
end

%% ---- per-class recall -------------------------------------------------
li = nLvl;
fprintf('\nper-class recall at %g %% noise (%%), %d seeds combined\n', ...
        NOISEPCT(li), nSd);
fprintf('class ');  fprintf('%8s', CONFIGS);  fprintf('\n');
for c = 1:nCls
    fprintf('%5s ', clsNames{c});
    for ci = 1:nCfg
        r = CM(c,:,ci,li);
        fprintf('%8.1f', 100*r(c)/sum(r));
    end
    fprintf('\n');
end

%% ---- error structure --------------------------------------------------
%  The five-seed run found 96 % of errors confined to a 4x4 block
%  covering healthy and the three neutral-clamp classes, with a
%  28 / 15 split between false alarms and missed faults. Check whether
%  that holds with ten seeds and for all three configurations.
fprintf('\nerror structure (healthy + clamp block)\n');
grp = [1 8 9 10];                       % classes 0, 7, 8, 9
for ci = 1:nCfg
    M     = CM(:,:,ci,li);
    tot   = sum(M(:)) - sum(diag(M));
    inGrp = sum(sum(M(grp,grp))) - sum(diag(M(grp,grp)));
    fa    = sum(M(1, grp(2:end)));      % healthy -> clamp fault
    miss  = sum(M(grp(2:end), 1));      % clamp fault -> healthy
    fprintf('  %s : %4d errors, %4d in block (%.0f %%), %3d false alarms, %3d missed\n', ...
            CONFIGS(ci), tot, inGrp, 100*inGrp/tot, fa, miss);
end

save('config_comparison_10seed.mat', ...
     'acc','tsec','CM','CONFIGS','NOISEPCT','SEEDS','FS','tcrit');

%% ---- figure -----------------------------------------------------------
figure('Color','w'); hold on; grid on
a = 100*squeeze(acc(:,:,1))';           % nSd x nCfg
for ci = 1:nCfg
    plot(ci + 0.14*(rand(nSd,1)-0.5), a(:,ci), 'o', 'MarkerSize',5);
end
for ci = 1:nCfg
    m = mean(a(:,ci)); e = std(a(:,ci));
    plot([ci-0.22 ci+0.22], [m m], 'k-', 'LineWidth', 2);
    plot([ci ci], [m-e m+e], 'k-', 'LineWidth', 1);
end
set(gca,'XTick',1:nCfg,'XTickLabel',CONFIGS); xlim([0.5 nCfg+0.5]);
ylabel('test accuracy (%)');
title(sprintf('Input representation at %g %% noise, %d seeds', NOISEPCT, nSd));

%% ---- helpers ----------------------------------------------------------
function tb = tinvApprox(df)
    % two-tailed 1.67 % critical value of t (Bonferroni, 3 tests at 5 %)
    tab = containers.Map([4 9 14 19 29], [3.96 2.93 2.72 2.63 2.54]);
    if isKey(tab, df), tb = tab(df); else, tb = 2.39; end
end

function Z = normRec(M)
    Z = M ./ max(abs(M),[],2);
end

function out = toFlat(M)                % -> 3000 x 1 x 1 x nObs
    out = reshape(single(M)', [size(M,2) 1 1 size(M,1)]);
end

function out = toChan(M, N)             % -> 1000 x 1 x 3 x nObs
    n = size(M,1);
    out = zeros(N, 1, 3, n, 'single');
    for i = 1:n
        out(:,1,:,i) = single(reshape(M(i,:), N, 3));
    end
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
    opts = trainingOptions('adam', ...
        'InitialLearnRate',1e-3,'MaxEpochs',30,'MiniBatchSize',32, ...
        'Shuffle','every-epoch','ValidationData',{Xv,Tv}, ...
        'ValidationFrequency',25,'ValidationPatience',8, ...
        'OutputNetwork','best-validation','Verbose',false,'Plots','none');
    net = trainnet(Xt, Tt, layers, "crossentropy", opts);
end
