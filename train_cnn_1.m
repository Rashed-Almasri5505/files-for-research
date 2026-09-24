%% train_cnn.m
%  DCNN fault classification for the three-phase T-type converter.
%  Replicates Table I of:
%    M. Sahani, M. D. Siddique, P. Sundararajan, S. K. Panda, NPEC 2023.
%
%  THREE CONFIGURATIONS, compared on identical data and hyperparameters:
%    A  3000 x 1, per-record peak normalised   <- the paper's input
%    B  1000 x 3, per-record peak normalised   <- phases as channels
%    C  3000 x 1, raw amps                     <- unnormalised control
%
%  Five seeds each, so differences can be separated from training noise.
%  The train/val/test split is FIXED across all runs - only network
%  initialisation and batch shuffling vary with the seed. That isolates
%  training variance rather than mixing in split variance.
%
%  -------------------------------------------------------------------
%  NOTE ON TABLE I OF THE PAPER
%
%  1. Pooling arithmetic only works with 'same' padding.
%     2950 -> 1475 with pool 10 stride 2 requires ceil(2950/2), i.e.
%     padded pooling. Plain pooling would give 1471. Same for the
%     other two pooling layers. 'same' padding is used here.
%
%  2. The order is Conv -> ReLU -> BatchNorm.
%     The usual order is Conv -> BatchNorm -> ReLU. Table I lists
%     ReLU first. Followed as printed; set SWAP_BN = true to test
%     the conventional order.
%
%  3. THREE FULLY CONNECTED LAYERS WITH NO ACTIVATION BETWEEN THEM.
%     Table I gives FC-500 -> FC-100 -> FC-10 with no ReLU listed
%     between them. Three stacked linear maps with no nonlinearity
%     collapse algebraically into a single linear map: the 500 and
%     100 unit layers add parameters but no representational power
%     over a single FC-10. Either the paper omitted the activations
%     from the table, or the network is carrying dead capacity.
%     Followed as printed by default. Set FC_RELU = true to insert
%     the activations and compare - that difference is worth a
%     paragraph in your paper either way.
%  -------------------------------------------------------------------

clear; clc;

load('ttype_dataset_v3.mat');          % X, Y, Xtrain/val/test, N, classes

FC_RELU = false;    % true -> insert ReLU between the fully connected layers
SWAP_BN = false;    % true -> conventional Conv -> BatchNorm -> ReLU order
SEEDS   = 1:5;
CONFIGS = ["A" "B" "C"];

nCls = numel(classes);

%% ---- prepare the three input tensors ---------------------------------
%  X rows are [Ia(1:1000) Ib(1:1000) Ic(1:1000)], so reshaping a row to
%  1000x3 puts each phase in its own column.

peak = @(M) max(abs(M),[],2);                  % one divisor per record

Xn_tr = Xtrain ./ peak(Xtrain);
Xn_va = Xval   ./ peak(Xval);
Xn_te = Xtest  ./ peak(Xtest);

% A and C: 3000 x 1 x 1 x nObs
to3000 = @(M) reshape(single(M)', [size(M,2) 1 1 size(M,1)]);

% B: 1000 x 1 x 3 x nObs
    function out = to1000x3(M, N)
        n = size(M,1);
        out = zeros(N, 1, 3, n, 'single');
        for i = 1:n
            W = reshape(M(i,:), N, 3);          % columns = phases
            out(:,1,:,i) = single(W);
        end
    end

data.A.tr = to3000(Xn_tr);  data.A.va = to3000(Xn_va);  data.A.te = to3000(Xn_te);
data.B.tr = to1000x3(Xn_tr,N); data.B.va = to1000x3(Xn_va,N); data.B.te = to1000x3(Xn_te,N);
data.C.tr = to3000(Xtrain); data.C.va = to3000(Xval);   data.C.te = to3000(Xtest);

Ttr = categorical(Ytrain);
Tva = categorical(Yval);
Tte = categorical(Ytest);
clsNames = categories(Ttr);

fprintf('A/C input %s   B input %s\n', mat2str(size(data.A.tr)), ...
                                       mat2str(size(data.B.tr)));

%% ---- network builder -------------------------------------------------
    function lg = buildNet(inSize, nCls, fcRelu, swapBn)
        blk = @(k,nf) localBlock(k, nf, swapBn);
        lg = [
            imageInputLayer(inSize, 'Normalization','none', 'Name','in')
            blk(51,15)
            maxPooling2dLayer([10 1],'Stride',[2 1],'Padding','same')
            blk(26,10)
            maxPooling2dLayer([10 1],'Stride',[2 1],'Padding','same')
            blk(16,5)
            maxPooling2dLayer([10 1],'Stride',[2 1],'Padding','same')
            ];
        if fcRelu
            lg = [lg
                fullyConnectedLayer(500); reluLayer
                fullyConnectedLayer(100); reluLayer
                fullyConnectedLayer(nCls); softmaxLayer];
        else
            lg = [lg
                fullyConnectedLayer(500)
                fullyConnectedLayer(100)
                fullyConnectedLayer(nCls)
                softmaxLayer];
        end
    end

    function b = localBlock(k, nf, swapBn)
        c = convolution2dLayer([k 1], nf);
        if swapBn
            b = [c; batchNormalizationLayer; reluLayer];     % conventional
        else
            b = [c; reluLayer; batchNormalizationLayer];     % as Table I
        end
    end

%% ---- training options ------------------------------------------------
    function o = opts(vaX, vaT)
        o = trainingOptions('adam', ...
            'InitialLearnRate', 1e-3, ...
            'MaxEpochs', 30, ...
            'MiniBatchSize', 32, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', {vaX, vaT}, ...
            'ValidationFrequency', 25, ...
            'ValidationPatience', 8, ...
            'OutputNetwork', 'best-validation', ...
            'Verbose', false, ...
            'Plots', 'none');
    end

%% ---- run -------------------------------------------------------------
acc  = zeros(numel(CONFIGS), numel(SEEDS));
CM   = zeros(nCls, nCls, numel(CONFIGS));    % summed over seeds
tAll = tic;

for ci = 1:numel(CONFIGS)
    cfg = CONFIGS(ci);
    d   = data.(cfg);
    inSize = size(d.tr, 1:3);

    for si = 1:numel(SEEDS)
        rng(SEEDS(si));
        layers = buildNet(inSize, nCls, FC_RELU, SWAP_BN);

        t0  = tic;
        net = trainnet(d.tr, Ttr, layers, "crossentropy", opts(d.va, Tva));

        scores = minibatchpredict(net, d.te);
        Ypred  = scores2label(scores, clsNames);

        correct = Ypred == Tte;
        acc(ci,si) = mean(correct);

        % confusion matrix without the Statistics toolbox
        ti = double(Tte);  pi_ = double(Ypred);
        CM(:,:,ci) = CM(:,:,ci) + accumarray([ti pi_], 1, [nCls nCls]);

        fprintf('%s  seed %d   test %.2f %%   (%.1f min)\n', ...
                cfg, SEEDS(si), 100*acc(ci,si), toc(t0)/60);

        if si == 1
            nets.(cfg) = net;        % keep one trained net per config
        end
    end
    fprintf('--- %s: %.2f %% +/- %.2f %%\n\n', cfg, ...
            100*mean(acc(ci,:)), 100*std(acc(ci,:)));
end
fprintf('Total %.1f minutes.\n', toc(tAll)/60);

%% ---- summary ---------------------------------------------------------
fprintf('\n  cfg     mean      std       min      max\n');
for ci = 1:numel(CONFIGS)
    a = 100*acc(ci,:);
    fprintf('   %s    %6.2f   %6.2f   %6.2f   %6.2f\n', ...
            CONFIGS(ci), mean(a), std(a), min(a), max(a));
end

save('cnn_results.mat','acc','CM','CONFIGS','SEEDS','FC_RELU','SWAP_BN');

%% ---- per-class accuracy, summed over seeds ---------------------------
%  Expect errors concentrated in classes 0, 7, 8, 9 - the healthy and
%  neutral-clamp classes that the DC-offset features could not separate.
fprintf('\nper-class recall (%%), summed over %d seeds\n', numel(SEEDS));
fprintf('class ');  fprintf('%7s', CONFIGS);  fprintf('\n');
for c = 1:nCls
    fprintf('%5s ', clsNames{c});
    for ci = 1:numel(CONFIGS)
        r = CM(c,:,ci);
        fprintf('%7.1f', 100*r(c)/sum(r));
    end
    fprintf('\n');
end

%% ---- confusion matrix for one configuration --------------------------
%  Change the index to 2 or 3 for configs B or C.
ci = 1;
figure('Color','w');
imagesc(CM(:,:,ci)); axis square; colorbar
set(gca,'XTick',1:nCls,'XTickLabel',clsNames, ...
        'YTick',1:nCls,'YTickLabel',clsNames);
xlabel('predicted'); ylabel('true');
title(sprintf('config %s, %d seeds combined', CONFIGS(ci), numel(SEEDS)));
for a = 1:nCls
    for b = 1:nCls
        if CM(a,b,ci) > 0
            text(b, a, num2str(CM(a,b,ci)), 'HorizontalAlignment','center', ...
                 'Color', 'w', 'FontSize', 8);
        end
    end
end

%% ---- what to do with the numbers -------------------------------------
%  1. If A and B overlap within one standard deviation, the input shape
%     does not matter and you should say so rather than claiming a win.
%  2. If C is close to A, normalisation was unnecessary here - which
%     would itself be worth reporting, since it means the network
%     learned to ignore load level on its own.
%  3. If any configuration hits exactly 100 % on every seed, be
%     suspicious before being pleased. Check that Ytest rows really
%     correspond to Xtest rows, and that no record appears in both the
%     training and test sets.
