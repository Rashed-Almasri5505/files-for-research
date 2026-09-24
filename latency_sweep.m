%% latency_sweep.m — detection latency across all nine switches
%
%  THE NEW EXPERIMENT. Neither reproduced paper reports anything like this.
%
%  diagnose_live.m measures detection latency for one switch at one fault
%  instant. Physics says the answer should depend strongly on both: a device
%  failing while it is conducting produces an immediate anomaly, whereas one
%  failing just after its conduction interval ends produces nothing
%  observable until the next interval, up to most of a cycle later. Clamp
%  switches should differ systematically from outer switches, because their
%  conduction intervals sit around the current zero crossings.
%
%  This script sweeps faultLoc = 1..9 against the fault instant across one
%  full fundamental cycle, at a FIXED operating point so that the only things
%  varying are the two quantities under study. It writes:
%
%      latency_results.mat   the raw matrices, plus a checkpoint after each switch
%      fig6_latency.png      the distribution figure, 600 dpi, manuscript size
%
%  REQUIREMENTS — these files in this folder, MATLAB cd'd into it:
%      latency_sweep.m
%      new_t_type_three_level.slx
%      diagnosis_net.mat            (or ttype_dataset_v3.mat, to retrain)
%      ttype_dataset_v3.mat
%
%  RUNTIME: 9 x 12 = 108 simulations. Expect roughly 10-15 minutes.
%  The operating point is held fixed, so powergui does not rebuild the
%  state-space model between runs; if you want to try to speed it up further,
%  set USE_FAST_RESTART = true below and CHECK the first few results against a
%  run with it false before trusting the whole sweep.
%
%  NOTE ON COLOUR: MATLAB R2025a and later export graphics in the desktop's
%  colour theme, so on a dark desktop exportgraphics writes a BLACK axes
%  background that prints as a solid block. Every colour below is set
%  explicitly and the figure theme is forced to light.
%
%  Tested against R2025b. No Statistics and Machine Learning Toolbox needed.

clear; clc;

%% ---- sweep definition -------------------------------------------------
SWITCHES  = 1:9;                     % every gate signal
N_INSTANT = 12;                      % fault instants across one cycle
f0        = 50;
T0        = 1/f0;                    % 20 ms
T_BASE    = 0.100;                   % first fault instant (s)
tFaultAll = T_BASE + (0:N_INSTANT-1)/N_INSTANT * T0;

USE_FAST_RESTART = false;

%% ---- fixed operating point -------------------------------------------
%  Held constant so latency is attributable to switch identity and fault
%  instant alone. These are the mid-range values of the dataset's sampled
%  ranges, and the same point diagnose_live.m uses.
mi        = 0.88;
Pload     = 95e3;
Qload     = Pload*0.25;
Vdc       = 1010;
portOrder = [7 1 2 3 8 4 5 6 9];
Ts        = 1/25000;
N         = 1000;                    % 40 ms window, 2 cycles at 25 kHz

MDL       = 'new_t_type_three_level';
NETFILE   = 'diagnosis_net.mat';
NOISE_AUG = 2;
FS        = 300;
STEP      = 5;                       % window step, samples (0.2 ms)

%% ---- network: load, or train and save ---------------------------------
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

%% ---- the sweep --------------------------------------------------------
load_system(MDL);
set_param(MDL,'ReturnWorkspaceOutputs','on');
set_param(MDL,'StopTime','0.2');
if USE_FAST_RESTART, set_param(MDL,'FastRestart','on'); end

nS = numel(SWITCHES);
firstMs  = nan(nS, N_INSTANT);   % post-fault current needed for first correct call
settleMs = nan(nS, N_INSTANT);   % ... and for the call to stay correct to the end
preOK    = nan(nS, N_INSTANT);   % % of pre-fault windows called healthy
postOK   = nan(nS, N_INSTANT);   % % of fully post-fault windows called correctly

tSweep = tic;
for a = 1:nS
    faultLoc = SWITCHES(a);
    for b = 1:N_INSTANT
        tFault = tFaultAll(b);

        out = sim(MDL);
        I   = out.Iabc;
        t   = (0:size(I,1)-1)'*Ts;

        starts = 1 : STEP : (size(I,1) - N + 1);
        nW = numel(starts);
        Xs = zeros(3*N, 1, 1, nW, 'single');
        for w = 1:nW
            Ww = I(starts(w):starts(w)+N-1, :);
            Xs(:,1,1,w) = single(normRec([Ww(:,1); Ww(:,2); Ww(:,3)]'))';
        end
        S = minibatchpredict(net, Xs);
        [~, kS] = max(S, [], 2);
        predS = double(string(clsNames(kS)));
        tEnd  = t(starts + N - 1);

        pre  = tEnd < tFault;
        post = t(starts) > tFault;
        preOK(a,b)  = 100*mean(predS(pre)  == 0);
        if any(post), postOK(a,b) = 100*mean(predS(post) == faultLoc); end

        % latency is measured on the window's TRAILING edge: the amount of
        % post-fault current the window had to contain before the call was right
        hit = find(tEnd > tFault & predS == faultLoc, 1);
        if ~isempty(hit), firstMs(a,b) = 1e3*(tEnd(hit) - tFault); end

        okRun = find(tEnd > tFault);
        stay  = okRun(find(arrayfun(@(q) all(predS(q:end) == faultLoc), okRun), 1));
        if ~isempty(stay), settleMs(a,b) = 1e3*(tEnd(stay) - tFault); end

        fprintf('S%d  t_f = %.4f s   first %6.2f ms   settled %6.2f ms   post %5.1f %%\n', ...
                faultLoc, tFault, firstMs(a,b), settleMs(a,b), postOK(a,b));
    end
    % checkpoint after every switch, so an interruption costs one switch
    save('latency_results.mat','SWITCHES','tFaultAll','firstMs','settleMs', ...
         'preOK','postOK','mi','Pload','Qload','Vdc','N_INSTANT','f0');
    fprintf('--- S%d done, checkpoint saved (%.1f min elapsed) ---\n', ...
            faultLoc, toc(tSweep)/60);
end
if USE_FAST_RESTART, set_param(MDL,'FastRestart','off'); end
fprintf('\nSweep finished in %.1f min\n', toc(tSweep)/60);

%% ---- summary ----------------------------------------------------------
OUTER = 1:6;  CLAMP = 7:9;
fprintf('\n%-6s %8s %8s %8s %8s %8s\n', ...
        'switch','median','min','max','n miss','post %');
for a = 1:nS
    v = firstMs(a,:);
    fprintf('S%-5d %8.2f %8.2f %8.2f %8d %8.1f\n', SWITCHES(a), ...
            median(v,'omitnan'), min(v), max(v), sum(isnan(v)), ...
            mean(postOK(a,:),'omitnan'));
end
fprintf('\nouter switches S1-S6 : median %.2f ms, range %.2f - %.2f ms\n', ...
        median(firstMs(OUTER,:),'all','omitnan'), ...
        min(firstMs(OUTER,:),[],'all'), max(firstMs(OUTER,:),[],'all'));
fprintf('clamp switches S7-S9 : median %.2f ms, range %.2f - %.2f ms\n', ...
        median(firstMs(CLAMP,:),'all','omitnan'), ...
        min(firstMs(CLAMP,:),[],'all'), max(firstMs(CLAMP,:),[],'all'));
fprintf('all switches         : median %.2f ms, range %.2f - %.2f ms\n', ...
        median(firstMs,'all','omitnan'), ...
        min(firstMs,[],'all'), max(firstMs,[],'all'));
fprintf('cases never identified: %d of %d\n', sum(isnan(firstMs(:))), numel(firstMs));

%% ---- figure -----------------------------------------------------------
FIG_W_IN = 4.05; FIG_H_IN = 1.76; DPI = 600;
INK = [0 0 0]; MID = [0.38 0.38 0.38]; LIGHT = [0.72 0.72 0.72];
GRID = [0.85 0.85 0.85]; WHITE = [1 1 1];
FS_LAB = 8; FS_TICK = 7.5;

fig = figure('Color',WHITE,'Units','inches', ...
             'Position',[1 1 FIG_W_IN FIG_H_IN],'PaperPositionMode','auto', ...
             'InvertHardcopy','off');
try, fig.Theme = 'light'; catch, end
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% -- panel (a): distribution per switch
ax1 = nexttile(tl); hold(ax1,'on')
off = linspace(-0.26, 0.26, N_INSTANT);
for a = 1:nS
    v = firstMs(a,:);
    isClamp = SWITCHES(a) >= 7;
    if isClamp
        plot(ax1, SWITCHES(a)+off, v, 's', 'MarkerSize',2.6, ...
             'MarkerFaceColor',MID,'MarkerEdgeColor',MID);
    else
        plot(ax1, SWITCHES(a)+off, v, 'o', 'MarkerSize',2.6, ...
             'MarkerFaceColor','none','MarkerEdgeColor',INK,'LineWidth',0.45);
    end
    plot(ax1, SWITCHES(a)+[-0.34 0.34], median(v,'omitnan')*[1 1], '-', ...
         'Color',INK,'LineWidth',1.3);
end
xline(ax1, 6.5, ':', 'Color',INK, 'LineWidth',0.9);
xlabel(ax1,'Faulted switch'); ylabel(ax1,'Post-fault current needed (ms)');
xlim(ax1,[0.4 9.6]); xticks(ax1,1:9);
xticklabels(ax1, compose('S%d',1:9));
text(ax1, 0.02, 0.97, '(a)','Units','normalized','FontSize',FS_TICK, ...
     'Color',INK,'VerticalAlignment','top');
% normalised units, so these stay put when ylim is set below
text(ax1, 0.34, 0.97, 'outer','Units','normalized','Color',INK, ...
     'HorizontalAlignment','center','VerticalAlignment','top','FontSize',FS_TICK-0.5);
text(ax1, 0.83, 0.97, 'clamp','Units','normalized','Color',INK, ...
     'HorizontalAlignment','center','VerticalAlignment','top','FontSize',FS_TICK-0.5);

% -- panel (b): dependence on the fault instant within the cycle
ax2 = nexttile(tl); hold(ax2,'on')
phase = (tFaultAll - T_BASE)/T0 * 360;
for a = 1:nS
    if SWITCHES(a) >= 7
        plot(ax2, phase, firstMs(a,:), 's', 'MarkerSize',2.6, ...
             'MarkerFaceColor',MID,'MarkerEdgeColor',MID);
    else
        plot(ax2, phase, firstMs(a,:), 'o', 'MarkerSize',2.6, ...
             'MarkerFaceColor','none','MarkerEdgeColor',INK,'LineWidth',0.45);
    end
end
xlabel(ax2,'Fault instant in the cycle (deg)');
xlim(ax2,[-15 375]); xticks(ax2,0:90:360);
text(ax2, 0.02, 0.97, '(b)','Units','normalized','FontSize',FS_TICK, ...
     'Color',INK,'VerticalAlignment','top');
lg = legend(ax2, {'outer, S_1-S_6','clamp, S_7-S_9'}, 'Location','northoutside', ...
       'Orientation','horizontal','Box','off','FontSize',FS_TICK);
lg.TextColor = INK;

topMs = max(firstMs(:));                  % max ignores NaN
if isempty(topMs) || ~isfinite(topMs), topMs = 20; end
yl = [0 1.10*topMs];
for ax = [ax1 ax2]
    grid(ax,'on'); box(ax,'on'); ylim(ax, yl);
    set(ax,'Color',WHITE,'XColor',INK,'YColor',INK, ...
           'FontName','Times New Roman','FontSize',FS_TICK, ...
           'GridColor',GRID,'GridAlpha',1,'LineWidth',0.6, ...
           'Layer','top','TickDir','out','TickLength',[0.012 0.012]);
    ax.XLabel.FontSize = FS_LAB; ax.XLabel.Color = INK;
    ax.YLabel.FontSize = FS_LAB; ax.YLabel.Color = INK;
end
set(ax2,'YTickLabel',[]);

try
    exportgraphics(fig,'fig6_latency.png','Resolution',DPI, ...
                   'BackgroundColor','white','Theme','light');
catch
    exportgraphics(fig,'fig6_latency.png','Resolution',DPI,'BackgroundColor','white');
end
fprintf('\nWrote fig6_latency.png  (%.2f x %.2f in at %d dpi)\n', ...
        FIG_W_IN, FIG_H_IN, DPI);
fprintf('latency_results.mat holds the numbers behind Section 4.5.\n');

%% ---- helpers ----------------------------------------------------------
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
