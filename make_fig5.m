%% make_fig5.m — Fig. 5 of the JURI manuscript
%
%  Runs the converter with a fault, slides the analysis window along the result
%  and asks the trained network which switch failed at every window position.
%
%  It writes TWO things:
%     fig5_sliding.png   the figure, 600 dpi, sized for the manuscript slot
%     fig5_data.mat      the plotted data, so the figure can be redrawn in the
%                        manuscript's house style without re-simulating
%
%  *** FIXED IN THIS VERSION: the dark theme. ***
%  MATLAB R2025a and later export graphics in the desktop's colour theme, so on
%  a dark desktop exportgraphics writes a figure with a BLACK axes background —
%  which prints as a solid black block. Every colour below is now set
%  explicitly, and the figure theme is forced to light.
%
%  REQUIREMENTS — four files in this folder, MATLAB cd'd into it:
%      make_fig5.m
%      new_t_type_three_level.slx
%      diagnosis_net.mat            (or ttype_dataset_v3.mat, to retrain)
%      ttype_dataset_v3.mat
%
%  Tested against R2025b. No Statistics and Machine Learning Toolbox needed.
%
%  WHAT TO EDIT: faultLoc and tFault, below. Edit them HERE, in the file —
%  the `clear` on the next line wipes anything set at the command line.

clear; clc;

%% ---- the case to plot -------------------------------------------------
faultLoc  = 8;          % 0 = healthy, 1..9 = OCF in S1..S9
tFault    = 0.105;      % s.  The manuscript's Fig. 5 is S8 at 0.105 s.

%% ---- figure geometry (matches the manuscript slot — do not change) ----
FIG_W_IN  = 4.00;       % inches
FIG_H_IN  = 2.42;
DPI       = 600;
OUTFILE   = 'fig5_sliding.png';
DATAFILE  = 'fig5_data.mat';

%% ---- operating point and model settings -------------------------------
MDL       = 'new_t_type_three_level';
NETFILE   = 'diagnosis_net.mat';
NOISE_AUG = 2;          % % of full scale, added during training
FS        = 300;        % A, sensor full scale
STEP      = 5;          % sliding-window step, samples (0.2 ms at 25 kHz)

mi        = 0.88;
Pload     = 95e3;
Qload     = Pload*0.25;
Vdc       = 1010;
portOrder = [7 1 2 3 8 4 5 6 9];
f0        = 50;
Ts        = 1/25000;
N         = 1000;       % 2 cycles at 50 Hz, 25 kHz

%% ---- network: load, or train and save ---------------------------------
if isfile(NETFILE)
    load(NETFILE,'net','clsNames');
    fprintf('Loaded %s\n', NETFILE);
else
    fprintf('No saved network. Training one (config A, ~2 min)...\n');
    D = load('ttype_dataset_v3.mat');
    rng(1);
    sg = NOISE_AUG/100*FS;
    Xt = flat(normRec(D.Xtrain + sg*randn(size(D.Xtrain))));
    Xv = flat(normRec(D.Xval   + sg*randn(size(D.Xval))));
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
fprintf('Simulated: faultLoc = %d at t = %.4f s\n', faultLoc, tFault);

%% ---- sliding window ---------------------------------------------------
starts = 1 : STEP : (size(I,1) - N + 1);
nW = numel(starts);
Xs = zeros(3*N, 1, 1, nW, 'single');
for w = 1:nW
    Ww = I(starts(w):starts(w)+N-1, :);
    Xs(:,1,1,w) = single(normRec([Ww(:,1); Ww(:,2); Ww(:,3)]'))';
end
S = minibatchpredict(net, Xs);
[confS, kS] = max(S, [], 2);
predS = double(string(clsNames(kS)));
tEnd  = t(starts + N - 1);            % time at the window's trailing edge

preFault = tEnd <  tFault;
fullPost = t(starts) > tFault;
nPre  = sum(preFault);   pctPre  = 100*mean(predS(preFault)==0);
nPost = sum(fullPost);   pctPost = 100*mean(predS(fullPost)==faultLoc);
fprintf('  pre-fault windows  : %3d, called healthy  %5.1f %%\n', nPre,  pctPre);
fprintf('  post-fault windows : %3d, called correctly %5.1f %%\n', nPost, pctPost);

hit = find(tEnd > tFault & predS == faultLoc, 1);
firstMs = NaN;
if ~isempty(hit)
    firstMs = 1e3*(tEnd(hit)-tFault);
    fprintf('  first correct at %5.2f ms of post-fault current\n', firstMs);
end

%% ---- save the plotted data so the figure can be redrawn ---------------
save(DATAFILE, 't', 'I', 'tEnd', 'predS', 'confS', 'tFault', 'faultLoc', ...
     'mi', 'Pload', 'Qload', 'Vdc', 'STEP', 'N', 'Ts', ...
     'nPre', 'pctPre', 'nPost', 'pctPost', 'firstMs');
fprintf('Wrote %s\n', DATAFILE);

%% ---- figure -----------------------------------------------------------
INK = [0 0 0]; MID = [0.35 0.35 0.35]; LIGHT = [0.65 0.65 0.65];
GRID = [0.85 0.85 0.85]; WHITE = [1 1 1];
FS_LAB = 8; FS_TICK = 7.5;

fig = figure('Color', WHITE, 'Units','inches', ...
             'Position',[1 1 FIG_W_IN FIG_H_IN], 'PaperPositionMode','auto', ...
             'InvertHardcopy','off');
% R2025a+ exports in the desktop theme; force light so the axes are not black
try, fig.Theme = 'light'; catch, end

tl = tiledlayout(fig,3,1,'TileSpacing','compact','Padding','compact');

% -- panel (a): the three phase currents
ax1 = nexttile(tl);
plot(ax1, t*1e3, I(:,1), '-',  'Color',INK,   'LineWidth',0.7); hold(ax1,'on')
plot(ax1, t*1e3, I(:,2), '--', 'Color',MID,   'LineWidth',0.7);
plot(ax1, t*1e3, I(:,3), '-.', 'Color',LIGHT, 'LineWidth',0.7);
xline(ax1, tFault*1e3, ':', 'Color',INK, 'LineWidth',0.9);
ylabel(ax1,'Current (A)');
lg = legend(ax1, {'$i_\mathrm{a}$','$i_\mathrm{b}$','$i_\mathrm{c}$'}, ...
       'Interpreter','latex','Orientation','horizontal', ...
       'Location','northoutside','Box','off','FontSize',FS_TICK);
lg.TextColor = INK;
text(ax1, 0.015, 0.94, '(a)', 'Units','normalized', 'Color',INK, ...
     'FontSize',FS_TICK, 'VerticalAlignment','top');

% -- panel (b): predicted class
ax2 = nexttile(tl);
yline(ax2, faultLoc, '-', 'Color',LIGHT, 'LineWidth',1.4); hold(ax2,'on')
plot(ax2, tEnd*1e3, predS, '.', 'Color',INK, 'MarkerSize',3.5);
xline(ax2, tFault*1e3, ':', 'Color',INK, 'LineWidth',0.9);
ylabel(ax2,'Predicted class'); ylim(ax2,[-0.6 9.6]); yticks(ax2,0:3:9);
text(ax2, 0.015, 0.94, '(b)', 'Units','normalized', 'Color',INK, ...
     'FontSize',FS_TICK, 'VerticalAlignment','top');
text(ax2, 0.985, 0.94, sprintf('true class %d', faultLoc), 'Color',INK, ...
     'Units','normalized','HorizontalAlignment','right', ...
     'FontSize',FS_TICK,'VerticalAlignment','top');

% -- panel (c): softmax confidence
ax3 = nexttile(tl);
plot(ax3, tEnd*1e3, 100*confS, '.', 'Color',INK, 'MarkerSize',3.5); hold(ax3,'on')
xline(ax3, tFault*1e3, ':', 'Color',INK, 'LineWidth',0.9);
ylabel(ax3,'Confidence (%)'); ylim(ax3,[0 108]); yticks(ax3,[0 50 100]);
xlabel(ax3,'Trailing edge of analysis window (ms)');
text(ax3, 0.015, 0.94, '(c)', 'Units','normalized', 'Color',INK, ...
     'FontSize',FS_TICK, 'VerticalAlignment','top');

for ax = [ax1 ax2 ax3]
    grid(ax,'on'); box(ax,'on');
    set(ax, 'Color', WHITE, 'XColor', INK, 'YColor', INK, ...
            'FontName','Times New Roman','FontSize',FS_TICK, ...
            'GridColor', GRID, 'GridAlpha', 1, 'LineWidth', 0.6, ...
            'Layer','top','XLim',[0 200],'TickDir','out','TickLength',[0.006 0.006]);
    ax.YLabel.Color = INK;  ax.YLabel.FontSize = FS_LAB;
    ax.XLabel.Color = INK;  ax.XLabel.FontSize = FS_LAB;
    ax.Title.Color  = INK;
end
set([ax1 ax2],'XTickLabel',[]);

%% ---- export -----------------------------------------------------------
try
    exportgraphics(fig, OUTFILE, 'Resolution', DPI, ...
                   'BackgroundColor','white', 'Theme','light');
catch
    exportgraphics(fig, OUTFILE, 'Resolution', DPI, 'BackgroundColor','white');
end
fprintf('\nWrote %s  (%.2f x %.2f in at %d dpi)\n', OUTFILE, FIG_W_IN, FIG_H_IN, DPI);
fprintf('Send me %s and I will redraw Fig. 5 in the manuscript house style.\n', DATAFILE);

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
