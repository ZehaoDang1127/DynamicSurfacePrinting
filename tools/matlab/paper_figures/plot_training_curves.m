function plot_training_curves(dataDir, outDir)
%PLOT_TRAINING_CURVES Per-epoch training and validation objectives.
%   PLOT_TRAINING_CURVES(dataDir, outDir) reads data/training_history.csv
%   (written by export_figure_data.py from output/<run>/history.json) and
%   writes training_curves.pdf (full text width: four 100-epoch servo models
%   and the cloth pretraining run) and training_curves_primary.pdf (one
%   column: primary model only) to outDir.
%
%   The objective is the training loss: current MAE + future MAE
%   + 2 x hidden-region MAE, in mm. The test set is not evaluated per epoch,
%   so the second curve is the validation objective.

here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(dataDir), dataDir = fullfile(here, 'data'); end
if nargin < 2 || isempty(outDir), outDir = fullfile(here, 'figures'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
S = paper_figure_style();
T = readtable(fullfile(dataDir, 'training_history.csv'), 'TextType', 'string');

servoTicks = [8 10 15 20 30 50];
clothTicks = [30 40 50 70 90];
% run id, title, tile index, tile span
panels = {
    'standard_scratch',    '(a) Standard masks, scratch',          1, [1 1]
    'standard_pretrained', '(b) Standard masks, cloth-pretrained', 2, [1 1]
    'mixed_scratch',       '(c) Mixed masks, scratch (primary)',   4, [1 1]
    'mixed_pretrained',    '(d) Mixed masks, cloth-pretrained',    5, [1 1]
    'cloth_pretraining',   '(e) Cloth pretraining (EPFL, 5 fps)',  3, [2 1]};

fig = figure('Units', 'inches', 'Position', [1 1 S.textWidth 3.6], 'Color', 'w');
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:size(panels, 1)
    ax = nexttile(tl, panels{k, 3}, panels{k, 4});
    R = sortrows(T(T.run == panels{k, 1}, :), 'epoch');
    if panels{k, 1} == "cloth_pretraining", ticks = clothTicks; else, ticks = servoTicks; end
    [hTrain, hVal] = draw_panel(ax, R, panels{k, 2}, ticks, S);
    if any(panels{k, 3} == [3 4 5]), xlabel(ax, 'Epoch'); end
    if any(panels{k, 3} == [1 4]), ylabel(ax, 'Objective (mm, log scale)'); end
end
lgd = legend(ax, [hTrain hVal], {'Training', 'Validation'}, 'Orientation', 'horizontal', ...
    'Box', 'off', 'FontName', S.font, 'FontSize', S.smallSize);
lgd.Layout.Tile = 'north';
export_figure(fig, fullfile(outDir, 'training_curves'));

fig = figure('Units', 'inches', 'Position', [1 1 S.columnWidth 2.1], 'Color', 'w');
tl = tiledlayout(fig, 1, 1, 'Padding', 'compact');
ax = nexttile(tl);
R = sortrows(T(T.run == "mixed_scratch", :), 'epoch');
[hTrain, hVal] = draw_panel(ax, R, '', servoTicks, S);
xlabel(ax, 'Epoch');
ylabel(ax, 'Objective (mm, log scale)');
lgd = legend(ax, [hTrain hVal], {'Training', 'Validation'}, 'Orientation', 'horizontal', ...
    'Box', 'off', 'FontName', S.font, 'FontSize', S.smallSize);
lgd.Layout.Tile = 'north';
export_figure(fig, fullfile(outDir, 'training_curves_primary'));
end

function [hTrain, hVal] = draw_panel(ax, R, titleText, ticks, S)
epoch = R.epoch;
train = R.train_objective_mm;
val = R.validation_objective_mm;
[bestVal, b] = min(val);
hold(ax, 'on');
xline(ax, epoch(b), ':', 'Color', S.muted, 'LineWidth', 0.5);
hTrain = plot(ax, epoch, train, '--', 'Color', S.train, 'LineWidth', 1.0);
hVal = plot(ax, epoch, val, '-', 'Color', S.val, 'LineWidth', 1.0);
plot(ax, epoch(b), bestVal, 'o', 'MarkerSize', 4, 'MarkerFaceColor', S.val, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
text(ax, 0.97, 0.93, sprintf('best: epoch %d\n%.2f mm', epoch(b), bestVal), ...
    'Units', 'normalized', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
    'FontName', S.font, 'FontSize', S.smallSize, 'Color', S.ink, ...
    'BackgroundColor', 'w', 'Margin', 1);
hold(ax, 'off');
set(ax, 'YScale', 'log', 'YTick', ticks, ...
    'YTickLabel', arrayfun(@(v) sprintf('%g', v), ticks, 'UniformOutput', false), ...
    'XLim', [0 epoch(end) + 1], 'YLim', [0.92 * ticks(1), 1.1 * max([train; val])]);
title(ax, titleText);
style_axes(ax, S);
end
