function plot_mae_vs_lead(dataDir, outDir)
%PLOT_MAE_VS_LEAD Hidden-cell MAE against forecast lead on the held-out motions.
%   PLOT_MAE_VS_LEAD(dataDir, outDir) reads data/hidden_mae_vs_lead.csv
%   (written by export_figure_data.py from the evaluation metrics.json files)
%   and writes occlusion_mae_vs_lead.pdf to outDir: (a) standard masks,
%   20% of the grid and moving; (b) large stationary masks, 70-90% of the
%   grid, log scale. All 1,740 test windows, all hidden cells with valid targets.

here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(dataDir), dataDir = fullfile(here, 'data'); end
if nargin < 2 || isempty(outDir), outDir = fullfile(here, 'figures'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
S = paper_figure_style();
T = readtable(fullfile(dataDir, 'hidden_mae_vs_lead.csv'), 'TextType', 'string');

% model id, legend label, line style, marker, filled marker
models = {
    'mixed_scratch',       'Mixed masks, 100 epochs (primary)',            '-',  'o', true
    'mixed_pretrained',    'Mixed masks, 100 epochs, cloth-pretrained',    '-',  'o', false
    'standard_scratch',    'Standard masks, 100 epochs',                   '--', 's', true
    'standard_pretrained', 'Standard masks, 100 epochs, cloth-pretrained', '--', 's', false};
endash = char(8211);
% mask id, title, y scale, y limits, y ticks, y label
masks = {
    'standard', '(a) Standard masks: 20% of grid, moving', 'linear', [0 4], 0:0.5:4, ...
        'Hidden-cell MAE (mm)'
    'large', ['(b) Large masks: 70' endash '90% of grid, stationary'], 'log', [2 50], [2 5 10 20 50], ...
        'Hidden-cell MAE (mm, log scale)'};

fig = figure('Units', 'inches', 'Position', [1 1 S.textWidth 3.0], 'Color', 'w');
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
handles = gobjects(1, size(models, 1));
for p = 1:size(masks, 1)
    ax = nexttile(tl);
    hold(ax, 'on');
    for m = 1:size(models, 1)
        R = sortrows(T(T.model == models{m, 1} & T.masks == masks{p, 1}, :), 'lead_step');
        color = S.model.(models{m, 1});
        if models{m, 5}, face = color; else, face = [1 1 1]; end
        handles(m) = plot(ax, R.lead_ms, R.hidden_mae_mm, 'LineStyle', models{m, 3}, ...
            'Marker', models{m, 4}, 'Color', color, 'MarkerFaceColor', face, ...
            'MarkerSize', 4, 'LineWidth', 1.2);
    end
    hold(ax, 'off');
    set(ax, 'YScale', masks{p, 3}, 'YLim', masks{p, 4}, 'YTick', masks{p, 5}, ...
        'YTickLabel', arrayfun(@(v) sprintf('%g', v), masks{p, 5}, 'UniformOutput', false), ...
        'XLim', [-10 210], 'XTick', (0:6) * 200 / 6, ...
        'XTickLabel', {'0', '33', '67', '100', '133', '167', '200'});
    title(ax, masks{p, 2});
    xlabel(ax, 'Forecast lead (ms)');
    ylabel(ax, masks{p, 6});
    style_axes(ax, S);
end
lgd = legend(ax, handles, models(:, 2), 'NumColumns', 2, 'Box', 'off', ...
    'FontName', S.font, 'FontSize', S.smallSize);
lgd.Layout.Tile = 'south';
export_figure(fig, fullfile(outDir, 'occlusion_mae_vs_lead'));
end
