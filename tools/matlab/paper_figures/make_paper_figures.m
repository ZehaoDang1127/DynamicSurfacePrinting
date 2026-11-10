%MAKE_PAPER_FIGURES Regenerate every data figure in the paper.
%   Run from any folder:  run('tools/matlab/paper_figures/make_paper_figures.m')
%   Requires MATLAB R2020b or later (tiledlayout tile legends, exportgraphics).
%
%   The CSV files in data/ are committed, so the figures can be rebuilt without
%   the training outputs. To refresh them after new runs, first execute
%     python tools/matlab/paper_figures/export_figure_data.py --output-root output
%
%   Set paperFigureDir to the paper's figures/ folder to write there directly.

here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, 'data');
paperFigureDir = fullfile(here, 'figures');

plot_training_curves(dataDir, paperFigureDir);  % Fig. training_curves (Section 4.1)
% plot_mae_vs_lead(dataDir, paperFigureDir);    % hidden MAE vs lead: dropped from the paper (values are in tab:synthetic)
