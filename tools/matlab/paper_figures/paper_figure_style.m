function S = paper_figure_style()
%PAPER_FIGURE_STYLE Shared fonts and colors for the paper figures.
%   Sizes suit an Elsevier 5p two-column page (text width 7.05 in,
%   column width 3.45 in) with the figure included at its natural width.

S.font = 'Times New Roman';
S.fontSize = 7;        % axis labels, titles
S.smallSize = 6.5;     % tick labels, annotations, legend
S.ink = hex2rgb('#0b0b0b');
S.muted = hex2rgb('#52514e');
S.grid = hex2rgb('#e4e3df');
S.train = hex2rgb('#2a78d6');   % training curve
S.val = hex2rgb('#eb6834');     % validation curve
% One color per model, in a fixed order (validated categorical palette).
S.model.mixed_scratch = hex2rgb('#2a78d6');
S.model.mixed_pretrained = hex2rgb('#eb6834');
S.model.standard_scratch = hex2rgb('#1baf7a');
S.model.standard_pretrained = hex2rgb('#eda100');
S.textWidth = 7.05;    % inches
S.columnWidth = 3.45;  % inches
end

function rgb = hex2rgb(hex)
rgb = sscanf(hex(2:end), '%2x')' / 255;
end
