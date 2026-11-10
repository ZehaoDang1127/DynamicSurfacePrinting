function style_axes(ax, S)
%STYLE_AXES Apply the common axis styling used in every paper figure.
set(ax, 'FontName', S.font, 'FontSize', S.smallSize, 'LineWidth', 0.5, ...
    'XColor', S.muted, 'YColor', S.muted, 'TickDir', 'out', 'TickLength', [0.015 0.015], ...
    'Box', 'off', 'XGrid', 'off', 'YGrid', 'on', 'XMinorTick', 'off', 'YMinorTick', 'off', ...
    'YMinorGrid', 'off', 'GridColor', S.grid, 'GridAlpha', 1, 'Layer', 'bottom');
ax.Title.FontWeight = 'normal';
ax.Title.FontSize = S.fontSize;
ax.Title.Color = S.ink;
ax.TitleHorizontalAlignment = 'left';
ax.XLabel.FontSize = S.fontSize;
ax.XLabel.Color = S.ink;
ax.YLabel.FontSize = S.fontSize;
ax.YLabel.Color = S.ink;
end
