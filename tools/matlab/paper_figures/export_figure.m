function export_figure(fig, basePath)
%EXPORT_FIGURE Write a vector PDF and a 300 dpi PNG preview.
exportgraphics(fig, [basePath '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, [basePath '.png'], 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('wrote %s.pdf\n', basePath);
end
