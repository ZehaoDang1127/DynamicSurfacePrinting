function visualize_ca_live(sessionDir)
% Browse a recorded constant-acceleration run in MATLAB.
% Pass a baseline-live output directory created with --record-input.
% With no argument, use output/ca_live beside this file.
% Maps use the recording's coordinate frame and millimetres.
% Scores use only cells valid in the forecast, source, and matched future map.
% Persistence means using the latest measured source height as the forecast.
% The future camera map is a noisy measurement, not independent ground truth.
if nargin < 1
    sessionDir = fullfile(fileparts(mfilename('fullpath')), 'output', 'ca_live');
end
changeThresholdMm = 5; % Separates small and large observed changes; not a motion ground truth.

if ~isfolder(sessionDir)
    error('Session folder does not exist: %s', sessionDir);
end
forecastManifest = jsondecode(fileread(fullfile(sessionDir, 'manifest.json')));
observationDir = fullfile(sessionDir, forecastManifest.observation_session);
observationManifest = jsondecode(fileread(fullfile(observationDir, 'manifest.json')));

lines = regexp(strtrim(fileread(fullfile(sessionDir, 'forecasts.jsonl'))), '\r?\n', 'split');
records = repmat(jsondecode(lines{1}), numel(lines), 1);
for i = 2:numel(lines)
    records(i) = jsondecode(lines{i});
end
if numel(records) ~= forecastManifest.forecast_count
    warning('Forecast log has %d entries; manifest reports %d.', ...
        numel(records), forecastManifest.forecast_count);
end

observations = observationManifest.frames;
sourceTimes = [records.source_timestamp_ms];
targetTimes = [records.target_timestamp_ms];
observationTimes = [observations.camera_timestamp_ms];
observationNumbers = [observations.frame_number];
[hasSource, sourceIndex] = ismember([records.source_frame_number], observationNumbers);

% Use the evaluation command's default tolerance: 20% of a frame interval.
matchToleranceMs = 200 / observationManifest.config.camera.fps;
futureIndex = zeros(size(sourceTimes));
for i = 1:numel(records)
    [gap, j] = min(abs(observationTimes - targetTimes(i)));
    if gap <= matchToleranceMs && observationTimes(j) > sourceTimes(i)
        futureIndex(i) = j;
    end
end

xy.x = observationManifest.grid.x_centers_mm(:)';
xy.y = observationManifest.grid.y_centers_mm(:);
xy.zLimits = [forecastManifest.geometry.grid.z_min_mm, ...
              forecastManifest.geometry.grid.z_max_mm];
xy.frameName = forecastManifest.geometry.frame_name;

timeZero = min(observationTimes(1), sourceTimes(1));
tForecast = (sourceTimes - timeZero) / 1000;
tObservation = (observationTimes - timeZero) / 1000;

overview = figure('Name', 'CA run overview', 'Color', 'w', ...
    'Position', [100 100 1050 720]);
layout = tiledlayout(overview, 3, 1, 'TileSpacing', 'compact');
title(layout, sprintf('Constant-acceleration recording: %d observations, %d forecasts', ...
    numel(observations), numel(records)));

ax1 = nexttile(layout);
plot(ax1, tObservation, [observations.valid_fraction], ...
    'Color', [0.25 0.55 0.8], 'DisplayName', 'Measured valid');
hold(ax1, 'on');
plot(ax1, tForecast, [records.prediction_valid_fraction], ...
    'Color', [0.9 0.35 0.1], 'DisplayName', 'Forecast fit support');
ylabel(ax1, 'Valid grid fraction');
ylim(ax1, [0 1]);
legend(ax1, 'Location', 'best');
grid(ax1, 'on');

ax2 = nexttile(layout);
plot(ax2, tForecast, [records.inference_ms], ...
    'DisplayName', 'Inference', 'Color', [0.2 0.55 0.25]);
hold(ax2, 'on');
plot(ax2, tForecast, [records.host_receive_to_prediction_ms], ...
    'DisplayName', 'Host receive to prediction', 'Color', [0.55 0.3 0.7]);
yline(ax2, forecastManifest.config.horizon_ms, '--k', ...
    'DisplayName', 'Nominal horizon');
ylabel(ax2, 'Time (ms)');
legend(ax2, 'Location', 'best');
grid(ax2, 'on');

ax3 = nexttile(layout);
lead = [records.nominal_host_lead_ms_upper_bound];
plot(ax3, tForecast, lead, 'Color', [0.2 0.4 0.8]);
hold(ax3, 'on');
yline(ax3, 0, '--k');
missed = [records.deadline_definitely_missed];
plot(ax3, tForecast(missed), lead(missed), 'r.', 'MarkerSize', 10);
ylabel(ax3, 'Lead upper bound (ms)');
xlabel(ax3, 'Elapsed camera time (s)');
grid(ax3, 'on');

cursor = [xline(ax1, tForecast(1), ':k'), ...
          xline(ax2, tForecast(1), ':k'), ...
          xline(ax3, tForecast(1), ':k')];

fprintf('Scoring %d matched forecasts against saved future measurements...\n', ...
    nnz(futureIndex > 0 & hasSource));
evaluation = scoreForecasts(sessionDir, observationDir, records, observations, ...
    hasSource, sourceIndex, futureIndex, changeThresholdMm);
plotErrorSummary(evaluation, records, observations, tForecast);
fprintf(['Scored %d frames and %d common cells (%.1f%% of grid cells in matched frames).\n' ...
    'CA: RMSE %.2f, MAE %.2f, median |error| %.2f, 95th |error| %.2f mm.\n' ...
    'Persistence: RMSE %.2f, MAE %.2f, median |error| %.2f, 95th |error| %.2f mm.\n'], ...
    evaluation.scoredFrames, evaluation.scoredCells, 100*evaluation.scoredGridFraction, ...
    evaluation.all.ca.rmse, evaluation.all.ca.mae, evaluation.all.ca.medianAE, evaluation.all.ca.p95AE, ...
    evaluation.all.persistence.rmse, evaluation.all.persistence.mae, ...
    evaluation.all.persistence.medianAE, evaluation.all.persistence.p95AE);
fprintf('Mean measured coverage %.1f%%; mean forecast fit support %.1f%%.\n', ...
    100*mean([observations.valid_fraction]), ...
    100*mean([records.prediction_valid_fraction]));
pair = evaluation.similarRmsePair;
if all(pair > 0)
    fprintf(['Example: forecasts %d and %d have similar RMSE (%.2f vs %.2f mm) ' ...
        'but different 95th |error| (%.2f vs %.2f mm).\n'], ...
        pair(1), pair(2), evaluation.caRmse(pair(1)), evaluation.caRmse(pair(2)), ...
        evaluation.caP95(pair(1)), evaluation.caP95(pair(2)));
end

viewer = figure('Name', 'CA frame browser', 'Color', 'w', ...
    'Position', [190 60 1700 900]);
maps = tiledlayout(viewer, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
maps.Position = [0.04 0.14 0.94 0.82];
mapAxes = gobjects(1, 8);
for i = 1:8
    mapAxes(i) = nexttile(maps);
end

slider = uicontrol(viewer, 'Style', 'slider', 'Units', 'normalized', ...
    'Position', [0.08 0.075 0.84 0.035], 'Min', 1, 'Max', numel(records), ...
    'Value', 1, 'SliderStep', [1 10] / max(numel(records)-1, 1));
label = uicontrol(viewer, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.08 0.015 0.84 0.045], 'BackgroundColor', 'w', ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

state = struct('sessionDir', sessionDir, 'observationDir', observationDir, ...
    'records', records, 'observations', observations, ...
    'sourceIndex', sourceIndex, 'hasSource', hasSource, ...
    'futureIndex', futureIndex, 'xy', xy, 'mapAxes', mapAxes, ...
    'slider', slider, 'label', label, 'cursor', cursor, ...
    'tForecast', tForecast, 'matchToleranceMs', matchToleranceMs, ...
    'evaluation', evaluation);
guidata(viewer, state);
slider.Callback = @(control, ~) showForecast(viewer, round(control.Value));
viewer.KeyPressFcn = @(~, event) browseWithKey(viewer, event);
initialIndex = find(futureIndex > 0 & hasSource, 1, 'first');
if isempty(initialIndex)
    initialIndex = 1;
end
showForecast(viewer, initialIndex);

fprintf(['Browse with the slider or left/right arrow keys. Gray cells are outside the ' ...
    'relevant valid mask; the support panel explains why.\n']);
fprintf('Future observation match tolerance: %.2f ms.\n', matchToleranceMs);
end

%% Local functions
function browseWithKey(fig, event)
    state = guidata(fig);
    index = round(state.slider.Value);
    switch event.Key
        case 'rightarrow'
            showForecast(fig, index + 1);
        case 'leftarrow'
            showForecast(fig, index - 1);
        case 'pageup'
            showForecast(fig, index + 10);
        case 'pagedown'
            showForecast(fig, index - 10);
    end
end

function showForecast(fig, index)
    state = guidata(fig);
    index = max(1, min(numel(state.records), index));
    state.slider.Value = index;
    record = state.records(index);
    forecast = readMaps(fullfile(state.sessionDir, record.file), ...
        {'prediction_mm', 'prediction_valid', 'velocity_mm_s', 'acceleration_mm_s2'});
    validPrediction = logical(forecast.prediction_valid) & isfinite(forecast.prediction_mm);

    observed = nan(size(forecast.prediction_mm));
    validObserved = false(size(observed));
    if state.hasSource(index)
        file = state.observations(state.sourceIndex(index)).file;
        measured = readMaps(fullfile(state.observationDir, file), {'height_mm', 'valid'});
        observed = measured.height_mm;
        validObserved = logical(measured.valid) & isfinite(observed);
    end

    future = nan(size(observed));
    validFuture = false(size(observed));
    futureTitle = 'Measured near target (no match)';
    if state.futureIndex(index) > 0
        j = state.futureIndex(index);
        file = state.observations(j).file;
        measured = readMaps(fullfile(state.observationDir, file), {'height_mm', 'valid'});
        future = measured.height_mm;
        validFuture = logical(measured.valid) & isfinite(future);
        deltaMs = state.observations(j).camera_timestamp_ms - record.target_timestamp_ms;
        futureTitle = sprintf('Measured near target (dt = %+.1f ms)', deltaMs);
    end

    errorMap = forecast.prediction_mm - future;
    validError = validPrediction & validObserved & validFuture & isfinite(errorMap);
    xy = state.xy;
    plotMap(state.mapAxes(1), xy, observed, validObserved, ...
        'Measured at source (mm)', xy.zLimits, parula(256));
    plotMap(state.mapAxes(2), xy, forecast.prediction_mm, validPrediction, ...
        'Forecast at target (mm)', xy.zLimits, parula(256));
    plotMap(state.mapAxes(3), xy, future, validFuture, ...
        futureTitle, xy.zLimits, parula(256));
    plotMap(state.mapAxes(4), xy, errorMap, validError, ...
        'Forecast - future measurement (mm)', ...
        symmetricLimits(errorMap(validError), 1), blueWhiteRed(256));
    plotMap(state.mapAxes(5), xy, forecast.velocity_mm_s, validPrediction, ...
        'Fitted velocity (mm/s)', ...
        symmetricLimits(forecast.velocity_mm_s(validPrediction), 1), blueWhiteRed(256));
    plotMap(state.mapAxes(6), xy, forecast.acceleration_mm_s2, validPrediction, ...
        'Fitted acceleration (mm/s^2)', ...
        symmetricLimits(forecast.acceleration_mm_s2(validPrediction), 1), blueWhiteRed(256));
    support = zeros(size(observed));
    support(validObserved) = 1;
    support(validPrediction) = 2;
    support(validError) = 3;
    plotSupport(state.mapAxes(7), xy, support);

    persistenceError = observed - future;
    plotErrorTail(state.mapAxes(8), errorMap(validError), persistenceError(validError));

    matchedText = 'no';
    if state.futureIndex(index) > 0
        matchedText = 'yes';
    end
    if state.evaluation.count(index) > 0
        resultText = sprintf('CA RMSE %.1f vs persistence %.1f mm | CA 95th |error| %.1f mm', ...
            state.evaluation.caRmse(index), state.evaluation.persistenceRmse(index), ...
            state.evaluation.caP95(index));
    else
        resultText = 'No common valid pixels to score';
    end
    state.label.String = {sprintf(['Forecast %d/%d | source frame %d | t = %.2f s | ' ...
        'fit support %.1f%% | matched future: %s'], ...
        index, numel(state.records), record.source_frame_number, ...
        state.tForecast(index), 100 * record.prediction_valid_fraction, matchedText), ...
        sprintf('%s | scored %.1f%% of grid', resultText, ...
        100 * state.evaluation.gridFraction(index))};
    for i = 1:numel(state.cursor)
        if isgraphics(state.cursor(i))
            state.cursor(i).Value = state.tForecast(index);
        end
    end
    drawnow limitrate;
end

function plotMap(ax, xy, values, valid, caption, limits, colors)
    cla(ax);
    img = imagesc(ax, xy.x, xy.y, values);
    img.AlphaData = double(valid & isfinite(values));
    ax.Color = [0.8 0.8 0.8];
    ax.YDir = 'reverse'; % RealSense camera +Y points down.
    axis(ax, 'image');
    xlim(ax, [xy.x(1), xy.x(end)]);
    ylim(ax, [xy.y(1), xy.y(end)]);
    clim(ax, limits);
    colormap(ax, colors);
    colorbar(ax);
    title(ax, caption, 'Interpreter', 'none');
    xlabel(ax, sprintf('%s X (mm)', xy.frameName));
    ylabel(ax, sprintf('%s Y (mm)', xy.frameName));
end

function limits = symmetricLimits(values, minimum)
    values = sort(abs(double(values(isfinite(values)))));
    if isempty(values)
        bound = minimum;
    else
        bound = max(minimum, values(max(1, ceil(0.98 * numel(values)))));
    end
    limits = [-bound, bound];
end

function colors = blueWhiteRed(n)
    half = n / 2;
    colors = [linspace(0.12, 1, half)', linspace(0.36, 1, half)', ones(half, 1); ...
              ones(half, 1), linspace(1, 0.20, half)', linspace(1, 0.12, half)'];
end

function plotSupport(ax, xy, support)
    cla(ax);
    imagesc(ax, xy.x, xy.y, support);
    ax.YDir = 'reverse';
    axis(ax, 'image');
    clim(ax, [-0.5 3.5]);
    colormap(ax, [0.8 0.8 0.8; 0.97 0.65 0.15; ...
                  0.25 0.52 0.83; 0.18 0.65 0.32]);
    cb = colorbar(ax);
    cb.Ticks = 0:3;
    cb.TickLabels = {'No source cell', 'Source, no fit', 'Fit, no target', 'Scored'};
    title(ax, 'Error-map support state');
    xlabel(ax, sprintf('%s X (mm)', xy.frameName));
    ylabel(ax, sprintf('%s Y (mm)', xy.frameName));
end

function plotErrorTail(ax, caError, persistenceError)
    cla(ax);
    if isempty(caError)
        text(ax, 0.5, 0.5, 'No matched valid cells', ...
            'Units', 'normalized', 'HorizontalAlignment', 'center');
        title(ax, 'Error exceedance');
        return;
    end
    thresholds = 0:2:100;
    caTail = arrayfun(@(t) mean(abs(caError) > t), thresholds);
    persistenceTail = arrayfun(@(t) mean(abs(persistenceError) > t), thresholds);
    plot(ax, thresholds, caTail, 'LineWidth', 1.5, 'DisplayName', 'CA');
    hold(ax, 'on');
    plot(ax, thresholds, persistenceTail, 'LineWidth', 1.5, ...
        'DisplayName', 'Persistence');
    ylim(ax, [0 1]);
    grid(ax, 'on');
    legend(ax, 'Location', 'northeast');
    title(ax, 'Error exceedance on scored cells');
    xlabel(ax, '|error| threshold (mm)');
    ylabel(ax, 'Fraction above threshold');
end

function evaluation = scoreForecasts(sessionDir, observationDir, records, ...
    observations, hasSource, sourceIndex, futureIndex, changeThresholdMm)
    n = numel(records);
    evaluation.count = zeros(1, n);
    evaluation.gridFraction = zeros(1, n);
    evaluation.caRmse = nan(1, n);
    evaluation.persistenceRmse = nan(1, n);
    evaluation.caP95 = nan(1, n);
    evaluation.persistenceP95 = nan(1, n);
    caParts = cell(n, 1);
    persistenceParts = cell(n, 1);
    caMoving = cell(n, 1);
    persistenceMoving = cell(n, 1);
    caStationary = cell(n, 1);
    persistenceStationary = cell(n, 1);
    scoredCells = 0;
    matchedGridCells = 0;
    targetValidCells = 0;
    scoredFrames = 0;
    for i = 1:n
        if mod(i, 100) == 0
            fprintf('Scoring forecast %d/%d...\n', i, n);
        end
        if ~hasSource(i) || futureIndex(i) == 0
            continue;
        end
        forecast = readMaps(fullfile(sessionDir, records(i).file), ...
            {'prediction_mm', 'prediction_valid'});
        source = readMaps(fullfile(observationDir, ...
            observations(sourceIndex(i)).file), {'height_mm', 'valid'});
        target = readMaps(fullfile(observationDir, ...
            observations(futureIndex(i)).file), {'height_mm', 'valid'});
        common = logical(forecast.prediction_valid) & logical(source.valid) & ...
            logical(target.valid) & isfinite(forecast.prediction_mm) & ...
            isfinite(source.height_mm) & isfinite(target.height_mm);
        matchedGridCells = matchedGridCells + numel(common);
        targetValidCells = targetValidCells + nnz(target.valid);
        evaluation.count(i) = nnz(common);
        evaluation.gridFraction(i) = nnz(common) / numel(common);
        if ~any(common, 'all')
            continue;
        end
        scoredFrames = scoredFrames + 1;
        scoredCells = scoredCells + nnz(common);
        caError = double(forecast.prediction_mm(common)) - double(target.height_mm(common));
        persistenceError = double(source.height_mm(common)) - double(target.height_mm(common));
        moving = abs(double(target.height_mm(common)) - double(source.height_mm(common))) > changeThresholdMm;
        caParts{i} = caError(:);
        persistenceParts{i} = persistenceError(:);
        caMoving{i} = caError(moving);
        persistenceMoving{i} = persistenceError(moving);
        caStationary{i} = caError(~moving);
        persistenceStationary{i} = persistenceError(~moving);
        evaluation.caRmse(i) = sqrt(mean(caError.^2));
        evaluation.persistenceRmse(i) = sqrt(mean(persistenceError.^2));
        evaluation.caP95(i) = percentileAbs(caError, 0.95);
        evaluation.persistenceP95(i) = percentileAbs(persistenceError, 0.95);
    end
    if scoredFrames == 0
        error('No forecasts have a matched future frame and common valid pixels.');
    end
    allCA = vertcat(caParts{:});
    allPersistence = vertcat(persistenceParts{:});
    evaluation.all.ca = errorStats(allCA);
    evaluation.all.persistence = errorStats(allPersistence);
    evaluation.moving.ca = errorStats(vertcat(caMoving{:}));
    evaluation.moving.persistence = errorStats(vertcat(persistenceMoving{:}));
    evaluation.stationary.ca = errorStats(vertcat(caStationary{:}));
    evaluation.stationary.persistence = errorStats(vertcat(persistenceStationary{:}));
    evaluation.allCA = allCA;
    evaluation.allPersistence = allPersistence;
    evaluation.scoredCells = scoredCells;
    evaluation.scoredFrames = scoredFrames;
    evaluation.scoredGridFraction = scoredCells / matchedGridCells;
    evaluation.scoredTargetFraction = scoredCells / targetValidCells;
    evaluation.changeThresholdMm = changeThresholdMm;
    evaluation.similarRmsePair = [0 0];
    bestTailDifference = -inf;
    for i = 1:n
        if evaluation.count(i) == 0
            continue;
        end
        for j = i+1:n
            if evaluation.count(j) == 0 || ...
                    abs(evaluation.caRmse(i)-evaluation.caRmse(j)) > 0.5
                continue;
            end
            tailDifference = abs(evaluation.caP95(i)-evaluation.caP95(j));
            if tailDifference > bestTailDifference
                bestTailDifference = tailDifference;
                evaluation.similarRmsePair = [i j];
            end
        end
    end
end

function stats = errorStats(error)
    error = double(error(:));
    if isempty(error)
        stats = struct('rmse', NaN, 'mae', NaN, 'medianAE', NaN, ...
            'p95AE', NaN, 'bias', NaN, 'above10', NaN);
        return;
    end
    stats.rmse = sqrt(mean(error.^2));
    stats.mae = mean(abs(error));
    stats.medianAE = percentileAbs(error, 0.5);
    stats.p95AE = percentileAbs(error, 0.95);
    stats.bias = mean(error);
    stats.above10 = mean(abs(error) > 10);
end

function value = percentileAbs(error, fraction)
    values = sort(abs(double(error(:))));
    if isempty(values)
        value = NaN;
        return;
    end
    rank = 1 + fraction * (numel(values) - 1);
    lower = floor(rank);
    upper = ceil(rank);
    value = values(lower) + (rank-lower) * (values(upper)-values(lower));
end

function plotErrorSummary(evaluation, records, observations, tForecast)
    fig = figure('Name', 'Prediction error beyond RMSE', 'Color', 'w', ...
        'Position', [140 80 1500 820]);
    layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact');
    title(layout, sprintf(['Matched forecasts: %d | scored pixels: %d | ' ...
        'only %.1f%% of matched grid cells scored'], ...
        evaluation.scoredFrames, evaluation.scoredCells, ...
        100 * evaluation.scoredGridFraction));

    ax = nexttile(layout);
    plot(ax, tForecast, evaluation.caRmse, '.', ...
        'DisplayName', 'CA', 'MarkerSize', 9);
    hold(ax, 'on');
    plot(ax, tForecast, evaluation.persistenceRmse, '.', ...
        'DisplayName', 'Persistence', 'MarkerSize', 9);
    xlabel(ax, 'Elapsed camera time (s)');
    ylabel(ax, 'Per-frame RMSE (mm)');
    title(ax, 'Error varies over time');
    legend(ax, 'Location', 'best');
    grid(ax, 'on');

    ax = nexttile(layout);
    bar(ax, [evaluation.all.ca.rmse, evaluation.all.persistence.rmse; ...
             evaluation.all.ca.mae, evaluation.all.persistence.mae; ...
             evaluation.all.ca.medianAE, evaluation.all.persistence.medianAE; ...
             evaluation.all.ca.p95AE, evaluation.all.persistence.p95AE]);
    ax.XTickLabel = {'RMSE', 'MAE', 'Median |e|', '95th |e|'};
    ylabel(ax, 'Error (mm)');
    title(ax, 'One RMSE hides the distribution');
    legend(ax, {'CA', 'Persistence'}, 'Location', 'best');
    grid(ax, 'on');

    ax = nexttile(layout);
    thresholds = 0:2:120;
    plot(ax, thresholds, arrayfun(@(t) mean(abs(evaluation.allCA)>t), thresholds), ...
        'LineWidth', 1.5, 'DisplayName', 'CA');
    hold(ax, 'on');
    plot(ax, thresholds, arrayfun(@(t) mean(abs(evaluation.allPersistence)>t), thresholds), ...
        'LineWidth', 1.5, 'DisplayName', 'Persistence');
    xlabel(ax, '|error| threshold (mm)');
    ylabel(ax, 'Fraction of scored cells above threshold');
    title(ax, sprintf('CA errors >10 mm: %.1f%%', 100*evaluation.all.ca.above10));
    legend(ax, 'Location', 'northeast');
    grid(ax, 'on');

    ax = nexttile(layout);
    bar(ax, [evaluation.moving.ca.rmse, evaluation.moving.persistence.rmse; ...
             evaluation.stationary.ca.rmse, evaluation.stationary.persistence.rmse]);
    ax.XTickLabel = {sprintf('Change >%.0f mm', evaluation.changeThresholdMm), ...
                     sprintf('Change <=%.0f mm', evaluation.changeThresholdMm)};
    ylabel(ax, 'RMSE (mm)');
    title(ax, 'The aggregate masks different changes');
    legend(ax, {'CA', 'Persistence'}, 'Location', 'best');
    grid(ax, 'on');

    ax = nexttile(layout);
    good = evaluation.count > 0;
    scatter(ax, evaluation.caRmse(good), evaluation.caP95(good), 24, ...
        100*evaluation.gridFraction(good), 'filled');
    hold(ax, 'on');
    pair = evaluation.similarRmsePair;
    if all(pair > 0)
        plot(ax, evaluation.caRmse(pair), evaluation.caP95(pair), ...
            'ko', 'MarkerSize', 10, 'LineWidth', 1.3);
        text(ax, evaluation.caRmse(pair), evaluation.caP95(pair), ...
            {sprintf(' #%d', pair(1)), sprintf(' #%d', pair(2))});
    end
    cb = colorbar(ax);
    cb.Label.String = 'Scored grid fraction (%)';
    xlabel(ax, 'Per-frame CA RMSE (mm)');
    ylabel(ax, 'Per-frame 95th |error| (mm)');
    title(ax, 'Similar RMSE can hide different tails');
    grid(ax, 'on');

    ax = nexttile(layout);
    bar(ax, 100*[mean([observations.valid_fraction]), ...
        mean([records.prediction_valid_fraction]), evaluation.scoredGridFraction]);
    ax.XTickLabel = {'Measured', 'Forecast fit', 'Scored overlap'};
    ylabel(ax, 'Fraction of grid (%)');
    ylim(ax, [0 100]);
    title(ax, 'Coverage is part of the result');
    grid(ax, 'on');
end

function maps = readMaps(npzFile, names)
    % NPZ is a ZIP of NPY arrays. MATLAB's unzip and this small NPY reader
    % avoid a Python bridge and require no extra toolbox.
    folder = tempname;
    mkdir(folder);
    cleanup = onCleanup(@() rmdir(folder, 's'));
    zipFile = fullfile(folder, 'arrays.zip');
    copyfile(npzFile, zipFile);
    unzip(zipFile, folder);
    maps = struct();
    for i = 1:numel(names)
        npyFile = fullfile(folder, [names{i} '.npy']);
        if ~isfile(npyFile)
            error('Missing %s in %s', names{i}, npzFile);
        end
        maps.(names{i}) = readNpy(npyFile);
    end
end

function values = readNpy(filename)
    fid = fopen(filename, 'r', 'ieee-le');
    if fid < 0
        error('Cannot open %s', filename);
    end
    cleanup = onCleanup(@() fclose(fid));
    magic = fread(fid, 6, '*uint8')';
    if ~isequal(magic, uint8([147 78 85 77 80 89]))
        error('Not a NumPy NPY file: %s', filename);
    end
    version = fread(fid, 2, '*uint8');
    if version(1) == 1
        headerLength = fread(fid, 1, 'uint16');
    elseif version(1) == 2 || version(1) == 3
        headerLength = fread(fid, 1, 'uint32');
    else
        error('Unsupported NPY version in %s', filename);
    end
    header = char(fread(fid, headerLength, '*uint8')');
    descriptor = regexp(header, '''descr'':\s*''([^'']+)''', 'tokens', 'once');
    shapeText = regexp(header, '''shape'':\s*\(([^)]*)\)', 'tokens', 'once');
    fortran = ~isempty(regexp(header, '''fortran_order'':\s*True', 'once'));
    if isempty(descriptor) || isempty(shapeText)
        error('Invalid NPY header in %s', filename);
    end
    dims = sscanf(shapeText{1}, '%d,')';
    if isempty(dims)
        count = 1;
    else
        count = prod(dims);
    end
    switch descriptor{1}
        case '<f4'
            type = 'single';
        case '<f8'
            type = 'double';
        case '|b1'
            type = 'uint8';
        case '<u4'
            type = 'uint32';
        case '<i8'
            type = 'int64';
        otherwise
            error('Unsupported NPY dtype %s in %s', descriptor{1}, filename);
    end
    raw = fread(fid, count, ['*' type]);
    if numel(raw) ~= count
        error('Truncated NPY array: %s', filename);
    end
    if strcmp(descriptor{1}, '|b1')
        raw = logical(raw);
    end
    if numel(dims) <= 1
        values = raw;
    elseif fortran
        values = reshape(raw, dims);
    else
        % NumPy stores C-order; reverse dimensions before MATLAB reshape.
        values = permute(reshape(raw, fliplr(dims)), numel(dims):-1:1);
    end
end
