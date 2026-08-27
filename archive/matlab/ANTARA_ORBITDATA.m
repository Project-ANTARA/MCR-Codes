%% ANTARA Mission ConOps Analysis
% Classifies orbits into 4 Types and compares statistics.
clear; clc; close all;

% --- 1. CONFIGURATION & DATA LOADING ---
filename = 'ANTARA_conops_log.csv'; % Ensure this file exists
fprintf('Loading data from %s...\n', filename);

opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
T = readtable(filename, opts);

% Extract columns for speed
OrbitNum = T.("Orbit #");
Time = T.("Time (s)");
Lat = T.Latitude;
Lon = T.Longitude;
IsGS = strcmpi(string(T.("GS (Goa)")), 'True');
IsSAA = strcmpi(string(T.("In SAA")), 'True');
IsEclipse = strcmpi(string(T.Eclipse), 'True');

% Get unique orbits (skip first/last if incomplete, but let's keep all for now)
uOrbits = unique(OrbitNum);
numOrbits = length(uOrbits);

% --- 2. ORBIT CLASSIFICATION LOOP ---
% We will store stats for every single orbit here
orbit_stats = struct('OrbitID', [], 'Type', [], 'TypeStr', [], ...
                     'Period', [], 'EclipseDur', [], 'GSDur', [], 'SAADur', []);

fprintf('Analyzing %d orbits...\n', numOrbits);

for i = 1:numOrbits
    oid = uOrbits(i);
    idx = (OrbitNum == oid); % Logical mask for this orbit
    
    % Slice data
    t_orb = Time(idx);
    gs_flag = any(IsGS(idx));
    saa_flag = any(IsSAA(idx));
    
    % 1. DETERMINE TYPE
    if ~gs_flag && ~saa_flag
        type_code = 1; type_str = 'Type 1: Quiet';
    elseif gs_flag && ~saa_flag
        type_code = 2; type_str = 'Type 2: Only GS';
    elseif ~gs_flag && saa_flag
        type_code = 3; type_str = 'Type 3: Only SAA';
    else
        type_code = 4; type_str = 'Type 4: Mixed';
    end
    
    % 2. CALCULATE DURATIONS (Approx: Count * 60s)
    % For precise duration, we sum the boolean flags * step size
    step_size = 60; 
    ecl_dur = sum(IsEclipse(idx)) * step_size;
    gs_dur = sum(IsGS(idx)) * step_size;
    saa_dur = sum(IsSAA(idx)) * step_size;
    period = length(t_orb) * step_size; % Approx period based on file rows
    
    % Save
    orbit_stats(i).OrbitID = oid;
    orbit_stats(i).Type = type_code;
    orbit_stats(i).TypeStr = type_str;
    orbit_stats(i).Period = period;
    orbit_stats(i).EclipseDur = ecl_dur;
    orbit_stats(i).GSDur = gs_dur;
    orbit_stats(i).SAADur = saa_dur;
end

% Convert to Table for easier analysis
StatsT = struct2table(orbit_stats);

% --- 3. GENERATE STATISTICS REPORT ---
fprintf('\n--- STATISTICS BY ORBIT TYPE ---\n');
type_groups = findgroups(StatsT.Type);

% Calculate Means and Stds
StatsSummary = table();
StatsSummary.Type = [1;2;3;4];
StatsSummary.Label = {'Quiet'; 'Only GS'; 'Only SAA'; 'Mixed'};
StatsSummary.Count = splitapply(@length, StatsT.Period, type_groups);
StatsSummary.Mean_Period_min = splitapply(@mean, StatsT.Period, type_groups) / 60;
StatsSummary.Mean_Eclipse_min = splitapply(@mean, StatsT.EclipseDur, type_groups) / 60;
StatsSummary.Mean_GS_min = splitapply(@mean, StatsT.GSDur, type_groups) / 60;
StatsSummary.Mean_SAA_min = splitapply(@mean, StatsT.SAADur, type_groups) / 60;

disp(StatsSummary);


% --- 4. PLOTTING ---

% Color Map: Green (Quiet), Blue (GS), Red (SAA), Magenta (Mixed)
colors = [0 0.8 0; 0 0 1; 1 0 0; 1 0 1]; 

% FIGURE 1: THE ORBIT "HEARTBEAT" (Evolution)
figure('Name', 'Orbit Type Evolution', 'Color', 'w');
scatter(StatsT.OrbitID, StatsT.Type, 50, colors(StatsT.Type, :), 'filled');
yticks([1 2 3 4]);
yticklabels({'Type 1 (Quiet)', 'Type 2 (GS)', 'Type 3 (SAA)', 'Type 4 (Mixed)'});
xlabel('Orbit Number');
title('Orbit Type Evolution Sequence (The "Heartbeat")');
grid on;
ylim([0.5 4.5]);

% FIGURE 2: REPRESENTATIVE GROUND TRACKS
figure('Name', 'Ground Tracks by Type', 'Color', 'w');
hold on; axis equal; grid on;
xlabel('Longitude'); ylabel('Latitude');
title('Representative Ground Tracks for Each Type');
xlim([-180 180]); ylim([-90 90]);

% Draw approximate continents (simple box for context)
rectangle('Position',[-180 -90 360 180], 'FaceColor', [0.95 0.95 1]); % Ocean
% SAA Box
rectangle('Position',[-90 -50 130 50], 'EdgeColor','r','LineStyle','--', 'LineWidth',1.5);
text(-80, -40, 'SAA Zone', 'Color', 'r');
% GS Box
rectangle('Position',[60 0 20 20], 'EdgeColor','b','LineStyle','--', 'LineWidth',1.5);
text(62, 22, 'GS Goa', 'Color', 'b');

% Plot one example of each type
types_found = unique(StatsT.Type);
for t = types_found'
    % Find first orbit of this type
    first_idx = find(StatsT.Type == t, 1);
    oid_example = StatsT.OrbitID(first_idx);
    
    % Get Lat/Lon data
    mask = (OrbitNum == oid_example);
    lats = Lat(mask);
    lons = Lon(mask);
    
    % Handle Longitude wrapping for clean plotting
    % (Insert NaN where lon jumps from 180 to -180)
    jump_idx = find(abs(diff(lons)) > 300);
    lats(jump_idx) = NaN; 
    lons(jump_idx) = NaN;
    
    p = plot(lons, lats, 'LineWidth', 2, 'Color', colors(t,:));
    
    % Add legend entry
    p.DisplayName = sprintf('Type %d (Orbit %d)', t, oid_example);
end
legend('show', 'Location', 'southoutside');


% FIGURE 3: DURATION COMPARISON
figure('Name', 'Duration Analysis', 'Color', 'w');
dur_data = [StatsSummary.Mean_Eclipse_min, StatsSummary.Mean_GS_min, StatsSummary.Mean_SAA_min];
b = bar(dur_data, 'grouped');
xticks(1:4);
xticklabels({'Quiet', 'GS Only', 'SAA Only', 'Mixed'});
ylabel('Duration (minutes)');
legend({'Eclipse', 'GS Pass', 'SAA Pass'}, 'Location', 'NorthWest');
title('Average Event Durations per Orbit Type');
grid on;

% Save the detailed list (Orbit #, Type, Durations)
writetable(StatsT, 'ANTARA_Orbit_Classification.csv');
fprintf('Detailed classification saved to "ANTARA_Orbit_Classification.csv"\n');

% Save the summary table (Averages per Type)
writetable(StatsSummary, 'ANTARA_ConOps_Summary.csv');
fprintf('Summary statistics saved to "ANTARA_ConOps_Summary.csv"\n');

fprintf('Analysis Complete. Check Figures.\n');