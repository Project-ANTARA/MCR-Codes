% ANALYZE_CONOPS.m
% Script to generate ConOps statistics from 6-day Orbit Data CSV

clear; clc;

% --- 1. LOAD DATA ---
filename = 'ANTARA_conops_log.csv'; % Make sure this matches your file name
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve'; % Keeps column names like "GS (Goa)" intact
T = readtable(filename, opts);

% Extract arrays for easier handling
Time = T.("Time (s)");
Lat = T.Latitude;
Eclipse_Raw = T.Eclipse;     % Likely cell array of strings 'True'/'False'
SAA_Raw = T.("In SAA");
GS_Raw = T.("GS (Goa)");

% Convert Text 'True'/'False' to Logical 1/0
IsEclipse = strcmpi(string(Eclipse_Raw), 'True');
IsSAA = strcmpi(string(SAA_Raw), 'True');
IsGS = strcmpi(string(GS_Raw), 'True');
IsSunlit = ~IsEclipse;

% --- 2. CALCULATE ORBIT PERIOD (Ascending Node) ---
% Find where Lat goes from negative to positive
crossings = find(Lat(1:end-1) < 0 & Lat(2:end) >= 0);
node_times = zeros(length(crossings), 1);

for i = 1:length(crossings)
    idx = crossings(i);
    t1 = Time(idx); t2 = Time(idx+1);
    l1 = Lat(idx);  l2 = Lat(idx+1);
    
    % Linear Interpolation for Lat = 0
    fraction = -l1 / (l2 - l1);
    node_times(i) = t1 + fraction * (t2 - t1);
end

OrbitPeriods = diff(node_times);

% --- 3. HELPER FUNCTION TO GET DURATIONS ---
function durs = get_durations(time_vec, bool_vec)
    % Find transitions
    % Pad with 0 to catch start/end events
    padded = [0; bool_vec; 0];
    diffs = diff(padded);
    
    starts = find(diffs == 1); % Index in time_vec where block starts
    ends = find(diffs == -1) - 1; % Index where block ends
    
    durs = zeros(length(starts), 1);
    
    % Step size (assuming constant 60s, or calculate dynamic)
    % We estimate duration = (Count of True) * StepSize
    % Or (Time_End - Time_Start)
    
    for k = 1:length(starts)
        s_idx = starts(k);
        e_idx = ends(k);
        
        % Robust duration: (Time at end) - (Time at start) + 1 step coverage
        % or simply count * 60
        t_start = time_vec(s_idx);
        t_end = time_vec(e_idx);
        
        % If single point, duration is 1 step (60s)
        if s_idx == e_idx
            durs(k) = 60;
        else
            durs(k) = (t_end - t_start) + 60; 
        end
    end
end

% --- 4. CALCULATE PARAMETERS ---
EclipseDurs = get_durations(Time, IsEclipse);
SunlitDurs = get_durations(Time, IsSunlit);
SAADurs = get_durations(Time, IsSAA);
GSDurs = get_durations(Time, IsGS);

% --- 5. COMPUTE STATISTICS ---
function stats_row = calc_stats(name, data)
    if isempty(data)
        stats_row = {name, 0, 0, 0, 0, 0, 0};
    else
        stats_row = {name, mean(data), var(data), std(data), min(data), max(data), length(data)};
    end
end

ResultRows = [
    calc_stats('Orbit Period', OrbitPeriods);
    calc_stats('Eclipse Duration', EclipseDurs);
    calc_stats('Sunlit Duration', SunlitDurs);
    calc_stats('GS Pass (Goa)', GSDurs);
    calc_stats('SAA Pass', SAADurs)
];

% --- 6. DISPLAY RESULTS ---
ResultTable = cell2table(ResultRows, ...
    'VariableNames', {'Parameter', 'Mean_s', 'Variance_s2', 'StdDev_s', 'Min_s', 'Max_s', 'Count'});

% Add Mean in Minutes for easier reading
ResultTable.Mean_min = ResultTable.Mean_s / 60;

disp('--- CONOPS ANALYSIS REPORT (6 DAYS) ---');
disp(ResultTable);

% Optional: Save to file
writetable(ResultTable, 'ConOps_Stats_Output.csv');
disp('Stats saved to ConOps_Stats_Output.csv');