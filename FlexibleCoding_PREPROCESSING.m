% FLEXIBLE CODING PREPROCESSING
% @AndreaBocincova 2019-2020-2021

% The following script load and organizes data into a format that works
% with the other scripts (Classification adn Pattern Similarity)


%%
% STEP 1
% arrange data into spike-time sequences (neuron x time point) & cropped into 
% epochs including a prestimulus interval and stimulus and delay period, i.e. 
% data up to the probe display 

% ******************************************
% ***************  Buschman  ***************
% ******************************************

% load and organize data
% directory
datDir = '/Users/andreabocincova/Desktop/Buschmanetal/nData/';
load('CapLimAnatomy_forSimon') % data file with anatomy information
nSess = 28; % number of sessions % two monkeys, each 14 session
load('/Users/andreabocincova/Desktop/Buschmanetal/TrialInfo/testData.mat')%load trial sequence data

% check how long of a pre-stimulus interval is available 
for sess = 1:nSess
    % load appropriate data file
    load(fullfile(datDir,['wm' num2str(sess) '.mat']))
    attempted_trl =  data.bhv.Attempted;
    prestim_int =  cellfun(@(x) x(14),data.bhv.CodeTimes);
    minmum_prestim(sess) = min(prestim_int(attempted_trl));
end
shortest_presstim = min(minmum_prestim); % all were longer than 500ms- 
% epochs will include 500ms prestim
common_prestim = 500;

% create spike x time sequences
% loop through sessions
for sess = 1:nSess
    disp(sess)
    % load data
    load(fullfile(datDir,['wm' num2str(sess) '.mat']))
    attempted_trl =  data.bhv.Attempted; % logical for attempted trials
    
    % spike sequences for each trial
    spikeseq = cellfun(@(x) x, data.trial);
    spikeseq = {spikeseq.spikes};
    
    % check if all attempted trials actually have data
    check = cellfun(@(x) size(x,2),spikeseq(attempted_trl));
    attempted_ind = find(attempted_trl);
    attempted_trl(attempted_ind(check == 0)) = 0;
    
    % ******************************************************
    % TRIAL EVENTS
    % Data structure  (data.bhv.CodeTimes)
    % used to select spike data for analysis
    % code 1 = trial onset?? (always in 12th position)
    % code 2 = fixation onset (always in 13th position)
    % code 3 = memory onset (always in 14th position)
    % code 4 = memory offset (always in 15th position)
    % code 5 = probe onset (always in 16th position)
    % code 6 = RT (always in 17th position)
    
    % timestamp for fixation onset (!!! variable delay between fixation and
    % stimulus)
    prestimonset = cellfun(@(x)  x(13),data.bhv.CodeTimes);
    prestimonset(~attempted_trl) = NaN; % nan aborted trials
    % timestamp for stimulus offset
    stimoffset = cellfun(@(x)  x(15),data.bhv.CodeTimes)-cellfun(@(x)  x(14),data.bhv.CodeTimes);
    stimoffset(~attempted_trl) = NaN; % nan aborted trials
    % timestamp for probe 
    probetime = zeros(1,length(attempted_trl));
    probetime(attempted_trl) = cellfun(@(x) x(16),data.bhv.CodeTimes(attempted_trl));
    probetime(~attempted_trl) = NaN;
   
    % calculate trial length in ms (i.e. from memory onset until probe)
    t_length = nan(1,length(attempted_trl));
    t_length(attempted_trl) = cellfun(@(x) x(16)-x(14),data.bhv.CodeTimes(attempted_trl));
    
    % calcualte prestimulus length in ms (i.e. from fixation until stimulus onset)
    pretr_length = zeros(1,length(attempted_trl));
    pretr_length(attempted_trl) = cellfun(@(x) x(14)-x(13),data.bhv.CodeTimes(attempted_trl));
    
    % indices for trials epoch (prestimulus + memory and delay)
    % t_interval expresses the beginning and end of each trial in ms
    t_interval = zeros(length(attempted_trl),2);
    t_interval(attempted_trl,:) = [-(pretr_length(attempted_trl));t_length(attempted_trl)-1]';
    t_interval(~attempted_trl,:) = repmat([NaN,NaN],sum(~attempted_trl),1); 
    % t_index provides the indices used to subselect the data from entire
    % trial recording
    t_index = [prestimonset;probetime]';
    
    % ******************************************************
    % Neurons used for analysis
    % identify the groups of neurons
    LIP = anatomy(sess).LIP_AnalyzedChanUnit;
    FEF = anatomy(sess).FEF_AnalyzedChanUnit;
    lPFC = anatomy(sess).lPFC_AnalyzedChanUnit;
    
    % logical for units that are to be used
    units = nan(length(data.ch),1);
    units(ismember( [data.ch;data.unit]', LIP, 'rows')) = 1;
    units(ismember( [data.ch;data.unit]', lPFC, 'rows')) = 2;
    units(ismember( [data.ch;data.unit]', FEF, 'rows')) = 3;
    units_names = repmat({NaN},length(data.ch),1);
    units_names(units==1) = {'LIP'};
    units_names(units==2) = {'lPFC'};
    units_names(units==3) = {'FEF'};
    % for each trial
    units = repmat({units},1,length(spikeseq));
    units_names = repmat({units_names},1,length(spikeseq));
    channels.units_names = units_names;
    channels.units = units;
    
    t_indices =  num2cell(t_index',1); % convert indices to cells for cellfun
    
    % ******************************************************
    % spikes for each trial
    spikes_alltrials = cell(1,length(attempted_trl));
    % grab spike sequences for attempted trials, units of interest and
    % subselect data based on t_indices
    spikes_alltrials(attempted_trl) = cellfun(@(x,z) x(:,z(1):z(2)),spikeseq(attempted_trl),...
        t_indices(attempted_trl),'UniformOutput',false);  % grab spiking activity within trial
    spikes_alltrials(~attempted_trl)  = {NaN};
    
    % align to zero - because prestimulus interval varies, data need to be
    % aligned to zero to create consistent structure for all trials
    desired_prestimulus = num2cell(repmat(common_prestim,1,length(attempted_trl)));
    actual_prestimulus =  num2cell(t_interval',1); % convert time intervals to cells to use in cellfun
    spikes_alltrials_aligned = cell(1,length(attempted_trl));
    spikes_alltrials_aligned(attempted_trl) = cellfun(@(x,y,z) x(:,abs(y(1))-z+1:end), ...
        spikes_alltrials(attempted_trl),actual_prestimulus(attempted_trl),...
        desired_prestimulus(attempted_trl),'UniformOutput',false);
    spikes_alltrials_aligned(~attempted_trl)  = {NaN};
       
    % adjust time interval values to desired_prestimulus
    t_interval(attempted_trl,1) = -500;
    
    % concatenate trials into trial x unit x time matrix 
    spikes_alltrial_mat = nancat(3,spikes_alltrials_aligned{:});
    spikes_alltrial_mat = permute(spikes_alltrial_mat,[3,1,2]);
    
%     % plot the matrix for SANITY CHECK
%     % convolve with gaussian for better visibility of spikes
%     K =  gauss(20,1);
%     temp = nanconv( spikes_alltrial_mat, permute( K, [1,3,2] ) );
%     all_neurons_spikes = reshape(permute(temp,[3,1,2]),size(spikes_alltrial_mat,3),...
%     size(spikes_alltrial_mat,2)*size(spikes_alltrial_mat,1))';
%     all_attempted = repmat(attempted_trl,size(spikes_alltrial_mat,2),1);
%     time = -common_prestim:max(t_interval(:,2))+1;
%     trials = 1:sum(all_attempted);
%     imagesc(time,trials,all_neurons_spikes(all_attempted,:))
%     xline(0)
%     figure
%     plot(time,nanmean(normalize(all_neurons_spikes(all_attempted,:),2),1))
%     xline(0)
    

    % calculate FR (matrix neuron x trial)
    FR = sq(nanmean(spikes_alltrial_mat(:,:,common_prestim+1:end),3))';
   
    % calculate baseline FR (matrix neuron x trial)
    FR_baseline = sq(nanmean(spikes_alltrial_mat(:,:,1:common_prestim),3))';
    
    
   % CONDITIONS 
   % locations
   % 12 rows - 6 possible locations, 2 possible stimuli at each location
   % row 1:2 - location 1, color 1 and color 2
   % row 3:4 - location 2, color 3 and color 4
   % etc
    stimuli = testData(sess).behav.StimArray.*repmat([1;2],...
        size(testData(sess).behav.StimDisp,1),1);
    trial_sequence = [sum(stimuli(1:2,:));sum(stimuli(3:4,:));sum(stimuli(5:6,:));...
        sum(stimuli(7:8,:));sum(stimuli(9:10,:));sum(stimuli(11:12,:))];
    
    % ******************************************************
    % combine all information into a table
    all_data = table(repmat(data.bhv.SubjectName,length(data.bhv.TrialNumber),1),...
        repmat(sess,length(data.bhv.TrialNumber),1),...
        data.bhv.TrialNumber,attempted_trl,prestimonset',probetime',stimoffset',t_length',t_interval,...
        t_index,units_names',units',trial_sequence',spikes_alltrials_aligned',...
        'VariableNames',{'Monkey','SessionNumber','TrialNumber','AttemptedTrials',...
        'PrestimOnsetTstamp','ProbetOnsetTstamp','StimPresLength','TrialLength','TimeInterval',...
        'TrialTimeIndex','UnitLocation','UnitIndices','ConditionLoc123456','Spikes'});
    
    cd('~/Desktop')
    if ~exist('BuschmanData', 'dir')
        mkdir('BuschmanData')
    end
    cd('~/Desktop/BuschmanData/')
    save(['Buschman_sess' num2str(sess) '_spikes'],'all_data','-v7.3')
    save(['Buschman_sess' num2str(sess) '_FR'],'FR','FR_baseline','trial_sequence','channels')
    
end
    
%%

% ******************************************
% **************  Lundqvist  ***************
% ******************************************    

DataOrig_Dir = '/Users/andreabocincova/Desktop/Lundqvistetal/Lundqvist2016Data/data/original';%'/Users/andreabocincova/Desktop/projects/Lundqvistetal/Lundqvist2016Data/Data/original/';

listing = dir(DataOrig_Dir);
names = [{listing(:).name}];
fun1 = @(x) length(x);
f_ind = find(cellfun(fun1,names)==24);
% flip indices to match monkeys from Buschman data
f_ind = [f_ind(15:28),f_ind(1:14)];

Sess_no = length(f_ind);
cd(DataOrig_Dir)
sess = 1;
load(listing(f_ind(sess)).name,'spikeTimesSchema','spikeTimes')

[trials, unit] =  size(spikeTimes);
time_epoch = spikeTimesSchema.userData.timeEpoch;
srate = spikeTimesSchema.smpRate;
time = linspace(time_epoch(1),time_epoch(2),srate/30*(time_epoch(2)-time_epoch(1)));
common_prestim = 300;

% NOTES ON SPIKES
% cells are from the PFC, LIP and FEF
% time 0 correspons to the start of the presentation of S1
% each sample stimulus was presented for .3 seconds
% ISI .3 seconds
% final delay 1.2 seconds for Load 2 and .6 for Load 3
% timing for load 2 .3+.3+.3+.3+.1.2 = 2.4 seconds up to test display (2.1
% according to plots?)
% timing for load 2 .3+.3+.3+.3+.3+.3+.6 = 2.4 seconds up to test display (2.1
% according to plots?)    

% Load all neural data and organize into a variable (NEW)
% ******************************************************
for sess = 1:Sess_no
    cd(DataOrig_Dir)
    disp(sess)
    
    % ******************************************************
    % spikes
    load(listing(f_ind(sess)).name,'spikeTimesSchema','spikeTimes','unitInfo',...
        'sessionInfo','trialInfo')
    
    % ******************************************************
    % variables
    [trials, unit] =  size(spikeTimes);
    time_epoch = spikeTimesSchema.userData.timeEpoch;
    attempted_trl = ~trialInfo.badTrials;% logical for attempted trials
    
    % ******************************************************
    % calculate trial length in ms (i.e. from fixation onset until probe)
    t_length = nan(1,length(attempted_trl));
    t_length(attempted_trl) = cellfun(@(x,y) length(x:y),...
        num2cell(round(trialInfo.fixationTime(attempted_trl)*1000)),...
        num2cell(round(trialInfo.testOn(attempted_trl,1)*1000-1)));

    % indices for trials epoch (prestimulus + memory and delay)
    % t_interval expresses the beginning and end of each trial in ms
    t_interval = zeros(length(attempted_trl),2);
    t_interval(attempted_trl,:) = round([trialInfo.fixationTime(attempted_trl)*1000,...
        trialInfo.testOn(attempted_trl,1)*1000-1]);
    t_interval(~attempted_trl,:) = repmat([NaN,NaN],sum(~attempted_trl),1); 
    % t_index provides the indices used to subselect the data (1st stimulus utnil probe) 
    % from the entire trial recording within "time_epoch"
    t_index = num2cell(nan(length(attempted_trl),2),2);
    t_index(attempted_trl,:) = cellfun(@(x,y) dsearchn(x',y')',repmat({time_epoch(1)*1000:time_epoch(2)*1000},...
        sum(attempted_trl),1), num2cell(t_interval(attempted_trl,:),2), 'UniformOutput', false);
    t_index = cat(1,t_index{:});
    t_indices =  num2cell(t_index',1); % convert indices to cells for cellfun
      

    % time intervals for individual stimuli (sequential presentation)
    t_interval_stim1 = nan(length(attempted_trl),2);
    t_interval_stim1(attempted_trl,:) = round([trialInfo.sampleOn(attempted_trl,1),...
        trialInfo.sampleOff(attempted_trl,1)]*1000); % in ms
    t_interval_stim2 = nan(length(attempted_trl),2);
    t_interval_stim2(attempted_trl,:) = round([trialInfo.sampleOn(attempted_trl,2),...
        trialInfo.sampleOff(attempted_trl,2)]*1000);
    t_interval_stim3 = nan(length(attempted_trl),2);
    t_interval_stim3(attempted_trl,:) = round([trialInfo.sampleOn(attempted_trl,3),...
        trialInfo.sampleOff(attempted_trl,3)]*1000);
    
%     % create an index for the different events from it
%     % (i.e., stim presentation + delay for each stimulus)
%     t_index_stim1 = nan(length(attempted_trl),2);
%     t_index_stim1(attempted_trl,:) = [t_index(attempted_trl,1),...
%         t_index(attempted_trl,1)+600]+abs(t_interval(attempted_trl,1));
%     t_index_stim2 = nan(length(attempted_trl),2);
%     t_index_stim2(attempted_trl,:) = [t_index(attempted_trl,1)+t_interval_stim2(attempted_trl,1),...
%         t_index(attempted_trl,1)++t_interval_stim2(attempted_trl,1)+600]+...
%         abs(t_interval(attempted_trl,1));
%     t_index_stim3 = zeros(length(attempted_trl),2);
%     t_index_stim3(attempted_trl,:) = [t_index(attempted_trl,1)+t_interval_stim3(attempted_trl,1),...
%        t_index(attempted_trl,1)+t_interval_stim3(attempted_trl,1)+600]+...
%        abs(t_interval(attempted_trl,1)); % last delay is ~300ms longer than ISIs so removing that part
    
time_ind(1,:) = dsearchn(time',[0,.6]')';
time_ind(2,:) = dsearchn(time',[0.6,1.2]')';
time_ind(3,:) = dsearchn(time',[1.2,1.8]')';
    % create an index for the different events from it
    % (i.e., stim presentation + delay for each stimulus)
    t_index_stim1 = nan(length(attempted_trl),2);
    t_index_stim1(attempted_trl,:) = repmat([time_ind(1,1),time_ind(1,2)],...
        sum(attempted_trl),1);
    t_index_stim2 = nan(length(attempted_trl),2);
    t_index_stim2(attempted_trl,:) = repmat([time_ind(2,1),time_ind(2,2)],...
        sum(attempted_trl),1);
    t_index_stim3 = zeros(length(attempted_trl),2);
    t_index_stim3(attempted_trl,:) = repmat([time_ind(3,1),time_ind(3,2)],...
        sum(attempted_trl),1); % last delay is ~300ms longer than ISIs so removing that part
    
   
    %     % SANITY CHECK (each stimulu+delay should be ~600ms
    %     t_index_stim1(:,2)-t_index_stim1(:,1)
    %     t_index_stim2(:,2)-t_index_stim2(:,1)
    %     t_index_stim3(:,2)-t_index_stim3(:,1)
    
    % ******************************************************
    % units
    units_names = [unitInfo.area];
    units = nan(length(units_names),1);
    units(strcmp( units_names', 'LIP')) = 1;
    units(strcmp( units_names', 'PFC')) = 2;
    units(strcmp( units_names', 'FEF')) = 3;
    
    units = repmat({units},1,length(attempted_trl));
    units_names = repmat({units_names},1,length(attempted_trl));
    
    channels.units_names = units_names;
    channels.units = units;
    
    % ******************************************************
    % spikes
    % create spike sequence within epoch defined in "time_epoch"
    spikeseq = [];
    for trl = 1:trials
        for cells = 1:unit
            spike_seq = zeros(1,length(time));
            if ~isempty([spikeTimes{trl,cells}])
                ind = dsearchn(time',spikeTimes{trl,cells}');
                spike_seq(ind) = 1;
            else
                spike_seq(:) = nan;
            end
            spikeseq{trl}(cells,:) = spike_seq;
        end % cell
    end % trl
     
    % spikes for each trial
    spikes_alltrials = cell(1,length(attempted_trl));
    % grab spike sequences for attempted trials, units of interest and
    % subselect data based on t_indices
    spikes_alltrials(attempted_trl) = cellfun(@(x,z) x(:,z(1):z(2)),spikeseq(attempted_trl),...
         t_indices(attempted_trl),'UniformOutput',false);  % grab spiking activity within trial
    spikes_alltrials(~attempted_trl)  = {NaN};
    
    % align to zero - because prestimulus interval varies, data need to be
    % aligned to zero to create consistent structure for all trials
    desired_prestimulus = num2cell(repmat(common_prestim,1,length(attempted_trl)));
    actual_prestimulus =  num2cell(t_interval',1); % convert time intervals to cells to use in cellfun
    spikes_alltrials_aligned = cell(1,length(attempted_trl));
    spikes_alltrials_aligned(attempted_trl) = cellfun(@(x,y,z) x(:,abs(y(1))-z+1:end), ...
        spikes_alltrials(attempted_trl),actual_prestimulus(attempted_trl),...
        desired_prestimulus(attempted_trl),'UniformOutput',false);
    spikes_alltrials_aligned(~attempted_trl)  = {NaN};
    
    % concatenate trials into trial x unit x time matrix 
    spikes_alltrial_mat = nancat(3,spikes_alltrials_aligned{:});
    spikes_alltrial_mat = permute(spikes_alltrial_mat,[3,1,2]);
    
    
%     % plot the matrix for SANITY CHECK
%     % convolve with gaussian for better visibility of spikes
%     K =  gauss(200,1);
%     temp = nanconv( spikes_alltrial_mat, permute( K, [1,3,2] ) );
%     all_neurons_spikes = reshape(permute(temp,[3,1,2]),size(spikes_alltrial_mat,3),...
%     size(spikes_alltrial_mat,2)*size(spikes_alltrial_mat,1))';
%     all_attempted = repmat(attempted_trl,size(spikes_alltrial_mat,2),1);
%     time = -common_prestim:max(t_interval(:,2));
%     trials = 1:sum(all_attempted);
%     imagesc(time,trials,all_neurons_spikes(all_attempted,:))
%     xline(0)
%     figure
%     plot(time,nanmean(normalize(all_neurons_spikes(all_attempted,:),2),1))
%     xline(0)
   
    % calculate FR (matrix neuron x trial)
    FR_entiretrial = sq(nanmean(spikes_alltrial_mat(:,:,common_prestim+1:end),3))';
    % calculate baseline FR (matrix neuron x trial)
    FR_baseline = sq(nanmean(spikes_alltrial_mat(:,:,1:common_prestim),3))';    
    
    FR_peritem = cell(length(attempted_trl),3);
    % stimulus1 FR
    FR_peritem(attempted_trl,1) = cellfun(@(x,y) nanmean(x(:,y(1):y(2)),2),...
        spikeseq(attempted_trl), num2cell(t_index_stim1(attempted_trl,:),2)',...
        'UniformOutput', false);
    FR_peritem(~attempted_trl,1) = {nan(length(units{1}),1)};
    FR_peritem(attempted_trl,2) = cellfun(@(x,y) nanmean(x(:,y(1):y(2)),2),...
        spikeseq(attempted_trl), num2cell(t_index_stim2(attempted_trl,:),2)',...
        'UniformOutput', false);
    FR_peritem(~attempted_trl,2) =  {nan(length(units{1}),1)};
    ss3trials = ~(isnan(t_index_stim3(:,1)) | t_index_stim3(:,1)==0);
    FR_peritem(ss3trials,3) = cellfun(@(x,y) nanmean(x(:,y(1):y(2)),2),...
        spikeseq(ss3trials), num2cell(t_index_stim3(ss3trials,:),2)',...
        'UniformOutput', false);
    FR_peritem(~ss3trials,3) = {nan(length(units{1}),1)};
    
    FR{1} = [FR_peritem{:,1}];
    FR{2} = [FR_peritem{:,2}];
    FR{3} = [FR_peritem{:,3}];
    
    % CONDITIONS   
    load(listing(f_ind(sess)).name,'trialInfo')
    
    trialSequence = nan(3,length(trialInfo.sampleLocs));
    trialSequence_new = nan(3,length(trialInfo.sampleLocs));
    for trl = 1:length(trialInfo.sampleLocs)
        if ~trialInfo.badTrials(trl)
            if trialInfo.load(trl) == 2
                trialSequence(trialInfo.sampleLocs(trl,1:2),trl) = trialInfo.sampleColors(trl,1:2);
                trialSequence(find(ismember([1 2 3],trialInfo.sampleLocs(trl,1:2))==0),trl) = 0;
            elseif trialInfo.load(trl) == 3
                trialSequence(trialInfo.sampleLocs(trl,:),trl) = trialInfo.sampleColors(trl,:);
            end
        end
    end
      
    order = trialInfo.sampleLocs;
    order(trialInfo.badTrials,:) = NaN;
    trial_sequence.trialseqorig = trialSequence;
    % recode colors as color 1 or color 2 per location
    % location 1
    trialSequence_new(1,trialSequence(1,:) == 1) = 1;
    trialSequence_new(1,trialSequence(1,:) == 4) = 2;
    trialSequence_new(1,trialSequence(1,:) == 0) = 0;
    % location 2
    trialSequence_new(2,trialSequence(2,:) == 2) = 1;
    trialSequence_new(2,trialSequence(2,:) == 5) = 2;
    trialSequence_new(2,trialSequence(2,:) == 0) = 0;
    % location 3
    trialSequence_new(3,trialSequence(3,:) == 3) = 1;
    trialSequence_new(3,trialSequence(3,:) == 6) = 2;
    trialSequence_new(3,trialSequence(3,:) == 0) = 0;
    
    trial_sequence.trialseq = trialSequence_new;
    trial_sequence.order = order;   
    
    % combine all information into a table
    all_data = table(repmat({sessionInfo.subject},length(attempted_trl),1),...
        repmat(sess,length(attempted_trl),1),[1:length(attempted_trl)]',...
        attempted_trl,t_length',t_interval,t_index,...
        [t_interval_stim1,t_interval_stim2,t_interval_stim3],...
        [t_index_stim1,t_index_stim2,t_index_stim3],...
        units_names',units',order,trialSequence_new',...
        spikeseq',spikes_alltrials_aligned',...
        'VariableNames',{'Monkey','SessionNumber','TrialNumber','AttemptedTrials',...
        'TrialLength','TimeInterval','TrialTimeIndex','StimulusSpecificTimeIntervals',...
        'StimulusSpecificIndices','UnitLocation','UnitIndices','Order123Loc',...
        'ColorLoc123','RawSpikes','EpochedSpikes'});

    
    cd('~/Desktop')
    if ~exist('LundqvistData', 'dir')
        mkdir('LundqvistData')
    end
    cd('~/Desktop/LundqvistData')
    % ******************************************************
    % save
    save(['Lundqvist_sess' num2str(sess) '_spikes'],'all_data','-v7.3')
    save(['Lundqvist_sess' num2str(sess) '_FR'],'FR_entiretrial','FR','FR_baseline',...
        'trial_sequence','channels')
    
end


%% Store all FRs in one variable

dirBusch = '~/Desktop/BuschmanData/';
dirLund =  '~/Desktop/LundqvistData/';

% initialize
FR_baseline_lundqvist_peritem = cell(28,3);
FR_all_lundqvist_peritem = cell(28,3);
% loop through sessions
for sess = 1:28
    % Buschman
    load([dirBusch '/Buschman_sess' num2str(sess) '_FR'])
    FR_all_buschman{sess} = FR;
    FR_baseline_buschman{sess} = FR_baseline;
    buschman_trial_sequence{sess} = trial_sequence;
    buschman_channels{sess} = channels.units;
    
    % Lundqvist
    load([dirLund '/Lundqvist_sess' num2str(sess) '_FR'])
    FR_all_lundqvist{sess} = FR_entiretrial;
    FR_all_lundqvist_peritem(sess,1:3) = FR;
    FR_baseline_lundqvist{sess} = FR_baseline;
    FR_baseline_lundqvist_peritem(sess,1:3) = [{FR_baseline}, {FR_baseline}, {FR_baseline}];
    lundqvist_trial_sequence{sess} = trial_sequence.trialseq;
    lundqvist_order_sequence{sess} = trial_sequence.order;
    lundqvist_channels{sess} = channels.units;
end

%% Behavior

% directory
datDir = '/Users/andreabocincova/Desktop/Buschmanetal/nData/';
nSess = 28; % number of sessions

% loop through sessions
for sess = 1:nSess
    disp(sess)
    % load appropriate data file
    load(fullfile(datDir,['wm' num2str(sess) '.mat']))
    
    buschman_proportion_correct(sess) = ...
        sum(data.bhv.Correct(data.bhv.Attempted))/length(data.bhv.Correct...
        (data.bhv.Attempted));
end

% directory
datDir = '/Users/andreabocincova/Desktop/Lundqvistetal/Lundqvist2016Data/data/original';
nSess = 28; % number of sessions
listing = dir(datDir);
names = {listing.name};
names(1:2) = [];

% loop through sessions
for sess = 1:nSess
    disp(sess)
    % load appropriate data file
    load(fullfile(datDir,names{sess}))
    
    lundqvist_proportion_correct(sess) = ...
        sum(trialInfo.correct(trialInfo.badTrials==0))/...
        length(trialInfo.correct(trialInfo.badTrials==0));
end

%%

cd(dirBusch)
save('Buschman_data_all_sess','FR_all_buschman','FR_baseline_buschman',...
    'buschman_proportion_correct','buschman_trial_sequence','buschman_channels','-v7.3')
cd(dirLund)
save('Lundqvist_data_all_sess','FR_all_lundqvist','FR_baseline_lundqvist',...
    'lundqvist_proportion_correct','lundqvist_trial_sequence',...
    'lundqvist_order_sequence','lundqvist_channels', '-v7.3')
save('Lundqvist_data_peritem_all_sess','FR_all_lundqvist_peritem','FR_baseline_lundqvist_peritem',...
    'lundqvist_trial_sequence','lundqvist_order_sequence','lundqvist_channels','-v7.3')

 
 
 

