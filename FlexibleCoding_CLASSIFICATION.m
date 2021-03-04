% FLEXIBLE CODING CLASSIFICATION

% @AndreaBocincova 2019-2020-2021

% This scripts loads neural data from two different studies and runs
% different types of classification to decode the contents of the neural
% activity

% 1. Classification over time- temprally resolved classification of
% information over the time course of a trial

% 2. Classification generalization- looks at how well a classifier trained
% at one part of the sessions can decode information in other parts of the
% session- the idea here is that if the representation of information in spikes
% evolves over time, generalization should be relatively high if trained and
% tested on blocks close in time, but poor if trained and tested on trials far apart in time

% 3. Window trained classification generalization- same as above but this
% time a window oftrials of different sizes is used to train the classifier
% and all trials outside the window are tested for generalization


% Functions called within the script:
% simple_classification_overtime
% classification_generalization
% train_on_moving_subset_test_on_the_rest


%% General Setup

addpath('/Users/andreabocincova/Documents/MATLAB/matlib')
dirBusch = '~/Desktop/BuschmanData/';
dirLund =  '~/Desktop/LundqvistData/';

%% *******************************************************
%  *******************************************************
%  ************* 1. Classification over time *************
%  *******************************************************
%  *******************************************************

% general settings
iter = 5;
time_window = 100;
time_step = 50;
smooth = 1;

%% 
% BUSCHMAN DATA SET

% specific settings
loc_no = 6;
channel2use = 3; % 1-LIP, 2-PFC, 3-FEF

% Load data
for sess = 1:28
    load([dirBusch 'Buschman_sess' num2str(sess) '_spikes.mat'])
    labels_color = all_data.ConditionLoc123456';
    
     % channels
    chan = all_data.UnitIndices{1} == channel2use;
    
    % recode 0 to nans
    labels_color(labels_color == 0) = NaN;
    spikes = nancat(3,all_data.Spikes{:});
    spikes = spikes(chan,:,:);
    spikes = permute(spikes,[3,1,2]);
    
    
    labels_loc = all_data.ConditionLoc123456';
    labels_loc(labels_loc == 2) = 1;
    
    % convolve spikes with gaussian
    if smooth == 1 % smooth data using gaussian kernel
        K =  gauss(20,1); %(exp(-[-100:100].^2/10000)); % short gaussian
        spikes = nanconv( spikes, permute( K, [1,3,2] ) );
    end
    
    % loop through locations
    parfor loc = 1:loc_no
        % classify color per location
        classification_accuracy_color(sess,loc,:) = ...
            nanmean(simple_classification_overtime(spikes, labels_color(loc,:),...
            iter, time_window, time_step),1);
        
        % classify locations (prsent vs absent)
        classification_accuracy_loc(sess,loc,:) = ...
            nanmean(simple_classification_overtime(spikes, labels_loc(loc,:),...
            iter, time_window, time_step),1);
    end
end

%%

name2save = 'Buschman_classification_over_time_FEF';

%%
save([dirBusch name2save],'classification_accuracy_color',...
    'classification_accuracy_loc')

%%

dsize = cellfun(@size,all_data.Spikes,'UniformOutput',false);
time = [min(all_data.TimeInterval(:,1)):time_step:max([dsize{:}])-time_window+min(all_data.TimeInterval(:,1))]+100;

% figure
% plot(time,sq(nanmean(nanmean(classification_accuracy_loc,2),1)))
% hold on
% plot(time,sq(nanmean(nanmean(classification_accuracy_loc,2),1)))

figure
errorBarPlot(sq(nanmean(classification_accuracy_loc,2)),'area',1,'xaxisvalues',time)
hold on
errorBarPlot(sq(nanmean(classification_accuracy_color,2)),'area',1,'xaxisvalues',time,'color','r')
xlim([time(1),time(length(time))])
yline(0.5,'k:')
xline(0,'k--')
xlabel('Time [ms]')
ylabel('Classifier Accuracy')

%% 
% LUNDQVIST

% specific settings
loc_no = 3;
no_items = 3;
channel2use = 2; % 1-LIP, 2-PFC, 3-FEF

load([dirLund 'Lundqvist_sess1_spikes.mat'])
dsize = cellfun(@size,all_data.EpochedSpikes,'UniformOutput',false);
time = [min(all_data.TimeInterval(:,1)):time_step:max([dsize{:}])-time_window+min(all_data.TimeInterval(:,1))]+100;


classification_accuracy_color = nan(28,no_items,loc_no,length(time));
classification_accuracy_loc = nan(28,no_items,length(time));

% Load data
for sess = 1:28
    load([dirLund 'Lundqvist_sess' num2str(sess) '_spikes.mat'])
    
    % channels
    chan = all_data.UnitIndices{1} == channel2use;
    
    if sum(chan) > 1
    % generate labels
    order_sequence = all_data.Order123Loc';
    color_sequence = all_data.ColorLoc123';
    
    t_indices = all_data.StimulusSpecificIndices(:,[1,3,5]);
    % loop through locations
    for order = 1:no_items
        % get spikes
        
        %         spikes_peritem = repmat({NaN},length(t_indices),1);
        %         spikes_peritem(all_data.AttemptedTrials,1) = cellfun(@(x,y,z)...
        %             x(z,y(1):y(2)),...
        %             all_data.RawSpikes(all_data.AttemptedTrials), ...
        %             num2cell([t_indices(all_data.AttemptedTrials,order),...
        %             t_indices(all_data.AttemptedTrials,order)+600-1],2),...
        %             repmat({chan},sum(all_data.AttemptedTrials),1), 'UniformOutput', false);
        %
        %         spikes = nancat(3,spikes_peritem{:});
        
        spikes = nancat(3,all_data.EpochedSpikes{:});
        spikes = spikes(chan,:,:);
        spikes = permute(spikes,[3,1,2]);
        
        % convolve spikes with gaussian
        if smooth == 1 % smooth data using gaussian kernel
            K =  gauss(20,1); %(exp(-[-100:100].^2/10000)); % short gaussian
            spikes = nanconv( spikes, permute( K, [1,3,2] ) );
        end
        
        for loc = 1:loc_no
            color_lables = nan(1,length(color_sequence));
            color_lables(order_sequence(order,:)==loc) = color_sequence(loc,order_sequence(order,:)==loc);
            % recode 0 to nans
            color_lables(color_lables == 0) = NaN;
            
            % classify color per location
            classification_accuracy_color(sess,order,loc,:) = ...
                nanmean(simple_classification_overtime(spikes, color_lables,...
                iter, time_window, time_step),1);
        end
        % classify locations
        classification_accuracy_loc(sess,order,:) = ...
            nanmean(simple_classification_overtime(spikes,...
            order_sequence(order,:),iter, time_window, time_step),1);
        
    end
    end
end

name2save = 'Lundqvist_classification_over_time_PFC';


save([dirLund name2save],'classification_accuracy_color',...
    'classification_accuracy_loc')

%% Plot
dsize = cellfun(@size,all_data.EpochedSpikes,'UniformOutput',false);
time = [min(all_data.TimeInterval(:,1)):time_step:max([dsize{:}])-time_window+min(all_data.TimeInterval(:,1))]+100;


figure
% errorBarPlot(sq(nanmean(classification_accuracy_loc,2)),'area',1,'xaxisvalues',time)
errorBarPlot(permute(classification_accuracy_loc,[1,3,2]),'area',1,'xaxisvalues',time)
xlim([time(1),time(length(time))])
yline(0.33333,'k:')
xline(0,'k--')

figure
errorBarPlot(permute(sq(nanmean(classification_accuracy_color,3)),[1,3,2]),...
    'area',1,'xaxisvalues',time)
xlim([time(1),time(length(time))])
yline(0.5,'k:')
xline(0,'k--')


%% *******************************************************
%  *******************************************************
%  ********** 2. classification generalization ***********
%  *******************************************************
%  *******************************************************
%%

% BUSCHMAN
number_of_sets = repmat({5},1,28);
iter = repmat({5},1,28);
channel2use = [1 2 3];  % 1-LIP, 2-PFC, 3-FEF
normalize_data = 0;
normalization_order = [2,1];
baseline = 0;

load([dirBusch 'Buschman_data_all_sess'])
busch_labels = buschman_trial_sequence;

FR_all_buschman = cellfun(@(x) x',FR_all_buschman,'UniformOutput',false);

% normalize data
if normalize_data == 1
    FR_all_buschman = cellfun(@(x,y) normalize(normalize(x,y(1)),y(2)),...
        FR_all_buschman,repmat({normalization_order},1,28),...
        'UniformOutput',false);
end

if baseline == 1
    FR_all_buschman = cellfun(@(x,y) x-y,FR_all_buschman,...
        FR_baseline_buschman,'UniformOutput',false);
end

if length(channel2use)>1
    channels = cellfun(@(x) x{1},buschman_channels,'UniformOutput',false);
    FR_all_buschman = cellfun(@(x,y) x(:,any(y==channel2use,2)),FR_all_buschman,...
        channels,'UniformOutput',false);
else
    channels = cellfun(@(x) x{1},buschman_channels,'UniformOutput',false);
    FR_all_buschman = cellfun(@(x,y) x(:,y==channel2use),FR_all_buschman,...
        channels,'UniformOutput',false);
end

for loc = 1:6
    
    % ************************** LOCATION **************************
    current_labels = cellfun(@replacevalues, cellfun(@(x) x(loc,:), busch_labels, 'UniformOutput', false),...
        repmat({2},1,28),repmat({1},1,28), 'UniformOutput', false);
    class_matrix = cellfun(@classification_generalization,...
        FR_all_buschman, current_labels, number_of_sets,iter,...
        'UniformOutput',false);
    temp = cellfun(@(x) sq(nanmean(x,1)), class_matrix,'UniformOutput',false);
    busch_class_gen_location(loc,:,:,:) = permute(cat(3,temp{:}),[3,1,2]);
    
    % ************************** COLOR **************************
    current_labels = cellfun(@replacevalues, cellfun(@(x) x(loc,:), busch_labels, 'UniformOutput', false),...
         repmat({0},1,28),repmat({NaN},1,28), 'UniformOutput', false);
     class_matrix = cellfun(@classification_generalization,...
        FR_all_buschman, current_labels, number_of_sets,iter,...
        'UniformOutput',false);
    temp = cellfun(@(x) sq(nanmean(x,1)), class_matrix,'UniformOutput',false);
    busch_class_gen_color(loc,:,:,:) = permute(cat(3,temp{:}),[3,1,2]);
    
end

all_generalization_matrix_location = sq(nanmean(busch_class_gen_location,1));
all_generalization_matrix_color = sq(nanmean(busch_class_gen_color,1));

settings.number_of_sets = number_of_sets{1};
settings.iter = iter{1};
settings.channels = channel2use;
settings.normalize = normalize_data;
settings.baseline = baseline;

%% 
name2save = 'Buschman_crossgen_classification_all';

%%
save([dirBusch name2save],'busch_class_gen_location',...
    'busch_class_gen_color','settings')

%%

% PLOT THE RESULTS
% location
range1 = [.5,.55];
range2 = [.45,.65];
chance = .5;

mapname = 'Blues';
figure
subplot(3,1,1)
plotting_generalization_matrix_results(all_generalization_matrix_location,range2,mapname)
subplot(3,1,2)
plotting_generalization_lineplot_results(all_generalization_matrix_location,range2,chance)
subplot(3,1,3)
old = brewermap(10,mapname);
mapname = old(6:end,:);
plotting_generalization_ind_lineplot_results(all_generalization_matrix_location,range2,chance,mapname)
set(gcf, 'Position',  [100, 100, 500, 1500])

% color
figure
subplot(3,1,1)
mapname = 'Oranges';
plotting_generalization_matrix_results(all_generalization_matrix_color,range1,mapname)
subplot(3,1,2)
plotting_generalization_lineplot_results(all_generalization_matrix_color,range2,chance)
subplot(3,1,3)
old = brewermap(10,mapname);
mapname = old(6:end,:);
plotting_generalization_ind_lineplot_results(all_generalization_matrix_color,range2,chance,mapname)
set(gcf, 'Position',  [100, 100, 500, 1500])



%%
% LUNDQVIST

number_of_sets = repmat({5},1,28);
iter = repmat({5},1,28);
channel2use = 1;  % 1-LIP, 2-PFC, 3-FEF
normalize_data = 0;
normalization_order = [2,1];
baseline = 1;

load([dirLund 'Lundqvist_data_peritem_all_sess.mat'])
FR_all_lundqvist_peritem = cellfun(@(x) x',FR_all_lundqvist_peritem,...
    'UniformOutput',false);

if normalize_data == 1
    FR_all_lundqvist_peritem = cellfun(@(x,y) normalize(normalize(x,y(1)),y(2)),...
        FR_all_lundqvist_peritem,repmat({normalization_order},1,28),...
        'UniformOutput',false);
end

if baseline == 1
    FR_all_lundqvist_peritem = cellfun(@(x,y) x-y',FR_all_lundqvist_peritem,...
        FR_baseline_lundqvist_peritem,'UniformOutput',false);
end

if length(channel2use)>1
    channels = cellfun(@(x) x{1},lundqvist_channels,'UniformOutput',false);
    FR_all_lundqvist_peritem = cellfun(@(x,y) x(:,any(y==channel2use,2)),FR_all_lundqvist_peritem,...
        repmat(channels',1,3),'UniformOutput',false);
else
    channels = cellfun(@(x) x{1},lundqvist_channels,'UniformOutput',false);
    FR_all_lundqvist_peritem = cellfun(@(x,y) x(:,y==channel2use),FR_all_lundqvist_peritem,...
        repmat(channels',1,3),'UniformOutput',false);
end


% ************************** LOCATION **************************
lund_loc_at_order = cellfun(@(x) reshape(x',size(x,1)*size(x,2),1)', ...
    lundqvist_order_sequence, 'UniformOutput', false);
for sess = 1:28; FR_all_lundqvist_peritemord{sess} = ...
        cat(3,FR_all_lundqvist_peritem{sess,:}); end
FR_all_lundqvist_peritem_ordered = ...
    cellfun(@(x) reshape(permute(x,[3,1,2]),size(x,1)*size(x,3),size(x,2)), ...
    FR_all_lundqvist_peritemord, 'UniformOutput', false);

lund_class_gen_location = cellfun(@classification_generalization,...
    FR_all_lundqvist_peritem_ordered, lund_loc_at_order, number_of_sets,iter,...
    'UniformOutput',false);

% ************************** ORDER **************************
lund_order = cellfun(@(x) reshape((~isnan(x).*[1,2,3])',size(x,1)*size(x,2),1)', ...
    lundqvist_order_sequence, 'UniformOutput', false);
lund_order = cellfun(@replacevalues, lund_order,...
    repmat({0},1,28),repmat({NaN},1,28), 'UniformOutput', false);
lund_class_gen_order = cellfun(@classification_generalization,...
    FR_all_lundqvist_peritem_ordered, lund_order, number_of_sets,iter,...
    'UniformOutput',false);

% ************************** COLOR **************************
% rearrange FR to represent location 1 2 3 (originally represents order 1 2 3)
for sess = 1:28; for loc = 1:3; order = sum((lundqvist_order_sequence{sess}==loc).*[1,2,3],2);
        FR_all_lundqvist_bylocation{sess,loc} = nan(size(FR_all_lundqvist_peritem{sess,loc}));
        FR_all_lundqvist_bylocation{sess,loc}(order==1,:) = ...
            FR_all_lundqvist_peritemord{sess}(order==1,:,1);
        FR_all_lundqvist_bylocation{sess,loc}(order==2,:) = ...
            FR_all_lundqvist_peritemord{sess}(order==2,:,2);
        FR_all_lundqvist_bylocation{sess,loc}(order==3,:) = ...
            FR_all_lundqvist_peritemord{sess}(order==3,:,3); end; end

for loc = 1:3
    current_labels = cellfun(@replacevalues, cellfun(@(x) x(loc,:), lundqvist_trial_sequence, 'UniformOutput', false),...
        repmat({0},1,28),repmat({NaN},1,28), 'UniformOutput', false);
    current_FR =  {FR_all_lundqvist_bylocation{:,loc}};
    temp = cellfun(@classification_generalization,...
        current_FR, current_labels, number_of_sets,iter,...
        'UniformOutput',false);
    lund_class_gen_color(loc,:,:,:) = permute(sq(nanmean(cat(4,temp{:}),1)),[3,1,2]);
end

% **************** LOCATION-ORDER CONTROLLED *******************
for order = 1:3
    current_labels = cellfun(@(x) x(:,order)',...
        lundqvist_order_sequence, 'UniformOutput', false);
    current_FR =  {FR_all_lundqvist_peritem{:,order}};
    temp = cellfun(@classification_generalization,...
        current_FR, current_labels, number_of_sets,iter,...
        'UniformOutput',false);
    lund_class_gen_location_order(order,:,:,:) = permute(sq(nanmean(cat(4,temp{:}),1)),[3,1,2]);
end

% *********** COLOR-LOCATION AND ORDER CONTROLLED *************
current_sess_labels = cell(28,3,3);
current_sess_FR = cell(28,3,3);
for sess = 1:28
    current_sess_labels(sess,:,:) = repmat({nan(size(current_labels{sess}))},3,3);
    current_sess_FR(sess,:,:) = repmat({nan(size(FR_all_lundqvist_bylocation{sess,1}))},3,3);
    for order = 1:3
        current_locations = cellfun(@(x) x(:,order)',...
            lundqvist_order_sequence, 'UniformOutput', false);
        for loc = 1:3
            current_sess_labels{sess,order,loc}(current_locations{sess} == loc) = ...
                lundqvist_trial_sequence{sess}(loc,current_locations{sess} == loc);
            
            current_sess_FR{sess,order,loc}(current_locations{sess} == loc,:) =  ...
                FR_all_lundqvist_bylocation...
                {sess,loc}(current_locations{sess} == loc,:);
        end
    end
end

temp = cellfun(@(x) sq(nanmean(x,1)), cellfun(@classification_generalization,...
    current_sess_FR, current_sess_labels, repmat(number_of_sets',1,3,3),...
    repmat(iter',1,3,3),...
    'UniformOutput',false), 'UniformOutput',false);

lund_class_gen_color_location_order = ...
    cat(4,reshape(cat(3,temp{:,:,1}),5,5,28,3),...
    reshape(cat(3,temp{:,:,2}),5,5,28,3),...
    reshape(cat(3,temp{:,:,3}),5,5,28,3));


all_generalization_matrix_location = permute(sq(nanmean...
    (cat(4,lund_class_gen_location{:}),1)),[3,1,2]);
all_generalization_matrix_order = permute(sq(nanmean...
    (cat(4,lund_class_gen_order{:}),1)),[3,1,2]);
all_generalization_matrix_color = sq(nanmean(lund_class_gen_color,1));
all_generalization_matrix_location_order = sq(nanmean...
    (lund_class_gen_location_order,1));
all_generalization_matrix_color_location_order = permute(nanmean(...
    lund_class_gen_color_location_order,4),[3,1,2]);


settings.number_of_sets = number_of_sets{1};
settings.iter = iter{1};
settings.channels = channel2use;
settings.normalize = normalize_data;
settings.baseline = baseline;

%% 
name2save = 'Lundqvist_crossgen_classification_PFC_baselined';

%%
save([dirLund name2save],...
    'lund_class_gen_location',...
    'lund_class_gen_order',...
    'lund_class_gen_color',...
    'lund_class_gen_location_order',...
    'lund_class_gen_color_location_order','settings')

%%
for i = 1:5
    if i == 1
        mapname = 'Greens';
        current = all_generalization_matrix_order;
        range1 = [.33,.55];
        range2 = [.3,.55];
        chance = .333;
    elseif i == 2
        mapname = 'Blues';
        current = all_generalization_matrix_location;
        range1 = [.33,.55];
        range2 = [.3,.55];
        chance = .333;   
    elseif i == 3
        mapname = 'Purples';
        current = all_generalization_matrix_location_order;
        range1 = [.33,.55];
        range2 = [.3,.55];
        chance = .333;
    elseif i == 4
        mapname = 'Reds';
        current = all_generalization_matrix_color;
        range1 = [.5,.6];
        range2 = [.45,.6];
        chance = .5;
    elseif i == 5
        mapname = 'Oranges';
        current = all_generalization_matrix_color_location_order;
        range1 = [.5,.6];
        range2 = [.45,.6];
        chance = .5;
        
    end
    
    
    figure
    subplot(3,1,1);
    plotting_generalization_matrix_results(current,range1,mapname)
    subplot(3,1,2)
    plotting_generalization_lineplot_results(current,range2,chance)
    subplot(3,1,3)
    
    old = brewermap(10,mapname);
    mapname = old(6:end,:);
    plotting_generalization_ind_lineplot_results(current,range2,chance,mapname)
    set(gcf, 'Position',  [100, 100, 500, 1500])

end


 
%% *******************************************************
%  *******************************************************
%  *** 3. Window trained classification generalization ***
%  *******************************************************
%  *******************************************************

%%

% BUSCHMAN
number_of_sets = repmat({5},1,28);
iter = repmat({5},1,28);
channel2use = 3;  % 1-LIP, 2-PFC, 3-FEF
normalize_data = 0;
normalization_order = [2,1];
baseline = 0;
numoftrialtotrainon = repmat({2},1,28);%repmat({100},1,28);


load([dirBusch 'Buschman_data_all_sess'])
busch_labels = buschman_trial_sequence;

FR_all_buschman = cellfun(@(x) x',FR_all_buschman,'UniformOutput',false);

% normalize data
if normalize_data == 1
    FR_all_buschman = cellfun(@(x,y) normalize(normalize(x,y(1)),y(2)),...
        FR_all_buschman,repmat({normalization_order},1,28),...
        'UniformOutput',false);
end

if baseline == 1
    FR_all_buschman = cellfun(@(x,y) x-y,FR_all_buschman,...
        FR_baseline_buschman,'UniformOutput',false);
end

if length(channel2use)>1
    channels = cellfun(@(x) x{1},buschman_channels,'UniformOutput',false);
    FR_all_buschman = cellfun(@(x,y) x(:,any(y==channel2use,2)),FR_all_buschman,...
        channels,'UniformOutput',false);
else
    channels = cellfun(@(x) x{1},buschman_channels,'UniformOutput',false);
    FR_all_buschman = cellfun(@(x,y) x(:,y==channel2use),FR_all_buschman,...
        channels,'UniformOutput',false);
end


for loc = 1:6   
    % ************************** LOCATION **************************
    current_labels = cellfun(@replacevalues, cellfun(@(x) x(loc,:), busch_labels, 'UniformOutput', false),...
        repmat({2},1,28),repmat({1},1,28), 'UniformOutput', false); 
    [accuracy_loc(loc,:),window_sizes_loc(loc,:)] = cellfun(@train_on_moving_subset_test_on_the_rest,...
        FR_all_buschman, current_labels, numoftrialtotrainon,...
        'UniformOutput',false);
         
    % ************************** COLOR **************************
    current_labels = cellfun(@replacevalues, cellfun(@(x) x(loc,:), busch_labels, 'UniformOutput', false),...
         repmat({0},1,28),repmat({100},1,28), 'UniformOutput', false);
     [accuracy_color(loc,:),window_sizes_color(loc,:)] = cellfun(@train_on_moving_subset_test_on_the_rest,...
         FR_all_buschman, current_labels, numoftrialtotrainon,...
         'UniformOutput',false);   
end


% test what window size is the minimum to achieve above chance
% classification performance
counter = 0;
for trials2trainon = 2%[5,10,15]
    counter = counter+1;
    numoftrialtotrainon = repmat({trials2trainon},1,28);
    for loc = 1:6
        % ************************** LOCATION **************************
        current_labels = cellfun(@replacevalues, cellfun(@(x) x(loc,:), busch_labels, 'UniformOutput', false),...
            repmat({2},1,28),repmat({1},1,28), 'UniformOutput', false);
        [accuracy_loc(loc,:),window_sizes_loc(loc,:)] = cellfun(@train_on_moving_subset_test_on_the_rest,...
            FR_all_buschman, current_labels, numoftrialtotrainon,...
            'UniformOutput',false);
        
        % ************************** COLOR **************************
        current_labels = cellfun(@replacevalues, cellfun(@(x) x(loc,:), busch_labels, 'UniformOutput', false),...
            repmat({0},1,28),repmat({100},1,28), 'UniformOutput', false);
        [accuracy_color(loc,:),window_sizes_color(loc,:)] = cellfun(@train_on_moving_subset_test_on_the_rest,...
            FR_all_buschman, current_labels, numoftrialtotrainon,...
            'UniformOutput',false);
    end
   accuracy_loc_all{counter} = nancat(3,accuracy_loc{:});
   accuracy_col_all{counter} = nancat(3,accuracy_color{:});
end

%% 
name2save = 'Buschman_windowclass_FEF_2tr';
settings.number_of_sets = number_of_sets{1};
settings.iter = iter{1};
settings.channels = channel2use;
settings.normalize = normalize_data;
settings.baseline = baseline;


%%
save([dirBusch name2save],'accuracy_loc',...
    'accuracy_color','settings')

%% 
d2use = accuracy_loc;
min_trials = cellfun(@(x) min(size(x)), d2use, 'UniformOutput', false);
mininum = min([min_trials{:}])/2;

cropped_acc = cellfun(@(x,y) x(1:y,1:y),d2use,repmat({mininum},6,28), 'UniformOutput', false);
avg_data = nanmean(cat(4,cat(3,cropped_acc{1,:}),cat(3,cropped_acc{3,:}),cat(3,cropped_acc{3,:}),...
     cat(3,cropped_acc{4,:}),cat(3,cropped_acc{5,:}),cat(3,cropped_acc{6,:})),4);

% ind = cellfun(@isnan, mat2cell(temp,ones(1,28),[1337]), 'UniformOutput', false);
% nanremoved = cellfun(@(x,y) x(~y), mat2cell(temp,ones(1,28),[1337]),ind, 'UniformOutput', false);
% smoothed = cellfun(@movmean, nanremoved,repmat({100},28,1), 'UniformOutput', false);
% d2plot = nancat(1,smoothed{:});

% d2plot = sq(nanmean(avg_data,1))';
% figure
% errorBarPlot(d2plot,'area',1)
% figure
% imagesc(d2plot)
%%
d2plot = sq(nanmean(avg_data,1))';
a = figure
a.Position = [ 54   193   962   370];
subplot(1,2,1)
errorBarPlot(d2plot,'area',1)
xlabel('Distance from training set [trials]')
ylabel('Classifier Accuracy')
hold on
plot(movmean(nanmean(d2plot,1),100),'LineWidth',3)
xlim([1,length(d2plot)])
subplot(1,2,2)
errorBarPlot(d2plot,'area',1)
xlabel('Distance from training set [trials]')
ylabel('Classifier Accuracy')
hold on
plot(movmean(nanmean(d2plot,1),100),'LineWidth',3)
xlim([1,50])


d2plot = sq(nanmean(avg_data,1))';
a = figure
a.Position = [ 54   193   962   370];
subplot(1,2,1)
errorBarPlot(d2plot,'area',1,'color','r')
xlabel('Distance from training set [trials]')
ylabel('Classifier Accuracy')
hold on
plot(movmean(nanmean(d2plot,1),100),'LineWidth',3)
xlim([1,length(d2plot)])
subplot(1,2,2)
errorBarPlot(d2plot,'area',1,'color','r')
xlabel('Distance from training set [trials]')
ylabel('Classifier Accuracy')
hold on
plot(movmean(nanmean(d2plot,1),100),'LineWidth',3)
xlim([1,50])


%% 
% Lundqvist
    
number_of_sets = repmat({5},1,28);
iter = repmat({5},1,28);
channel2use = 1;  % 1-LIP, 2-PFC, 3-FEF
normalize_data = 0;
normalization_order = [2,1];
baseline = 0;
numoftrialtotrainon = repmat({100},1,28);

load([dirLund 'Lundqvist_data_peritem_all_sess.mat'])
FR_all_lundqvist_peritem = cellfun(@(x) x',FR_all_lundqvist_peritem,...
    'UniformOutput',false);

if normalize_data == 1
    FR_all_lundqvist_peritem = cellfun(@(x,y) normalize(normalize(x,y(1)),y(2)),...
        FR_all_lundqvist_peritem,repmat({normalization_order},1,28),...
        'UniformOutput',false);
end

if baseline == 1
    FR_all_lundqvist_peritem = cellfun(@(x,y) x-y',FR_all_lundqvist_peritem,...
        FR_baseline_lundqvist_peritem,'UniformOutput',false);
end

if length(channel2use)>1
    channels = cellfun(@(x) x{1},lundqvist_channels,'UniformOutput',false);
    FR_all_lundqvist_peritem = cellfun(@(x,y) x(:,any(y==channel2use,2)),FR_all_lundqvist_peritem,...
        repmat(channels',1,3),'UniformOutput',false);
else
    channels = cellfun(@(x) x{1},lundqvist_channels,'UniformOutput',false);
    FR_all_lundqvist_peritem = cellfun(@(x,y) x(:,y==channel2use),FR_all_lundqvist_peritem,...
        repmat(channels',1,3),'UniformOutput',false);
end


% ************************** LOCATION **************************
lund_loc_at_order = cellfun(@(x) reshape(x',size(x,1)*size(x,2),1)', ...
    lundqvist_order_sequence, 'UniformOutput', false);
for sess = 1:28; FR_all_lundqvist_peritemord{sess} = ...
        cat(3,FR_all_lundqvist_peritem{sess,:}); end
FR_all_lundqvist_peritem_ordered = ...
    cellfun(@(x) reshape(permute(x,[3,1,2]),size(x,1)*size(x,3),size(x,2)), ...
    FR_all_lundqvist_peritemord, 'UniformOutput', false);
  
[accuracy_loc,window_sizes_loc] = cellfun(@train_on_moving_subset_test_on_the_rest,...
    FR_all_lundqvist_peritem_ordered, lund_loc_at_order, numoftrialtotrainon,...
    'UniformOutput',false);
    

% ************************** ORDER **************************
lund_order = cellfun(@(x) reshape((~isnan(x).*[1,2,3])',size(x,1)*size(x,2),1)', ...
    lundqvist_order_sequence, 'UniformOutput', false);
lund_order = cellfun(@replacevalues, lund_order,...
    repmat({0},1,28),repmat({NaN},1,28), 'UniformOutput', false);
[accuracy_order,window_sizes_order] = cellfun(@train_on_moving_subset_test_on_the_rest,...
    FR_all_lundqvist_peritem_ordered, lund_order, numoftrialtotrainon,...
    'UniformOutput',false);


% ************************** COLOR **************************
% rearrange FR to represent location 1 2 3 (originally represents order 1 2 3)
for sess = 1:28; for loc = 1:3; order = sum((lundqvist_order_sequence{sess}==loc).*[1,2,3],2);
        FR_all_lundqvist_bylocation{sess,loc} = nan(size(FR_all_lundqvist_peritem{sess,loc}));
        FR_all_lundqvist_bylocation{sess,loc}(order==1,:) = ...
            FR_all_lundqvist_peritemord{sess}(order==1,:,1);
        FR_all_lundqvist_bylocation{sess,loc}(order==2,:) = ...
            FR_all_lundqvist_peritemord{sess}(order==2,:,2);
        FR_all_lundqvist_bylocation{sess,loc}(order==3,:) = ...
            FR_all_lundqvist_peritemord{sess}(order==3,:,3); end; end

lund_class_gen_color = [];
for loc = 1:3
    current_labels = cellfun(@replacevalues, cellfun(@(x) x(loc,:), lundqvist_trial_sequence, 'UniformOutput', false),...
        repmat({0},1,28),repmat({NaN},1,28), 'UniformOutput', false);
    current_FR =  {FR_all_lundqvist_bylocation{:,loc}};
    [accuracy_temp,window_sizes_temp] = cellfun(@train_on_moving_subset_test_on_the_rest,...
    current_FR, current_labels, numoftrialtotrainon,...
    'UniformOutput',false);

    lund_class_gen_color = nancat(4,lund_class_gen_color,permute(nancat(3,accuracy_temp{:}),[3,1,2]));
end

% **************** LOCATION-ORDER CONTROLLED *******************
lund_class_gen_location_order = [];
for order = 1:3
    current_labels = cellfun(@(x) x(:,order)',...
        lundqvist_order_sequence, 'UniformOutput', false);
    current_FR =  {FR_all_lundqvist_peritem{:,order}};
 
    [accuracy_temp,window_sizes_temp] = cellfun(@train_on_moving_subset_test_on_the_rest,...
    current_FR, current_labels, numoftrialtotrainon,...
    'UniformOutput',false);

    lund_class_gen_location_order = nancat(4,lund_class_gen_location_order,permute(nancat(3,accuracy_temp{:}),[3,1,2]));
end

% *********** COLOR-LOCATION AND ORDER CONTROLLED *************
current_sess_labels = cell(28,3,3);
current_sess_FR = cell(28,3,3);
for sess = 1:28
    current_sess_labels(sess,:,:) = repmat({nan(size(current_labels{sess}))},3,3);
    current_sess_FR(sess,:,:) = repmat({nan(size(FR_all_lundqvist_bylocation{sess,1}))},3,3);
    for order = 1:3
        current_locations = cellfun(@(x) x(:,order)',...
            lundqvist_order_sequence, 'UniformOutput', false);
        for loc = 1:3
            current_sess_labels{sess,order,loc}(current_locations{sess} == loc) = ...
                lundqvist_trial_sequence{sess}(loc,current_locations{sess} == loc);
            
            current_sess_FR{sess,order,loc}(current_locations{sess} == loc,:) =  ...
                FR_all_lundqvist_bylocation...
                {sess,loc}(current_locations{sess} == loc,:);
        end
    end
end


temp = cellfun(@train_on_moving_subset_test_on_the_rest,...
    current_sess_FR, current_sess_labels, repmat(numoftrialtotrainon',1,3,3),...
    'UniformOutput',false);

lund_class_gen_color_location_order = ...
    nancat(4,nancat(3,temp{:,:,1}),...
    nancat(3,temp{:,:,2}),...
    nancat(3,temp{:,:,3}));



%%    
% plot
d2use = accuracy_loc;

% remove sessions with no results (due to lack of suitable units)
s2remove = cellfun(@isempty, d2use);
d2use(s2remove) = [];
min_trials = cellfun(@(x) min(size(x)), d2use, 'UniformOutput', false);
mininum = round(min([min_trials{:}])/2);

cropped_acc = cellfun(@(x,y) x(1:y,1:y),d2use,repmat({mininum},size(d2use)), 'UniformOutput', false);
avg_data = cat(3,cropped_acc{:});

% fit regression line
x = [ones(1,size(avg_data,1));1:size(avg_data,1)];
y = sq(nanmean(nanmean(avg_data,1),3));
b = x'\y;

d2plot = sq(nanmean(avg_data,1))';
a = figure
a.Position = [ 54   193   962   370];
subplot(1,2,1)
errorBarPlot(d2plot,'area',1)
xlabel('Distance from training set [trials]')
ylabel('Classifier Accuracy')
yline(0.5,'k:','LineWidth',2)
hold on
plot(movmean(nanmean(d2plot,1),100),'LineWidth',3)

yCalc1 = b(1)+b(2)*x(2,:);
hold on
plot(1:length(x),yCalc1,'LineWidth',3)

xlim([1,length(d2plot)])
subplot(1,2,2)
errorBarPlot(d2plot,'area',1)
hold on
yline(0.5,'k:','LineWidth',2)
xlabel('Distance from training set [trials]')
ylabel('Classifier Accuracy')
hold on
plot(movmean(nanmean(d2plot,1),50),'LineWidth',3)
plot(1:length(x),yCalc1,'LineWidth',3)
xlim([1,50])

d2use = accuracy_color;

% remove sessions with no results (due to lack of suitable units)
s2remove = cellfun(@isempty, d2use);
d2use(s2remove) = [];
min_trials = cellfun(@(x) min(size(x)), d2use, 'UniformOutput', false);
mininum = round(min([min_trials{:}])/2);

cropped_acc = cellfun(@(x,y) x(1:y,1:y),d2use,repmat({mininum},size(d2use)), 'UniformOutput', false);
avg_data = cat(3,cropped_acc{:});

% fit regression line
x = [ones(1,size(avg_data,1));1:size(avg_data,1)];
y = sq(nanmean(nanmean(avg_data,1),3));
b = x'\y;

d2plot = sq(nanmean(avg_data,1))';
a = figure
a.Position = [ 54   193   962   370];
subplot(1,2,1)
errorBarPlot(d2plot,'area',1,'color','r')
yline(0.5,'k:','LineWidth',2)
xlabel('Distance from training set [trials]')
ylabel('Classifier Accuracy')
hold on
plot(movmean(nanmean(d2plot,1),100),'LineWidth',3)

yCalc1 = b(1)+b(2)*x(2,:);
hold on
plot(1:length(x),yCalc1,'LineWidth',3)

xlim([1,length(d2plot)])
subplot(1,2,2)
errorBarPlot(d2plot,'area',1,'color','r')
hold on
yline(0.5,'k:','LineWidth',2)
xlabel('Distance from training set [trials]')
ylabel('Classifier Accuracy')
hold on
plot(movmean(nanmean(d2plot,1),50),'LineWidth',3)
plot(1:length(x),yCalc1,'LineWidth',3)
xlim([1,50])

