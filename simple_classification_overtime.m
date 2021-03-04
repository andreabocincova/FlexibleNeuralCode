function classification_accuracy = simple_classification_overtime(spikes, labels,...
    iter, time_window, time_step)
% classification over time
% calculates FR over time_window and moves the classification window by
% time_step ammount
%
% INPUT:
% spikes        - matrix of spikes (trial x neuron x time[ms])
% labels        - vector with labels for classification
% iter          - number of iterations for splitting labels into equal
%                 parts (i.e., equal number of trials per condition)
% time_window   - size of time window to use to calculate FR for
%                 classification [in ms]
% time_step     - size of classification steps [in ms]
%
% OUTPUT
% classification_accuracy - cross-validated classifier accuracy for each
%                           iteration (iter) and each time point 

% remove all nan trials (aborted) before splitting trials into sets
toremove1 = isnan(labels);
toremove2 = sum(isnan(sq(nanmean(spikes,3))),2)==size(spikes,2);

toremove = find(any([toremove1;toremove2']));

nonan_spikes = spikes;
nonan_spikes(toremove,:,:) = [];
nonnan_labels = labels;
nonnan_labels(toremove) = [];

% calculate time indices based on time step size
time_ind = 1:time_step:size(spikes,3)-time_window;

%% initialize
classification_accuracy = nan(iter,length(time_ind));

% loop through iterations
for iteration = 1:iter
    % equalize condition trial numbers
    indices_for_equal_cond_proportions = equalize_conditions(nonnan_labels);
    data = nonan_spikes(indices_for_equal_cond_proportions,:,:);
    labels = nonnan_labels(indices_for_equal_cond_proportions);
    
    % remove neurons that have no spikes recorded (NaN)
    remove_neurons = isnan(sq(mean(nanmean(data,3),1)));
    data(:,remove_neurons,:) = [];
    
    classification_t = nan(1,length(time_ind));
    % loop over time
    for t = 1:length(time_ind)
        % calculate FR over time window of interset
        current_data = nanmean(data(:,:,time_ind(t):time_ind(t)+time_window-1),3);
        
        if size(current_data,2) > length(unique(labels))*2
            
            % calculate within session section classification accuracy
            train_partitions = cvpartition(length(labels)/length(unique(labels)),'LeaveOut');
            accuracy = [];
            
            % cross-validation loop
            parfor leaveout = 1:size(current_data,1)/length(unique(labels)) % holdout = 1:10 %
                
                trn_ind = training(train_partitions,leaveout); % get training trial rows
                trn_ind = repmat(trn_ind,length(unique(labels)),1);
                tst_ind = test(train_partitions,leaveout); % get test trial rows
                tst_ind = repmat(tst_ind,length(unique(labels)),1);
                
                
                trn_dat = current_data(trn_ind,:); % isolate training data
                tst_dat = current_data(tst_ind,:); % isolate test data
                trn_labels = labels(trn_ind);
                tst_labels = labels(tst_ind);
                
                % fit  
                DISCR = fitcdiscr(trn_dat,trn_labels,'discrimType','pseudoLinear');
                % calculate accuracy
                accuracy(leaveout) = mean(predict(DISCR, tst_dat) == tst_labels');
                
            end % leavout
            
            classification_t(t) = nanmean(accuracy);
        end % size check
    end % t
    classification_accuracy(iteration,:) = classification_t;
    
end
