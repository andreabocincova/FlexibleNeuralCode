function  [accuracy,window_sizes] = train_on_moving_subset_test_on_the_rest....
    (FR,labels,numoftrialtotrainon)
% classification over a subset of moving set of trials of equal size
%
% INPUT:
% FR                    - matrix of firing rates (trial x neuron)
% labels                - vector with labels for classification
% numoftrialtotrainon   - size of time window expressed in terms of number
%                         of trials used to train the classifier
%
% OUTPUT
% accuracy              - cross-validated classifier accuracy for each
%                         iteration (iter)
% window_sizes          - in order to keep the no of trails used to train
%                         the classifier the same (i.e., nunoftrialtotrainon)
%                         window sizes vary, this variable show this
%                         variability

% remove all nans before splitting trials into sets
toremove1 = isnan(labels);
toremove2 = sum(isnan(FR),2)==size(FR,2);

toremove = find(any([toremove1;toremove2']));

nonnan_FR = FR;
nonnan_FR(toremove,:) = [];

% remove neurons with any nans
toremove3 = isnan(mean(nonnan_FR,1));
nonnan_FR(:,toremove3) = [];


nonnan_labels = labels;
nonnan_labels(toremove) = [];

nonnan_labels(nonnan_labels==100) = NaN;

% preset variables
window_sizes = nan(1,size(nonnan_FR,1)-numoftrialtotrainon-1);
accuracy = nan(length(window_sizes),length(window_sizes));

% do not run analysis if no neurons are available
if ~isempty(nonnan_FR)
    
% move trough the data set step by step training on subset of trials and
% testing on the rest
for step = 1:size(nonnan_FR,1)-numoftrialtotrainon-1
    
    window_size = step+numoftrialtotrainon-1;
    enough_trials = 0;
    stop_classification = 0;
    % stretch window size until it contains enough trials
    while enough_trials == 0
        window_labels = nonnan_labels(step:window_size);
        
        groups = unique(window_labels);
        groups(isnan(groups)) = [];
        trial_no = [];
        for i = 1:length(groups)
            trial_no(i) = sum(window_labels==groups(i));
        end
        
        if all(trial_no-numoftrialtotrainon > 0) && ~isempty(trial_no)
            enough_trials = 1;
        elseif window_size > size(nonnan_FR,1)-numoftrialtotrainon-1
            enough_trials = 1;
            stop_classification = 1;
        else
            window_size = window_size+1;
        end
    end
    
    if ~stop_classification
        % define training data
        traindata = nonnan_FR(step:window_size,:);
        trainlabels = nonnan_labels(step:window_size);
        
        % remove these trials from test data
        testdata = nonnan_FR;
        testdata(1:window_size,:) = [];
        testlabels = nonnan_labels;
        testlabels(1:window_size) = [];
        
        % store the indices of the test data
        indices = 1:length(nonnan_labels);
        indices(step:window_size) = [];
        tested_indices = indices;
        
        % remove zero trials (color classification only)
        tested_indices(isnan(testlabels)) = [];
        testdata(isnan(testlabels),:) = [];
        testlabels(isnan(testlabels)) = [];
        
        % equalize condition trial numbers
        indices_for_equal_cond_proportions = equalize_conditions(trainlabels);
        
        if length(indices_for_equal_cond_proportions) > length(groups)
            
            final_train_data = traindata(indices_for_equal_cond_proportions,:);
            final_train_labels = trainlabels(indices_for_equal_cond_proportions);
            
            % train classifier
            DISCR = fitcdiscr(final_train_data,final_train_labels,'discrimType','pseudoLinear');
            
            % test on all other trials
            accurate = predict(DISCR, testdata) == testlabels';
            accuracy(step,1:size(testlabels,2)) = accurate;
            
            window_sizes(step) = window_size-step;
        end
    end
end

end

end