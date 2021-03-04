function full_generalization_matrix = classification_generalization...
    (FR, labels, number_of_sets, iter)
% train on one set(block of consequtive trials) of trials and test on a 
% trial from another set (similar to train on one time
% point but testing on another one)
%
% INPUT:
% FR             - matrix of Firing Rates (trial x neuron)
% labels         - vector with labels for classification
% iter           - number of iterations for splitting labels into equal
%                 parts (i.e., equal number of trials per condition)
% number_of_sets - number of blocks the dataset should be split into
%
%
% OUTPUT
% full_generalization_matrix - cross-validated classifier accuracy matrix 
%                              for each iteration (iter) and all
%                              combinations of training and testing blocks

% remove all nans before splitting trials into sets
toremove1 = isnan(labels);
toremove2 = sum(isnan(FR),2)==size(FR,2);

toremove = find(any([toremove1;toremove2']));

nonan_FR = FR;
nonan_FR(toremove,:) = [];
nonnan_labels = labels;
nonnan_labels(toremove) = [];

% e.g., split trials into 5 groups (early, early middle, middle, late middle,
% late)
no_of_trials = floor(size(nonan_FR,1)/number_of_sets);

% reshape FR and labels to sets
reshaped_FR = reshape(nonan_FR(1:no_of_trials*number_of_sets,:),...
    [no_of_trials,number_of_sets,size(nonan_FR,2)]);
reshaped_labels = reshape(nonnan_labels(1:no_of_trials*number_of_sets),...
    [no_of_trials,number_of_sets]);

%%
full_generalization_matrix = nan(iter,number_of_sets,number_of_sets);

if ~isempty(reshaped_FR)
    
parfor iteration = 1:iter
    % equalize condition trial numbers
    indices_for_equal_cond_proportions = equalize_conditions(reshaped_labels);
    %%
    subsampled_data = [];
    subsampled_labels = [];
    for set = 1:number_of_sets
        subsampled_data(set,:,:) = sq(reshaped_FR(indices_for_equal_cond_proportions(:,set),set,:));
        subsampled_labels(set,:) = reshaped_labels(indices_for_equal_cond_proportions(:,set),set);
    end
    
    remove_neurons = sum(isnan(sq(mean(subsampled_data,2))),1) > 0;
    subsampled_data(:,:,remove_neurons) = [];
    
    generalization_matrix = nan(number_of_sets,number_of_sets);
    if size(subsampled_data,2) > length(unique(subsampled_labels))*2 && ...
            size(subsampled_data,3)>1
        for training_group = 1:number_of_sets
            
            data = sq(subsampled_data(training_group,:,:));
            labels = subsampled_labels(training_group,:);
            
            % calculate within session section classification accuracy
            train_partitions = cvpartition(length(labels)/length(unique(labels)),'LeaveOut');
            
            correct_within = [];
            correct_between = [];
            for leaveout = 1:size(data,1)/length(unique(labels)) % holdout = 1:10 %
                %                 train_partitions = cvpartition(length(labels),'HoldOut',round(length(labels)/10));
                trn_ind = training(train_partitions,leaveout); % get training trial rows
                trn_ind = repmat(trn_ind,length(unique(labels)),1);
                %                 trn_ind = training(train_partitions,1); % get training trial rows
                tst_ind = test(train_partitions,leaveout); % get test trial rows
                tst_ind = repmat(tst_ind,length(unique(labels)),1);
                %                 tst_ind = test(train_partitions,1); % get test trial rows
                trn_dat = data(trn_ind,:); % isolate training data
                tst_dat = data(tst_ind,:); % isolate test data
                trn_labels = labels(trn_ind);
                tst_labels = labels(tst_ind);
                DISCR = fitcdiscr(trn_dat,trn_labels,'discrimType','pseudoLinear');
                correct_within(leaveout) = mean(predict(DISCR, tst_dat) == tst_labels');
                %                 correct_within(holdout) = mean(predict(DISCR, tst_dat) == tst_labels');
                
                % calculate between session section classification accuracy
                testing_groups = 1:number_of_sets;
                testing_groups(training_group) = [];
                
                for testing_group = 1:number_of_sets-1
                    test_data = sq(subsampled_data(testing_groups(testing_group),:,:));
                    test_labels =  subsampled_labels(testing_groups(testing_group),:)';
                    
                    %                 test_data = sq(reshaped_FR(:,testing_groups(testing_group),:));
                    %                 test_labels =  reshaped_labels(:,testing_groups(testing_group));
                    
                    %                 remove_trials = isnan(test_labels);
                    %                 test_data(remove_trials,:) = [];
                    %                 test_data(:,remove_neurons) = [];
                    %                 test_labels(remove_trials,:) = [];
                    
                    correct_between(leaveout,testing_group) = mean(predict(DISCR, test_data) == test_labels);
                    %                     correct_between(holdout,testing_group) = mean(predict(DISCR, test_data) == test_labels);
                end
            end
            
            generalization_matrix(training_group,training_group) = mean(correct_within);
            generalization_matrix(training_group,testing_groups) = mean(correct_between,1);
        end
    end
    full_generalization_matrix(iteration,:,:) = generalization_matrix;
    
end
end
end
