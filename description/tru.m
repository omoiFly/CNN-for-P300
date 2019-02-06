dataset = data();
cv = cvpartition(size(dataset,1),'holdout',0.04);

training_set = dataset(training(cv), :);
testing_set = dataset(test(cv), :);

training_label = training_set(:, 1);
training_data = training_set(:, 2:5);

testing_label = testing_set(:, 1);
testing_data = testing_set(:, 2:5);



