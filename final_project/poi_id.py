#!/usr/bin/python
# coding=utf-8
import sys
import pickle
from pprint import pprint
import numpy as np
from matplotlib import pyplot
from sklearn import metrics
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, chi2, RFE

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
	data_dict = pickle.load(data_file)


def pre_process():
	pass


def choose_best_est(features, labels, scoring):
    list_elems = [
        {
            'est': SVC(), 
            'params': {
                'C': range(1, 10), 'kernel': ('rbf', 'linear', 'poly', 'sigmoid')
            }
        }, 
        {
            'est': DecisionTreeClassifier(), 
            'params': {
                'max_depth': range(5, 15), 'min_samples_leaf': range(1, 5), 
            }
        },
        {
            'est': KNeighborsClassifier(),
            'params': {
                'n_neighbors': range(1, 10), 'weights': ('uniform', 'distance'), 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')
            }
        }, 
        {
            'est': RandomForestClassifier(), 
            'params': {
                'n_estimators': range(5, 15), 'min_samples_split': range(2, 5), 'max_depth': range(5, 15), 'min_samples_leaf': range(1, 5)
            }
        }
    ]

    clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=1)
    dict_params = {'max_depth': range(5, 15), 'min_samples_leaf': range(1, 5)}

    list_grids = []
    for elem in list_elems:
        grid = GridSearchCV(estimator=elem['est'], param_grid=elem['params'], scoring=scoring)
        grid.fit(features, labels)
        list_grids.append(grid)

    sorted(list_grids, key=lambda grid: grid.best_score_, reverse=True)
    grid = list_grids[0]
    # clf = grid.best_estimator_
    for grid in list_grids:
        print(grid.best_score_)
    print('=======================')
    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)
    print('=======================')
    return grid.best_estimator_


def score_by_shuffle(clf, features, labels, n_splits=None):
    sss = StratifiedShuffleSplit(n_splits=len(labels), test_size=0.3, random_state=0)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in sss.split(features, labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print ("Warning: Found a predicted label not == 0 or 1.")
                print ("All predictions should take value 0 or 1.")
                print ("Evaluating performance for processed predictions:")
                break
    # selector = SelectPercentile(f_classif, percentile=30)
    # selector.fit(features, labels)
    try:
        PERF_FORMAT_STRING = "\
        \tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
        Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
        RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
        \tFalse negatives: {:4d}\tTrue negatives: {:4d}"
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print (clf)
        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print ("")
    except Exception as e:
        print (e)
        print ("Got a divide by zero when trying out:", clf)
        print ("Precision or recall may be undefined due to a lack of true positive predicitons.")
    result = {'CLF': clf, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'F2': f2}
    return result


def count_scores(l_true, l_pred):
	f1 = metrics.f1_score(l_true, l_pred)
	recall = metrics.recall_score(l_true, l_pred)
	precision = metrics.precision_score(l_true, l_pred)
	print('F1 Score:', f1)
	print('Recall Score:', recall)
	print('Precision Score:', precision)


def draw_plot(data):
	'''
		Draw a plot with salary as X axis and bonus as Y axis
	'''
	for point in data:
		salary = point[1]
		bonus = point[2]
		pyplot.scatter(salary, bonus)
	pyplot.xlabel('salary')
	pyplot.ylabel('bonus')
	pyplot.show()


def clean_outliers(dataset, count=1, percent=10):
	'''
		Remove kinds ratio of outliers from dataset, 
		the top 1 as default, otherwise, remove 10%.
	'''
	def sort_func(x):
		'''
			This func is just for sortting finance data with salary
		'''
		if np.isnan(x[0]):
			return 0
		return -x[0]

	tmp_list = []
	key_list = []
	for name, features in dataset.items():
		tmp_list.append(np.array([features['salary'], features['bonus']], dtype='float64'))
	tmp_list = sorted(tmp_list, key=sort_func)
	if count:
		outliers_list = tmp_list[:int(count)]
	else:
		outliers_list = tmp_list[:int(len(tmp_list)/percent)]
	for outlier in outliers_list:
		for key, features in dataset.items():
			if features['salary'] == outlier[0] and features['bonus'] == outlier[1]:
				key_list.append(key)
	for key in key_list:
		dataset.pop(key, 0)


def gen_features(dataset):
	set_features = set()
	# Erase features of messages
	list_filter = ['from_poi_to_this_person', 'to_messages', 'email_address', 
	'shared_receipt_with_poi', 'from_messages', 'from_this_person_to_poi']
	count = 0
	for _, features in dataset.items():
		if count:
			set_features = set_features.intersection(set(features.keys()))
		else:
			set_features = set(features.keys())
		count += 1
	set_features = list(set_features)
	for i in list_filter:
		if i in set_features:
			set_features.pop(set_features.index(i))
	poi = set_features.pop(set_features.index('poi'))
	salary = set_features.pop(set_features.index('salary'))
	bonus = set_features.pop(set_features.index('bonus'))
	set_features.insert(0, poi)
	set_features.insert(1, salary)
	set_features.insert(2, bonus)
	return set_features


### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
clean_outliers(my_dataset)
features_list = gen_features(my_dataset)
# for name, features in my_dataset.items():
# 	pprint.pprint(features)
# print(features_list)
# features_list_2 = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
# 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
# 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
# draw_plot(data)
features = MinMaxScaler().fit_transform(features)
# features = Normalizer().fit_transform(features)
print(features.shape)
print(features[0])
sb = SelectKBest(chi2, k=2)
features = sb.fit_transform(features, labels)
# rfe = RFE(estimator=SVC(kernel='linear', C=1), n_features_to_select=2)
# features = rfe.fit_transform(features, labels)
print(features.shape)
print(features[0])

list_score = ['f1', 'recall', 'accuracy', 'precision']
list_result = []
for scoring in list_score:
    print('---====[{0}]====---'.format(scoring))
    clf = choose_best_est(features, labels, scoring)
    result = score_by_shuffle(clf, features, labels)
    list_result.append(result)
    # print('CLF:\n', clf)
pprint(list_result)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
# clf.fit(x_train, y_train)
# pred = clf.predict(x_test, y_test)
# print(clf.score(x_test, y_test))
# print(metrics.precision_score(y_test, pred))
# print(metrics.recall_score(y_test, pred))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)