
import numpy
import pandas
import random

from mnist import MNIST

from sklearn.svm import SVC
from sklearn import metrics

DATA_DIR = "./data"
DATA_REDUCTION_RATIO = 0.05

def get_data(is_train):
	print("Loading data...")
	mn = MNIST(DATA_DIR)
	if is_train:
		x, y = mn.load_training()
	else:
		x, y = mn.load_testing()
	x, y = numpy.array(x), numpy.array(y)

	# Reduce data size for debug, only for train
	if is_train and DATA_REDUCTION_RATIO != 1:
		assert DATA_REDUCTION_RATIO > 0 and DATA_REDUCTION_RATIO <= 1
		num_sample = x.shape[0]
		num_remain = round(num_sample * DATA_REDUCTION_RATIO)
		assert num_remain > 0
		idx_to_keep = random.sample(list(range(num_sample)), num_remain)
		x = x[idx_to_keep, :]
		y = y[idx_to_keep]
	
	print("Done...")
	return x, y

def reduce_minority_class(x_train, y_train, minority_class, target_minority_size):
	print("Reducing minority class...")
	assert target_minority_size > 0
	minority_bool_idx = y_train == minority_class
	minority_size = minority_bool_idx.sum()
	assert target_minority_size <= minority_size
	minority_idx = numpy.array(range(len(y_train)))[minority_bool_idx]
	num_to_del = minority_size - target_minority_size
	idx_to_del = random.sample(list(minority_idx), num_to_del)
	x_train = numpy.delete(x_train, idx_to_del, axis=0)
	y_train = numpy.delete(y_train, idx_to_del, axis=0)
	print("Done...")
	return x_train, y_train

def transform_y_to_binary(y, minority_class):
	y[y == minority_class] = 255
	y[y != 255] = 0
	y[y == 255] = 1
	assert y.sum() > 0
	return y

def train_rbf_svm(x_train, y_train, C, gamma):
	print(f"Training model with: C={C} gamma={gamma}...")
	trainer = SVC(kernel="rbf", C=C, gamma=gamma, verbose=True)
	model = trainer.fit(x_train, y_train)
	print("Done...")
	return model

def get_performance(y, y_hat):
	return {"f1": metrics.f1_score(y, y_hat),
			"accuracy": metrics.accuracy_score(y, y_hat),
			"precision": metrics.precision_score(y, y_hat),
			"recall": metrics.recall_score(y, y_hat)}
	
def main():

	# Get training data
	x_train, y_train = get_data(True)

	# Pick the class that will be minority
	minority_class = random.randint(0, 9)

	# Subsample from minority class
	minority_size_target = 10
	x_train, y_train = reduce_minority_class(x_train, y_train, minority_class, minority_size_target) # TODO
	assert (y_train == minority_class).sum() == minority_size_target

	# Make the problem binary
	y_train = transform_y_to_binary(y_train, minority_class)

	# Train model and predict
	model = train_rbf_svm(x_train, y_train, 1, "auto")
	
	# Get train performance and relax the memory
	print("Predicting train...")
	y_hat_train = model.predict(x_train)
	train_performance = get_performance(y_train, y_hat_train)
	print(train_performance)
	del x_train, y_train
	print("Done...")
	
	# Get test data and get test performance
	x_test, y_test = get_data(False)
	print("Predicting test...")
	y_hat_test = model.predict(x_test)
	y_test = transform_y_to_binary(y_test, minority_class)
	test_performance = get_performance(y_test, y_hat_test)
	print(test_performance)
	print("Done...")


	

if __name__ == "__main__":
	print("Script start...")
	main()
	print("Script end...")

