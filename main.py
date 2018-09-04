import uuid
import copy
import numpy
import pandas
import random
import argparse

from mnist import MNIST
from sklearn.svm import SVC
from sklearn import metrics

from smote import *

DATA_DIR = "./data"
OUT_DIR = "./experiment_out"
REPEAT = 10

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

def expand_with_smote(x, y, minority_class, expansion_rate, k):
	assert k > 0
	assert expansion_rate > 1
	assert (y==minority_class).sum() >= k

	x_minority = x[y == minority_class, :]
	return smote(x_minority, expansion_rate, k)

def expand_with_i_smote(x, y, minority_class, expansion_rate, k):
	assert k > 0
	assert expansion_rate > 1
	assert (y==minority_class).sum() >= k

	x_minority = x[y == minority_class, :]
	return i_smote(x_minority, expansion_rate, k)

def rescale_x(x):
	return x/255

def run_experiment_loop(x_train, y_train, x_test, y_test):

	# Subsample from minority class
	print(f"Reducing minority class size to: {MINORITY_SIZE_TARGET}")
	x_train, y_train = reduce_minority_class(x_train, y_train, MINORITY_CLASS, MINORITY_SIZE_TARGET)
	assert (y_train == MINORITY_CLASS).sum() == MINORITY_SIZE_TARGET

	# Expand the minority class
	if EXPANSION_METHOD in ["SMOTE", "I-SMOTE"]:
		if EXPANSION_METHOD == "SMOTE":
			x_generated = expand_with_smote(x_train, y_train, MINORITY_CLASS, EXPANSION_RATE, K)
			
		elif EXPANSION_METHOD == "I-SMOTE":
			x_generated = expand_with_i_smote(x_train, y_train, MINORITY_CLASS, EXPANSION_RATE, K)
		
		y_generated = numpy.ones(x_generated.shape[0]) * MINORITY_CLASS
		x_train = numpy.append(x_train, x_generated, axis=0)
		y_train = numpy.append(y_train, y_generated, axis=0)

	elif EXPANSION_METHOD == "NONE":
		print("No Oversampling...")

	else:
		raise ValueError("Unexpected Expansion Method")

	# Make the problem binary
	y_train = transform_y_to_binary(y_train, MINORITY_CLASS)

	# Rescale features
	x_train = rescale_x(x_train)

	# Train model and predict
	model = train_rbf_svm(x_train, y_train, C, GAMMA)
	
	# Get train performance and relax the memory
	print("Predicting train...")
	y_hat_train = model.predict(x_train)
	train_performance = get_performance(y_train, y_hat_train)
	print(train_performance)
	del x_train, y_train
	print("Done...")
	
	# Get test performance
	print("Predicting test...")
	x_test = rescale_x(x_test)
	y_hat_test = model.predict(x_test)
	y_test = transform_y_to_binary(y_test, MINORITY_CLASS)
	test_performance = get_performance(y_test, y_hat_test)
	print(test_performance)
	print("Done...")
	
	# Save results to file
	print("Saving results...")
	save_experiment_results(train_performance, test_performance)
	print("Done...")

	
def main():

	print(f"Minority class is: {MINORITY_CLASS}")

	# Get data
	x_train, y_train = get_data(True)
	x_test, y_test = get_data(False)

	# Repeat because of randomness
	for repeat in range(REPEAT):
		print(f"Repeat {repeat} of {REPEAT}...")
		run_experiment_loop(copy.copy(x_train), 
							copy.copy(y_train), 
							copy.copy(x_test), 
							copy.copy(y_test))


def save_experiment_results(train_performance, test_performance):
	# Bundle all experiment info together
	save_dict = {"train_f1": train_performance["f1"],
				 "test_f1": test_performance["f1"],
				 "train_accuracy": train_performance["accuracy"],
				 "test_accuracy": test_performance["accuracy"],
				 "train_precision": train_performance["precision"],
				 "test_precision": test_performance["precision"],
				 "train_recall": train_performance["recall"],
				 "test_recall": test_performance["recall"],
				 "data_reduction_ratio": DATA_REDUCTION_RATIO,
				 "expansion_method": EXPANSION_METHOD,
				 "C": C,
				 "gamma": GAMMA,
				 "minority_size_target": MINORITY_SIZE_TARGET,
				 "expansion_rate": EXPANSION_RATE,
				 "K": K,
				 "minority_class": MINORITY_CLASS,
				 "data_reduction_ratio": DATA_REDUCTION_RATIO}

	# Get save path id
	experiment_uid = uuid.uuid4()
	save_path = f"{OUT_DIR}/{experiment_uid}.npy"
	
	numpy.save(save_path, save_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_reduction_ratio", type=float, 
    	help="Ratio of training data to be used for debugging purposes.")
    parser.add_argument("expansion_method", type=str, choices=["I-SMOTE", "SMOTE", "NONE"],
    	help="Method to use to expand minority class.")
    parser.add_argument("C", type=float, help="C for SVC")
    parser.add_argument("gamma", help="gamma for SVC")
    parser.add_argument("minority_size_target", type=int,
    	help="Number of instance minority class must have in experiment.")
    parser.add_argument("expansion_rate", type=float,
    	help="The rate of expansion minority class should have using expansion method.")
    parser.add_argument("K", type=int, help="K for (i-)SMOTE.")
    parser.add_argument("minority_class", type=int, 
    	help="Class to take as minority")

    args = parser.parse_args()

    # Type checkers
    assert args.data_reduction_ratio > 0 and args.data_reduction_ratio <= 1
    assert args.minority_size_target >= 2
    assert args.expansion_rate > 1
    assert args.K > 1
    assert args.K <= args.minority_size_target
    assert args.minority_class in list(range(10))

    return args
	

if __name__ == "__main__":
	print("Script start...")
	
	args = parse_args()
	
	DATA_REDUCTION_RATIO = args.data_reduction_ratio
	EXPANSION_METHOD = args.expansion_method
	C = args.C
	GAMMA = args.gamma
	MINORITY_SIZE_TARGET = args.minority_size_target
	EXPANSION_RATE = args.expansion_rate
	K = args.K
	MINORITY_CLASS = args.minority_class

	main()

	print("Script end...")


