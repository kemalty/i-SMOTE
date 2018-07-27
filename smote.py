import numpy
import random

from sklearn.neighbors import NearestNeighbors


def _fit_nn(x, k):
	neigh = NearestNeighbors(n_neighbors=k)
	return neigh.fit(x)

def _generate_instance(nn_finder, sample_x, all_x, gen_type):
	nn_idx = nn_finder.kneighbors(sample_x.reshape(1, -1))[1]
	picked_nn = random.sample(nn_idx[0].tolist(), 1)[0]
	nn_x = all_x[picked_nn, :]

	generated_x = numpy.zeros(nn_x.shape)
	for attr_idx in range(len(nn_x)):
		diff = nn_x[attr_idx] - sample_x[attr_idx]
		gap = random.random()

		if gen_type == "SMOTE":
			generated_x[attr_idx] = sample_x[attr_idx] + gap * diff
		elif gen_type == "I-SMOTE":
			generated_x[attr_idx] = sample_x[attr_idx] - gap * diff
		else:
			raise ValueError("Unexpected Generation Type")

	return generated_x

def _smote_entry(x, expansion_ratio, k, gen_type):
	num_sample = x.shape[0]
	target_num_sample = round(num_sample * expansion_ratio)
	num_to_generate = target_num_sample - num_sample

	nn_finder = _fit_nn(x, k)
	generated_x = []
	for i in range(num_to_generate):
		picked_instance = x[random.randint(0, num_sample-1), :]
		generated_instance = _generate_instance(nn_finder, picked_instance, x, gen_type)
		generated_x.append(generated_instance)

	return numpy.array(generated_x)

def i_smote(x, expansion_ratio, k):
	print("Oversampling with i-SMOTE...")
	generated_x = _smote_entry(x, expansion_ratio, k, "I-SMOTE")
	print("Done...")
	return generated_x

def smote(x, expansion_ratio, k):
	print("Oversampling with SMOTE...")
	generated_x = _smote_entry(x, expansion_ratio, k, "SMOTE")
	print("Done...")
	return generated_x
	
