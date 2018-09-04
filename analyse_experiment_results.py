import numpy
import pandas

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join


OUT_DIR = "./experiment_out"


def load_all_out_files_as_df():
	file_paths = [f"{OUT_DIR}/{file}" for file in listdir(OUT_DIR) if isfile(join(OUT_DIR, file)) and file[0] != "."]
	list_of_dict = [numpy.load(path).item() for path in file_paths]
	return pandas.DataFrame.from_dict(list_of_dict)

def plot_size_vs_f1(out_data):
	plot_data = out_data[(out_data.K==5) & (out_data.expansion_rate==50)]
	plt.figure()
	idx = 0
	for method_name, method_data in plot_data.groupby("expansion_method"):
		color = f"C{idx}"
		idx += 1
		for size, size_data in method_data.groupby("minority_size_target"):

			plt.errorbar(size, size_data.test_f1.mean(),
						 yerr=size_data.test_f1.std(), capthick=0.1,
						 label=method_name, marker="x", color=color)
	plt.legend()
	plt.show()


def main():
	out_data = load_all_out_files_as_df()
	
	plot_size_vs_f1(out_data)


if __name__ == "__main__":
	print("Script start...")
	main()
	print("Script end...")