import json 
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":

	path = os.path.join("checkpoints", "metrics.json")
	with open(path, "r") as json_file:
		metrics = json.load(json_file)

	decorr_keys = ["correlation_train_loss", "correlation_val_loss",
					"whitening_train_loss", "whitening_val_loss"]

	backprop_keys = ["train_perplexity", "val_perplexity", 
					"cross_entropy_train_loss", "cross_entropy_val_loss"]

	colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange"]
	linestyles = ["-", "dashed", "-", "dashed"]

	# model performance
	
	plt.figure()
	for i, key in enumerate(backprop_keys):
		plt.plot(
			np.arange(0,len(metrics[key])), metrics[key], label=key,
			color=colors[i], ls=linestyles[i])

	plt.legend()
	plt.xlim([0,len(metrics[backprop_keys[0]])-1])
	plt.xlabel("Epoch")
	plt.xticks(range(0, len(metrics[key]), 4))
	plt.ylabel("BCE/perplexity")
	plt.title("BCE loss and perplexity across epochs")
	plt.show()

	# decorrelation performance
	
	plt.figure()
	for i, key in enumerate(decorr_keys):
		plt.plot(
			np.arange(0,len(metrics[key])), metrics[key], label=key,
			color=colors[i], ls=linestyles[i])

	plt.legend()
	plt.xlim([0,len(metrics[backprop_keys[0]])-1])
	plt.xlabel("Epoch")
	plt.xticks(range(0, len(metrics[key]), 4))	
	plt.ylabel("Correlation/whitening loss")
	plt.title("Correlation and whitening losses across epochs")
	plt.show()

