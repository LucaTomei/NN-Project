import matplotlib
import numpy as np
import matplotlib.pyplot as plt



def get_data(lines):
	losses = []
	accuracies = []
	val_losses = []
	val_accuracies = []
	for line in lines:
		if '==' in line:
			if 'loss' in line: loss = line.split("loss: ")[1].split("-")[0].replace(' ', '')
			if 'accuracy' in line: accuracy = line.split("accuracy: ")[1].split("-")[0].replace(' ', '')
			if 'acc' in line and not 'accuracy' in line: 
				accuracy = line.split("acc: ")[1].split("-")[0].replace(' ', '')

				
			if 'val_loss' in line: val_loss = line.split("val_loss: ")[1].split("-")[0].replace(' ', '')
			
			if 'val_accuracy' in line: val_accuracy = line.split("val_accuracy: ")[1].split('\n')[0]
			if 'val_acc' in line and not 'val_accuracy' in line:
				val_accuracy = line.split("val_acc: ")[1].split("-")[0].replace(' ', '')
			
			if 'loss' in line and 'val_loss' in line:
				losses.append(float(loss))
				accuracies.append(float(accuracy))
				val_losses.append(float(val_loss))
				val_accuracies.append(float(val_accuracy))
	return (losses, accuracies, val_losses, val_accuracies)


def read_file(filename):
	file = open(filename, "r")
	content = file.readlines()
	file.close()
	return content


if __name__ == '__main__':
	files = ["/Users/lucasmac/Downloads/CNN2_MNIST/output_LeakyReLU.txt", 
	"/Users/lucasmac/Downloads/CNN2_MNIST/output_PReLU.txt", 
	"/Users/lucasmac/Downloads/CNN2_MNIST/output_ReLU.txt", 
	"/Users/lucasmac/Downloads/CNN2_MNIST/Risultati_MySReLU.txt",
	"/Users/lucasmac/Downloads/CNN2_MNIST/Risultati_SReLU.txt"]

	

	(losses1, accuracies1, val_losses1, val_accuracies1) = get_data(read_file(files[0]))
	(losses2, accuracies2, val_losses2, val_accuracies2) = get_data(read_file(files[1]))
	(losses3, accuracies3, val_losses3, val_accuracies3) = get_data(read_file(files[2]))
	(losses4, accuracies4, val_losses4, val_accuracies4) = get_data(read_file(files[3]))
	(losses5, accuracies5, val_losses5, val_accuracies5) = get_data(read_file(files[4]))

	
	# # print(val_accuracies1 == val_accuracies2)


	# Plot history: MAE
	plt.grid(True)
	plt.plot(val_accuracies1, label='LeakyReLU')
	plt.plot(val_accuracies2, label='PReLU')
	plt.plot(val_accuracies3, label='ReLU')
	plt.plot(val_accuracies4, label='MySReLU')
	plt.plot(val_accuracies5, label='SReLU')
	#plt.title('MAE for Chennai Reservoir Levels')
	plt.ylabel('Accuracy')
	plt.xlabel('No. epoch')
	plt.legend(loc="upper left")
	plt.show()