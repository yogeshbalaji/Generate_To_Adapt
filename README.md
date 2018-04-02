# Generate_To_Adapt
Implementation of "Generate To Adapt: Aligning Domains using Generative Adversarial Networks" in PyTorch

Datasets:
Please download the dataset from http://www.cs.umd.edu/~yogesh/datasets/digits.zip and extract it. This folder contains the dataset in the same format as need by our code.

Training:

Let us train the Lenet model for SVHN->MNIST Domain adaptation. Obtain the baseline numbers by running

	python main.py --dataroot [path to the dataset] --method sourceonly
	
To train our method(GTA), run

	python main.py --dataroot [path to the dataset] --method GTA

This code trains and stores the trained models in result folder. Current checkpoint and the model that gives best performance on the validation set are stored.

Evaluation:

To evaluate the trained models on the target domain (MNIST), run 

	python eval.py --dataroot [path to the dataset] --method GTA --model_best False
