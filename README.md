# driver-behavior-scanner
Project to identify driver behavior during vehicle conduction based on a machine learning system. 

#Introduction
The idea of this project is to develop a machine learning based system capable of identifying different drivers behaviors during
driving. This system could be used for any rideshare company to evaluate its fleet behavior and misconduct actions.

The project consists in utilizing a pre-trained convolutional neural network and perform a transfer learning to our use
case scenario. The project consists in three steps :
1. Data uploading and visualization
2. Hyperparameter tuning
3. Model training / profiling / debugging
4. Model deployment

#Instructions
TO DO : Containerize the project into a DockerFile.

* All of those previous steps can be found in the ipython notebook in the repository. 
* The code folder contains the scripts used for :
- Code

  |_ hpo.py : hyperparameter tunning
  
  |_ train_model.py : train model with profiling and debugging
  
  |_ inference.py : functions needed to PyTorchModel estimator performs inferences
  
  |_ test_inference.py : simulate inference before deploying model
