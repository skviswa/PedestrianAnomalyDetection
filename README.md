# Anomaly Detection using Prednet
This project consists of the following steps:

   1.Pre-processing data and generating hickle files using process.py to easily run train and inference <br />
   2.Setting the required hyperparameters and feeding the dataset to train a Prednet model from scratch on the data with train.py <br />
   3.Running inference with the trained model on the test dataset using evaluate.py and collecting various metrics and visualizations to analyse the performance <br />

Processing data: <br />
   The UCSD Anomaly Detection Dataset (http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm), needs to be downloaded and extracted before we can move further.<br />
   Do note that we had randomly split the test folder of both Ped1 and Ped2, to get some validation data. And we did some pre-processing to shift the ground-truth folders in the Test folder to a separate GT folder so that we dont end up processing it by mistake. <br />
   
   The following files are used:<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i) settings.py - This utility needs to be set by the user. It documents where we find data, model, logs and results for the entire project. <br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii) process.py - This code will take path to the dataset and load each frame of each video, preprocess it and write it to the hickle file. <br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                This automatically separates files for train, test and validation data. <br />

Once the above step is completed, the we will be ready to train our models.<br />

The training process:<br />
   The following files are used:<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i) train.py - This file instantiates the Prednet model, the loss function and the optimizer.<br />
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; It then trains the model for the choice of hyperparameters that are set by us using the fix_hyperparams function. <br />
               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  We document the training logs, checkpoint the model to save the best one that can be used later in the inference process.<br />
			   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  To learn more about Prednet, visit https://coxlab.github.io/prednet/ which has both the paper and the code in more detail. <br />
        
Finally we will be ready to run inference and analyse our performance:<br />
   The following files are used:<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i) evaluate.py - This file instantiates the Prednet model, loads the right pre-trained model, and runs the network on the test dataset.<br />
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; There are various utilities that collects results and plots visualizations. <br />
               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  We generate visualizations such as MSE and SD error plots between model and test videos, and previous frame of test videos respectively. <br />
			   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  We can find all of that stored in the RESULTS_DIR path under the respective dataset names. <br />

You can find example animations of the results at https://youtu.be/ueJi8Tfn-6E <br />

## Libraries used
This has been documented in the environment yaml file. This can be used to set up the environment as well. <br />
Here is an example of how to update the current environment using conda distribution: <br />
source activate myenv <br />
conda env update -f=environment.yml

##Development/IDEs
Spyder(3.3 or above)