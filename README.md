# From Individuals to Communities: Community-Aware Language Modeling for the Detection of Hate Speech

This repository is part of our group's The Tide: racist and abusive trends in online communities project. 

If you use this code or data please cite:

> @inproceedings{arviv2020s,  
>  title={It's a Thin Line Between Love and Hate: Using the Echo in Modeling Dynamics of Racist Online Communities},  
>  author={Arviv, Eyal and Hanouna, Simo and Tsur, Oren},  
>  booktitle={Proceedings of the Fifteenth International AAAI Conference on Web and Social Media (ICWSM)},  
>  year={2021},  
> }.  

The paper is avaliable at: https://arxiv.org/abs/2012.01133

Lab page: https://www.naslab.ise.bgu.ac.il/


## Data
The annotated datasets used for training and testing the algorithms are available under `data`.
The raw unlabeled dataset is available at: https://www.dropbox.com/s/f9y0d6u5sjrbemd/echo_share.tar.gz?dl=0 (4.1GB)


## Code

To avoid environment issues you can create a new conda env using the following line:

`conda create --name <env> --file requirements.txt`

This project contains two main code segments:
1. Hate Speech Detection - under the module `detection`
2. Hate Networks - under the model `hate_networks`


## Hate Speech Detection
This is the main module of the thesis. It contains both the post level models (PLMs) and the user level models (ULMs) to detect hate speech.

The configuration of the execution for this module is under the file `config.detection_config.py`. Under this file you can control what dataset is running for the specific execution.

There are four entry points for this module. 
To execute each of the following experiments from the root dir of the project run:
 
    `python detection/experiments/{experiment_file_name}.py`
    
The entry points reside under `experiments` directory in the following files:
* Post-level experiments
    * `post_level__experiment.py` - run this file to execute a specific experiment of a model on a specific data set as configured in `config.detection_config.py`.
     To run this file set the parameter `multiple_experiments` to `False` under `post_level_execution_config` in `config.detection_config.py` file.
    * `post_level__multiple_experiments.py` - run this file to execute multiple experiments of several models and compare them.
    To run this file set the parameter `multiple_experiments` to `True` under `post_level_execution_config` in `config.detection_config.py` file.

    **Important note**: to run BertFineTuning model make sure you are using a gpu. 

* User-level experiments
    * `user_level__experiments.py` - run this file to execute the user level experiment. 
    It will load the PLM from the `post_level_execution_config` config and predict the probabilities of all of the posts by all users in the given data to contain hate speech.
    Then it will run the FFNN using the streams of data by the user itself, his followees and his followers, together with network features.
        * **Important note**: one must run the hate network module with the desired dataset before running this code as it uses some output from it.
    * `user_level__threshold_models.py` - Run this file to execute threshold-based ULMs.
        * **Important note**: one must run the experiment file `user_level__experiments.py` with the desired dataset before running this code as it uses some output from it.

* The outputs of the models' executions will be saved under `detection/outputs/{data_name}_{model_name}`.


## Hate Networks
In this module we create networks promoting hate using data from social networks that present engagement between users, i.e., mentions, retweets. 
Using this module you can create unsupervised segmentation of users to various communities based on the set configuration.
The unsupervised methods are topic models (LDA/NMF) and Word2Vec modeling based on the users' texts.
Using these methods you are able to color the users in the reconstructed user-network.
You can also color the users using the predictions created using the user-classifier from the `detection` module.
This can be done by setting the param `plot_supervised_networks` to true in `general_conf` under `hate_networks_config.py` file and using the `user_pred` param under path_conf for each dataset.

Important notes:
* The entry point for this module is the function `main()` under the file `main.py`. To execute the code from the root dir of the project run:
 
    `python hate_networks/main.py`
    
* The execution of this code is configured using the configuration dict under the `config` module in the file `hate_networks_config.py`.

* The outputs of the constructed network and the relevant files that are created with it will be saved under `hate_networks/outputs/{data_name}_networks`.
