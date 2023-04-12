# Harmony
This is the repo for MobiSys 2023 paper: "Harmony: Heterogeneous Multi-Modal Federated Learning through Disentangled Model Training".
<br>

# Requirements
The program has been tested in the following environment:

Cloud Cluster
* Python 3.9.7
* Pytorch 1.12.0
* torchvision 0.13.0
* CUDA Version 10.2
* sklearn 0.24.2
* numpy 1.20.3

Nvidia Xavier NX
* Ubuntu 18.04.6
* Python 3.6.9
* Pytorch 1.8.0
* CUDA Version 10.2
* sklearn 0.24.2
* numpy 1.19.5
<br>

# Harmony Overview
<p align="center" >
	<img src="https://github.com/xmouyang/Harmony/blob/main/figure/harmony-system-overview.png" width="800">
</p>

First Stage: Modality-Wise FL
* Harmony on client: 
	* For all nodes: local unimodal training; For multimodal nodes: resource allocation on different unimodal tasks;
	* For all nodes: send model weights to the server.
* Harmony on server: 
	* Use different ports for different unimodal FL subsystems to recieve model weights from clients;
	* Aggregate model weights of different modalities with Fedavg;
	* Send the aggregated model weights and the round completion time to each client.
	
Second Stage: Federated Fusion Learning (among multimodal nodes)
* Harmony on client: 
	* Local fusion: train the classifier layers and finetune the unimodal encoders;
	* Measure the modality bias of unimodal encoders;
	* Send model weights and modality bias to the server.
* Harmony on server: 
	* Cluster the nodes based on their modality bias;
	* Average the classifier layers of nodes in the same cluster. 


# Project Strcuture
```
|--harmony-AD-accuracy // codes running on cloud clusters with multiple GPUs, for evaluating the accuracy on the self-collected AD dataset

	|-- client                    // codes in the client side
	    |-- run_unifl_all.sh/	// run the first stage of all clients on a cloud cluster 
	    |-- run_fedfusion_all.sh/	// run the second stage of all clients on a cloud cluster 
	    |-- main_unimodal.py/	// main file of running first stage on the client
	    |-- main_fusion.py or main_fusion_3modal.py or main_fusion_2modal.py/	// main file of running second stage on the client
	    |-- communication.py/	// set up communication with server
	    |-- data_pre.py/		// load the data for clients in FL
	    |-- model.py/ 	// model configuration for different datasets 
	    |-- sample_index.zip/ 	// index of training and testing data, need to be uncompressed
	    |-- util.py		// utility functions


	|-- server/    // codes in the server side
	    |-- main_server_stage1_uniFL.py/        // main file of the first stage of the server
	    |-- main_server_stage2_fedfusion_3modal.py/ // main file of the second stage of the server

|--harmony-AD-edge-schedule // codes running on edge devices (Nvidia Xavier NX), for evaluating the resource allocation scheme on the self-collected AD dataset

|--harmony-Flash // codes running on cloud clusters with multiple GPUs, for evaluating the accuracy on the FLASH dataset

|--harmony-MHAD // codes running on cloud clusters with multiple GPUs, for evaluating the accuracy on the MHAD dataset

|--harmony-USC // codes running on cloud clusters with multiple GPUs, for evaluating the accuracy on the USC dataset

```
<br>

# Quick Start 
* Download the codes for each dataset in this repo. Put the folder `client` on your client machine and `server` on your server machine.
* Download the `dataset` (three publich datasets and one data collected by ourselves for AD monitoring) from [Harmony-Datasets](https://github.com/xmouyang/Harmony/blob/main/dataset.md) to your client machine.
* Choose one dataset from the above four datasets and put the folder `under the same folder` with corresponding codes. You can also change the path of loading datasets in 'data_pre.py' to the data path on your client machine.
* Change the argument "server_address" in 'main_unimodal.py' and 'main_fedfuse.py' as your true server address. If your server is located in the same physical machine of your nodes, you can choose "localhost" for this argument.
* Run the following code on the client machine
	* For running clients on the cloud cluster (clients are assigned to different GPUs)
		* Run the first stage
		    ```bash
		    ./run_unifl_all.sh
		    ```
		* Run the second stage
		    ```bash
		    ./run_fedfusion_all.sh
		    ```
	* For running clients on the edge devices (clients are assigned to different Nvidia Xavier NX device)
		* Run the first stage: move the bash file of node xx (run_unifl_xx.sh or run_unifl_schedule_xx.sh) from the folder 'node-run-stage1' to the folder 'client'
			* For single-modal nodes: 
			    ```bash
			    ./run_unifl_xx.sh
			    ```
			* For multi-modal nodes: 
			    ```bash
			    ./run_unifl_schedule_xx.sh
			    ```
		* Run the second stage: move the bash file of node xx (run_fedfusion_xx.sh) from the folder 'node-run-stage2' to the folder 'client'
		    ```bash
		    ./run_fedfusion_xx.sh
		    ```
* Run the following code on the server machine
	* Run the first stage: run multiple tasks for different unimodal FL subsystems
	    ```bash
	    python3 main_server_stage1_uniFL.py --modality_group 0
	    python3 main_server_stage1_uniFL.py --modality_group 1
	    python3 main_server_stage1_uniFL.py --modality_group 2
	    ```
	* Run the second stage
	    ```bash
	    python3 main_server_stage2_fedfusion_3modal.py
	    ```
    ---

<!--
# Citation
The code and datasets of this project are made available for non-commercial, academic research only. If you would like to use the code or datasets of this project, please cite the following papers:
```
@inproceedings{ouyang2023harmony,
  title={Harmony: Heterogeneous Multi-Modal Federated Learning through Disentangled Model Training},
  author={Ouyang, Xiaomin and Xie, Zhiyuan and Fu, Heming, and Chen, Sitong and Pan Li, and Ling Neiwen, and Xing, Guoliang, and Zhou Jiayu, and Huang Jianwei},
  booktitle={Proceedings of the 21th Annual International Conference on Mobile Systems, Applications, and Services},
  year={2023}
}
```
-->
    

