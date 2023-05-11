# Harmony
This is the repo for MobiSys 2023 paper: "Harmony: Heterogeneous Multi-Modal Federated Learning through Disentangled Model Training".
<br>

# Requirements
The program has been tested in the following environment:
* Computing Clusters: Python 3.9.7, Pytorch 1.12.0, torchvision 0.13.0, CUDA Version 10.2, sklearn 0.24.2, numpy 1.20.3
* Nvidia Xavier NX: Ubuntu 18.04.6, Python 3.6.9, Pytorch 1.8.0, CUDA Version 10.2, sklearn 0.24.2, numpy 1.19.5

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
* Download the codes for each dataset in this repo. Put the folder `client` on your client machines and `server` on your server machine.
* Download the `dataset` (three public datasets and one dataset collected by ourselves for AD monitoring) from [Harmony-Datasets](https://github.com/xmouyang/Harmony/blob/main/dataset.md) to your client machines.
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
		* NOTE: You may need to change the running scripts "run_unifl_all.sh" and "run_fedfusion_all.sh" if you want to run multiple nodes on the same GPUs or run the nodes on different machines. For example, if you want to run 14 nodes in the USC dataset on only 4 GPUs, please run the shell scripts "run_unifl_all-4GPU.sh" and "run_fedfusion_all-4GPU.sh"; if you want to run 16 nodes in the self-collected AD dataset on 4 different machines, please move the shell scripts from the folder "node-run-stage1-4cluster" and "node-run-stage1-4cluster" to the source folder and run one script on each machine.
	* For running clients on the edge devices (clients are assigned to different Nvidia Xavier NX device)
		* Move the running script of each node (run_unifl_xx.sh, run_unifl_schedule_xx.sh and run_fedfusion_xx.sh) from the folder 'node-run-stage1' and 'node-run-stage2' to the folder 'client'
		* Run the first stage: 
			* For single-modal nodes: 
			    ```bash
			    ./run_unifl_xx.sh
			    ```
			* For multi-modal nodes: 
			    ```bash
			    ./run_unifl_schedule_xx.sh
			    ```
		* Run the second stage: 
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

# Run Your Own Node Configurations
If you only have limited GPU resources (e.g., 4GPUs) or edge devices, and want to run a small-scale Harmony. You can easily revise the following files to achieve it. Take running six nodes from the Flash dataset on 4 GPUs as an example.
* On the client side: 
	* Revise the shell script "run_unifl_all.sh": Here, "CUDA_VISIBLE_DEVICES=xx", "--local_modality xx", and "--usr_id xx" assigns the device id of GPU, the local data modality, and the ID of nodes, respectively. Please ensure that the "CUDA_VISIBLE_DEVICES" and "usr_id" start from 0. And you can assign multiple nodes in FL on the same GPU device. For example, in the below commands, node 0 is a single-modal node that trains the gps model on GPU0, and node 4 is a multi-modal node that trains three unimodal models (gps, lidar, and image) on GPU2.
	 ```bash
	CUDA_VISIBLE_DEVICES=0 python3 main_unimodal.py --local_modality gps --usr_id 0 &
	CUDA_VISIBLE_DEVICES=0 python3 main_unimodal.py --local_modality gps --usr_id 1 &
	CUDA_VISIBLE_DEVICES=1 python3 main_unimodal.py --local_modality lidar --usr_id 2 &
	CUDA_VISIBLE_DEVICES=1 python3 main_unimodal.py --local_modality image --usr_id 3 &
	CUDA_VISIBLE_DEVICES=2 python3 main_unimodal.py --local_modality gps --usr_id 4 &
	CUDA_VISIBLE_DEVICES=2 python3 main_unimodal.py --local_modality lidar --usr_id 4 &
	CUDA_VISIBLE_DEVICES=2 python3 main_unimodal.py --local_modality image --usr_id 4 &
	CUDA_VISIBLE_DEVICES=3 python3 main_unimodal.py --local_modality gps --usr_id 5 &
	CUDA_VISIBLE_DEVICES=3 python3 main_unimodal.py --local_modality lidar --usr_id 5 &
	CUDA_VISIBLE_DEVICES=3 python3 main_unimodal.py --local_modality image --usr_id 5 &
	```
	* Revise the shell script "run_fedfusion_all.sh": To run the second stage, you only need to include multi-modal nodes that appear in "run_unifl_all.sh" as follows, where now you can assign their modality as "all" (or "both" for the dataset with only two modalities) since they train multi-modal fusion models in this stage. The device id of the GPU and the node ID can remain the same.
	 ```bash
	CUDA_VISIBLE_DEVICES=2 python3 main_fusion.py --local_modality all --usr_id 4 &
	CUDA_VISIBLE_DEVICES=3 python3 main_fusion.py --local_modality all --usr_id 5 &
	```
* On the server side: 
	* Revise the text file "reorder_id_stage1_xxx.txt": These files re-order the node ID on the server of each FL unimodal subsystem, to start from 0 and end with num_of_users (-1) in the subsystem. Here the left column denotes the original node ID in the whole multi-modal FL system, and the right column denotes the re-ordered node ID in the subsystem. Note that we set the reordered ID as 1000, if the original node has no corresponding data modality. For example, in the above node configuration in the client side, we have four nodes in the unimodal FL subsystem of gps. Then the customized "reorder_id_stage1_gps.txt" is as follows.
	```bash
	0	0
	1	1
	2	1000
	3	1000
	4	2
	5	3
	```
	* Revise the text file "reorder_id_stage2.txt": This file re-orders the node ID of multi-modal nodes on the server in the second stage, to start from 0 and end with num_of_users (-1) of all multi-modal nodes. Here the left column denotes the original node ID in the whole multi-modal FL system, and the right column denotes the re-ordered node ID of multi-modal nodes. Note that we set the reordered ID as 1000, if the original node is a single-modal node. For example, in the above node configuration in the client side, we have two multi-modal nodes in the second stage. Then the customized "reorder_id_stage2.txt" is as follows.
	```bash
	0	1000
	1	1000
	2	1000
	3	1000
	4	0
	5	1
	```	
# Citation
The code and datasets of this project are made available for non-commercial, academic research only. If you would like to use the code or datasets of this project, please cite the following papers:
```
@inproceedings{ouyang2023harmony,
  title={Harmony: Heterogeneous Multi-Modal Federated Learning through Disentangled Model Training},
  author={Ouyang, Xiaomin and Xie, Zhiyuan and Fu, Heming and Chen, Sitong and Pan, Li and Ling, Neiwen and Xing, Guoliang and Zhou, Jiayu and Huang, Jianwei},
  booktitle={Proceedings of the 21th Annual International Conference on Mobile Systems, Applications, and Services},
  year={2023}
}
```

<!---->
    

