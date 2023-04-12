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
	<img src="https://github.com/xmouyang/Harmony/blob/main/figure/harmony-system-overview.png" width="500">
</p>

First Stage: modality-wise FL
* Harmony on client: 
	* For all nodes: local unimodal training;
	* For multimodal nodes: resource allocation on different unimodal tasks;
	* For all nodes: send model weights to the server.
* Harmony on server: 
	* Run multiple threads for different unimodal FL subsystems to recieve model weights from clients;
	* Aggregate model weights of different modalities with Fedavg;
	* Send the aggregated model weights, and the round completion time (for calculating resource ratio) to each client.
	
Second Stage: federated fusion learning among multimodal nodes
* Harmony on client: 
	* Load the model weights of unimodal encoders trained in the first stage;
	* Local fusion: train the classifier layers and finetune the unimodal encoders;
	* Measure the modality bias of unimodal encoders;
	* Send model weights and modality bias to the server.
* Harmony on server: 
	* Cluster the nodes based on their modality bias;
	* Average the classifier layers of nodes in the same cluster. 


# Project Strcuture
```
|-- client                    // code in client side
    |-- client_cfmtl.py/	// main file of client 
    |-- communication.py/	// set up communication with server
    |-- data_pre.py/		// prepare for the FL data
    |-- model_alex_full.py/ 	// model on client 
    |-- desk_run_test.sh/	// run client 

|-- server/    // code in server side
    |-- server_cfmtl.py/        // main file of client
    |-- server_model_alex_full.py/ // model on server 

|-- README.md

|-- pictures               // figures used this README.md
```
<br>

# Quick Start
* Download the `dataset` (three publich datasets and one data collected by ourselves for AD monitoring) from [Harmony-Datasets](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155136315_link_cuhk_edu_hk/EvkjzyZRYBBIuaBD-tR8-7QBzaJ1xIfa1eQGUIveGTwPPw?e=cFo7cC) to your client machine.
* Chooose one dataset from the above four datasets and change the "read-path" in 'data_pre.py' to the path on your client machine.
* Change the "server_addr" and "server_port" in 'client_cfmtl.py' as your true server address. 
* Run the following code on the client machine
    ```bash
    cd client
    ./desk_run_test.sh
    ```
* Run the following code on the server machine
    ```bash
    cd server
    python3 server_cfmtl.py
    ```
    ---

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
    

