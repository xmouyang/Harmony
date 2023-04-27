# Heterogeneous-Multi-Modal-FL-Datasets

This repo includes four real-world multi-modal datasets collected under federated learning settings, which are used in the MobiSys 2023 paper: "Harmony: Heterogeneous Multi-Modal Federated Learning through Disentangled Model Training".

The first dataset is a multi-modal dataset for Alzheimer's Disease monitoring collected by ourselves and first appeared in the paper. The other three are public datasets pre-processed by us to federated learning settings.


# Download

  The four pre-processed datasets can be downloaded in the [onedrive folder](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155136315_link_cuhk_edu_hk/EvkjzyZRYBBIuaBD-tR8-7QBzaJ1xIfa1eQGUIveGTwPPw?e=wdET4Y). Please refer to the following discriptions of collecting and pre-processing for each dataset. 
  
  
### Alzheimer’s Disease Monitoring Dataset (self-collected): 

* Task: Detect 11 behavior biomarkers in natural home environments, including cleaning the living area, taking medication, using mobile phones, writing, sitting, standing, moving in/out of chair/bed, walking, sleeping, eating, and drinking.
* Sensor Modalities: Depth Camera, mmWave Radar and Microphone.
* Number of Sensor Nodes: 16 nodes deployed in 16 elderly subjects' homes.
* Size of the dataset: About 8GB.
* Original paper of the dataset:
```
@inproceedings{ouyang2023harmony,
  title={Harmony: Heterogeneous Multi-Modal Federated Learning through Disentangled Model Training},
  author={Ouyang, Xiaomin and Xie, Zhiyuan and Fu, Heming, and Chen, Sitong and Pan Li, and Ling Neiwen, and Xing, Guoliang, and Zhou Jiayu, and Huang Jianwei},
  booktitle={Proceedings of the 21th Annual International Conference on Mobile Systems, Applications, and Services},
  year={2023}
}
```


### FLASH Dataset: 

* Task: Select the high-band sector for mmWave beamforming in mobile V2X communication scenarios. 64 sectors in total. 
* Sensor Modalities: GPS, LiDAR, and image.
* Number of Sensor Nodes: Up to 210 nodes representing the data collected in different settings (vehicles and scenarios). Only the data of 30 nodes is shared in the link.
* Size of the dataset: About 1.5GB.
* Original paper of the dataset:
```
@inproceedings{salehi2022flash,
  title={FLASH: Federated learning for automated selection of high-band mmWave sectors},
  author={Salehi, Batool and Gu, Jerry and Roy, Debashri and Chowdhury, Kaushik},
  booktitle={IEEE INFOCOM 2022-IEEE Conference on Computer Communications},
  pages={1719--1728},
  year={2022},
  organization={IEEE}
}
```

### MHAD Dataset: 

* Task: Recognize 11 human actions. 
* Sensor Modalities: 3-axis accelerometer and skeleton data.
* Number of Sensor Nodes: 12 nodes representing the data collected from 12 subjects.
* Size of the dataset: about 165MB.
* Original paper of the dataset:
```
@inproceedings{ofli2013berkeley,
  title={Berkeley mhad: A comprehensive multimodal human action database},
  author={Ofli, Ferda and Chaudhry, Rizwan and Kurillo, Gregorij and Vidal, Ren{\'e} and Bajcsy, Ruzena},
  booktitle={2013 IEEE workshop on applications of computer vision (WACV)},
  pages={53--60},
  year={2013},
  organization={IEEE}
}
```

### USC Dataset: Gesture Recognition using Depth Camera

* Task: Recognize 12 human activities. 
* Sensor Modalities: 3-axis accelerator and 3-axis gyroscope data.
* Number of Sensor Nodes: 14 nodes representing the data collected from 14 subjects.
* Size of the dataset: about 6MB.
* Original paper of the dataset:
```
@inproceedings{zhang2012usc,
  title={USC-HAD: A daily activity dataset for ubiquitous activity recognition using wearable sensors},
  author={Zhang, Mi and Sawchuk, Alexander A},
  booktitle={Proceedings of the 2012 ACM conference on ubiquitous computing},
  pages={1036--1043},
  year={2012}
}
```


# Citation
The code and datasets of this project are made available for non-commercial, academic research only. If you would like to use the code or the Alzheimer’s Disease Monitoring datasets of this project, please cite the following papers:
```
@inproceedings{ouyang2023harmony,
  title={Harmony: Heterogeneous Multi-Modal Federated Learning through Disentangled Model Training},
  author={Ouyang, Xiaomin and Xie, Zhiyuan and Fu, Heming, and Chen, Sitong and Pan Li, and Ling Neiwen, and Xing, Guoliang, and Zhou Jiayu, and Huang Jianwei},
  booktitle={Proceedings of the 21th Annual International Conference on Mobile Systems, Applications, and Services},
  year={2023}
}
```
