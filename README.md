# DenseNet-MURA-TensorFlow 

This repository contains a TensorFlow implementation of the 169-layers [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) (DenseNet-169) model for the task of abnormality detection in musculoskeletal Radiographs on [MURA](https://arxiv.org/pdf/1712.06957.pdf) dataset.


## Model Initialization
 The weights of the model are initialized from a pretrained model on ImageNet using caffe, [DenseNet-Caffe](https://github.com/shicai/DenseNet-Caffe). The caffemodel weights are then extracted as Numpy arrays and saved as a pickle file using [caffe_weight_converter](https://github.com/pierluigiferrari/caffe_weight_converter). The extracted weights can be downloaded [here](https://drive.google.com/file/d/1dylF-d_09F8hinlepSrmumg1-IxtBD1x/view?usp=sharing), or generated using the above-mentioned repositories.
 
 
 ## Training 
 
 The network is fine-tuned end-to-end using Adam optimiser with default parameters (β1 = 0.9 and β2 = 0.999), and an initial learning rate of 1e-4 which is divided by 10 every time the validation loss plateaus. Two training strategies have been tested. First, the network is trained on the entire dataset, resulting in a single model which is used for inference. Second, the network is trained on each of the seven body parts (elbow, finger, forearm, hand, humerus, shoulder, wrist) in the dataset, resulting in seven part-specific models which are ensembled for inference.
 
 * Training on the entire dataset : `<python train.py --bpart=all>`
 * Training on the "elbow" dataset: `<python train.py --bpart=elbow>`
 
 
 ## Evaluation
 
 * Single model evaluation: `<python evaluate.py --bpart=all>`
 * Ensemble model evaluation: `<python evaluate_ensemble.py --bpart=all>`
 
 
 ## Results
 1. Validation set
    * Single model
    
      Body part | AUROC | Accuracy | Kappa|
      ------------ | ------------- | ------------- | ------------- 
      All  | 0.8625 | 0.8040  | 0.6024 
      
    * Ensemble model
      
      Body part | AUROC | Accuracy | Kappa|
      ------------ | ------------- | ------------- | ------------- 
      All  | 0.8797| 0.8307  | 0.6556
      
 2. Test set
 
    * Single model
    * Ensemble model
    
## Citation

@article{rajpurkar2018mura, 
  title={MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs},
  author={Rajpurkar, Pranav and Irvin, Jeremy and Bagul, Aarti and Ding, Daisy and Duan, Tony and Mehta, Hershel and Yang,      Brandon and Zhu, Kaylie and Laird, Dillon and Ball, Robyn L and others},
  year={2018} 
}


@inproceedings{huang2017densely, 
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
