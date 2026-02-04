# **RoofX-Net: A Tailored Approach to Accurate Multi-Type Rooftop Segmentation in Remote Sensing Images Using Edge and Scale Awareness**

## **Overview**
This repository provides the implementation of the RoofX-Net model, a solution designed for **multi-type rooftop segmentation** in remote sensing images. RoofX-Net leverages edge and scale awareness to achieve precise segmentation results.

## **Requirements**
### **Hardware**
- **GPU**: NVIDIA GPU with CUDA support (e.g., NVIDIA RTX 4090 or higher).

### **Software**
- **Python**: 3.9.10.
- **Torch**: 1.13.1+cu117.
- **Dependencies**:
  All required libraries are listed in `requirements.txt`.

To install the dependencies, run the following command:
```bash
pip3 install -r requirements.txt
```

## **Dataset**
### **Rooftop+ Dataset**
The Rooftop+ dataset is available upon reasonable request from the corresponding [authors](mailto:cwy@uestc.edu.cn).

### **WHU Dataset**
The WHU Dataset is publicly accessible and can be downloaded [here](https://gpcv.whu.edu.cn/data/building_dataset.html). 


## **Training**
To train the RoofX-Net model, use the provided script: `./train.sh`:
```bash
bash train.sh
```

## **Evaluation**
To evaluate the performance of the trained model, run the following command: `./test.sh`:
```bash
bash test.sh
```

## **Visualization**
For inference and visualization on new images, execute the following script: `./vis.sh`:
```bash
bash vis.sh
```
