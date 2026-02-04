# **RoofX-Net: A Tailored Approach to Accurate Multi-Type Rooftop Segmentation in Remote Sensing Images Using Edge and Scale Awareness**

## **Overview**
This repository contains the implementation of RoofX-Net model for Multi-Type Rooftop Segmentation in Remote Sensing Images

## **Requirements**
### **Hardware**
- **GPU**: NVIDIA GPU with CUDA support (e.g., NVIDIA RTX 4090 or higher).

### **Software**
- **Python**: 3.8.10.
- **Libraries**:
  Please refer to ./requirements.txt

Install dependencies using:
```bash
pip3 install -r requirements.txt
```

## **Dataset**
### **Rooftop+ Dataset**
The Rooftop+ dataset can access available from the corresponding authors upon reasonable request.

### **WHU Dataset**
The WHU Dataset is publicly available [here](https://gpcv.whu.edu.cn/data/building_dataset.html). 


## **Training**
Train the model using the command in `./train.sh`:
```bash
bash train.sh
```

## **Evaluation**
Evaluate the model performance using the command in `./test.sh`:
```bash
bash test.sh
```

## **Visualization**
Perform inference on new images using the command in `./vis.sh`:
```bash
bash vis.sh
```
