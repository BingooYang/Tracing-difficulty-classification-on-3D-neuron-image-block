## Deep learning based tracing difficulty classification on 3D neuron image block (pytorch)

### 0. Dataset sourece

Raw experimental data including whole brain data of No. 17545 mouse and No.
17302 mouse and corresponding gold standard reconstruction were provided by
the Southeast University-Allen Institute Joint Center.

### 1. Data processing

Three types of sample data need to be constructed to achieve the tracing difficulty classification: 1. 3D image blocks, gold blocks, and auto blocks; 2. Neuron distance (ND) of gold blocks and auto blocks, and L-Measure (LM) of auto blocks; 3. Annotation data of 3D image blocks. 

### 2. Model

A model called 3D-SSM is designed to classify the tracing difficultyof 3D image blocks, which is based on ResNet, Fully Connected Neural Net-work (FCNN) and Long Short-Term Memory network (LSTM). 3D-SSMconsists of three modules: Structure Feature Extraction (SFE), Sequence Informa-tion Extraction (SIE) and Model Fusion (MF). SFE utilizes a 3D-ResNet and aFCNN to extract two kinds of features in 3D neuron image blocks and automaticreconstruction blocks. SIE uses two LSTMs to extract sequence information hid-den in features of sequential blocks produced in SFE. MF adopts a concatenationoperation and a FCNN to fuse outputs from SIE.

<div align=center>![image](https://github.com/BingooYang/Tracing-difficulty-classification-on-3D-neuron-image-block/blob/main/3D-SSM.PNG)

### 3. Code

```
├── annotation               
│   ├── deal_feature        Extraction and analysis of L-Measure and neuron distance features.
│   ├── automatic_label     Train an FFCNs model and use it to generate automatic labels.
├── classification_model    Models are designed to solve the tracing difficulty classification of 3D image blocks.
├── vaa3d_plugin            Plug-in for getting 3D image blocks, gold blocks, and auto blocks.
├── README.md

```
1. Get 3D image blocks, gold blocks, and auto blocks by **vaa3d_plugin**.
2. Get L-Measure and neuron distance features by **annotation**.
3. To predict the tracing difficulty of 3D image blocks by model file.


