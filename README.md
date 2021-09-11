## Deep learning based tracing difficulty classification on 3D neuron image block (pytorch)

### 0. Dataset sourece

Raw experimental data including whole brain data of No. 17545 mouse and No.
17302 mouse and corresponding gold standard reconstruction were provided by
the Southeast University-Allen Institute Joint Center.

### 1. Data processing

Three types of sample data need to be constructed to achieve the tracing difficulty classification: 1. 3D image blocks, gold blocks, and auto blocks; 2. Neuron distance (ND) [28] of gold blocks and auto blocks, and L-Measure (LM) [29] of auto blocks; 3. Annotation data of 3D image blocks. 

### 2. Model

3D-SSM mainly consists of three parts: SFE, SIE and MF. First of all, 3D-ResNet and FFCNs in SEF are trained by using 3D image blocks and LM-32 of corresponding auto blocks, and their parameters are saved. Then, LSTM in SIE is used to extract the sequence information hidden in blocks, and the network parameters are saved, as well. Finally, the output features of SIE are fused by concatenate and FFCNs, and the SFE, SIE and MF are trained together.

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


