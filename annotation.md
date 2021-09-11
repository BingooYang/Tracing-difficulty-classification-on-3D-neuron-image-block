#### Annotation

Gold blocks and auto blocks are used to generate labels of 3D image blocks. For an automatic tracing algorithm, the tracing difficulty (low or high) on a 3D image block can be determined according to the similarity between corresponding gold block and auto block. If they are very consistent, the 3D image block is labeled as low tracing difficulty (low-TDB), otherwise as high tracing difficulty (high-TDB). According to the following rules, 2954 3D image blocks from brain-A are labeled by one annotator and checked by other two annotators.Above 2954 manually labelled samples are used to train and test a FCNN model for classify the similarity of each gold block and auto block pair.

##### Low-TDBs
1. The gold standard basically overlaps with automatic reconstruction on Vaa3D platform.
2. The gold standard is close to or parallel to automatic reconstruction on Vaa3D platform.
3. The gold standard doesn't have the same number of fragments as automatic reconstruction (caused by a break somewhere in the middle), but they are close.
4. There's more or less reconstruction at the endpoints, but the number of nodes is less than 5.

##### High-TDBs
Other cases that are not low-TDBs.
