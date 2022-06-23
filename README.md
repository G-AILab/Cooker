# Cooker: Self-supervised Adaptive Aggregator Learning on Graph

## Paper

   Self-supervised Adaptive Aggregator Learning on Graph, Pacific-Asia Conference on Knowledge Discovery and Data Mining, 2021
   
   Please cite this paper. 
   
 ```
 @inproceedings{lin2021self,
  title={Self-supervised Adaptive Aggregator Learning on Graph},
  author={Lin, Bei and Luo, Binli and He, Jiaojiao and Gui, Ning},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={29--41},
  year={2021},
  organization={Springer}
}
```
## Usage
Here we provide an implementation of Cooker in Python, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `input/` contains the necessary dataset files for Cora;
- `model.py` contains the implementation of the Cooker pipeline and a random walk layer;
- `utils.py` contains the necessary processing subroutines.
- `main.py` puts all of the above together and may be used to execute a full training run on Cora.

## Installation
+ Requirement
    + `Python==3.6`
    + `numpy==1.18.5`
    + `pandas==1.0.4`
    + `scikit-learn==0.23.1`
    + `tensorflow==2.2.0`
    + `networkx==2.4`
