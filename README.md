# The SAE-NAD model for Point-of-Interest Recommendation
The implementation of the paper:

*Chen Ma, Yingxue Zhang, Qinglong Wang, and Xue Liu, "Point-of-Interest Recommendation: Exploiting Self-Attentive Autoencoders with Neighbor-Aware Influence", in the 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)*

**Please cite our paper if you use our code. Thanks!**

Author: Chen Ma (allenmc1230@gmail.com)

## Environments

- python 3.6
- PyTorch (version: 0.4.0)
- numpy (version: 1.15.0)
- scipy (version: 1.1.0)
- sklearn (version: 0.19.1)


## Dataset

In our experiments, the Foursquare and Yelp datasets are from http://spatialkeyword.sce.ntu.edu.sg/eval-vldb17/. And the Gowalla dataset is from https://snap.stanford.edu/data/loc-gowalla.html (if you need the data after preprocessing, please send me an email).

## Example to run the codes		

Data preprocessing:

Run the *cal_poi_pairwise_relation.py* to calculate the pairwise relations between locations, which is stored in *./data/Foursquare*.

```
python cal_poi_pairwise_relation.py	
```

Train and evaluate the model (you are strongly suggested to run the program on a machine with GPU):

```
python run.py
```
