# The SAE-NAD model for Point-of-Interest Recommendation
The implementation of the paper:

*Chen Ma, Yingxue Zhang, Qinglong Wang, and Xue Liu, "**Point-of-Interest Recommendation: Exploiting Self-Attentive Autoencoders with Neighbor-Aware Influence**", in the 27th ACM International Conference on Information and Knowledge Management (**CIKM 2018**)* 

Arxiv: https://arxiv.org/abs/1809.10770

**Please cite our paper if you use our code. Thanks!**

Author: Chen Ma (allenmc1230@gmail.com)

**Bibtex**
```
@inproceedings{DBLP:conf/cikm/MaZWL18,
  author    = {Chen Ma and
               Yingxue Zhang and
               Qinglong Wang and
               Xue Liu},
  title     = {Point-of-Interest Recommendation: Exploiting Self-Attentive Autoencoders
               with Neighbor-Aware Influence},
  booktitle = {{CIKM}},
  pages     = {697--706},
  publisher = {{ACM}},
  year      = {2018}
}
```

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

Run the ```cal_poi_pairwise_relation.py``` to calculate the pairwise relations between locations, which will be stored in ```./data/Foursquare/```.

```
python cal_poi_pairwise_relation.py	
```

Train and evaluate the model (you are strongly recommended to run the program on a machine with GPU):

```
python run.py
```
