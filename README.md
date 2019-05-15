# X-ray-classification

Using Inception-ResNet v2 to classify chest X-ray images into normal(healthy) **vs** abnormal(sick)


## Requirements

  python 3.6
  tensorflow = 1.0.1
  matplotlib
  lxml


## Getting started

##### get the data :
In the `data` folder (`cd data/`) :

  1 - Use `scraper.py` to download scrapped image data from [openi.nlm.nih.gov](https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m=1&n=101). It has a large base of Xray,MRI, CT scan images publically available.Specifically Chest Xray Images have been scraped.The images will be downloaded and saved in `images/` and the labels in `data_new.json` (it might take a while)

  Some info about the dataset :
  ```
    Total number of Images : 7469
    The classes with most occurence in the dataset:

    		 ('normal', 2696)
    		 ('No Indexing', 172)
    		 ('Lung/hypoinflation', 88)
    		 ('Thoracic Vertebrae/degenerative/mild', 55)
    		 ('Thoracic Vertebrae/degenerative', 44)
    		 ('Spine/degenerative/mild', 36)
    		 ('Spine/degenerative', 35)
    		 ('Spondylosis/thoracic vertebrae', 33)
    		 ('Granulomatous Disease', 32)
    		 ('Cardiomegaly/mild', 32)
  ```
  2 - Use `python gen_data.py` to sort labels into Normal/Abnormal classes, generate full path to coresponding Images and write them to `data.txt`


  ```
  number of normal chest Images(healthy people) 2696:
  number of abnormal chest Images(sick people) 4773:
  ```

  3 - Use `python convert_to_tf_records.py` to generate tf records of the data.


#### training & evaluation:

  Download the Pre-trained inception model in [here](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz) and unzip it in `ckpt/` folder.

  Use `python train.py` to start the training !(trained model will be saved in `logs/`)

  Use `python evaluate.py` to run evaluation using the model saved in `logs/`(metric : streaming accuracy over all mini batches)

## References

  [Xvision](https://github.com/ayush1997/Xvision)

  [tensorflow.slim](https://github.com/tensorflow/models/tree/master/slim)

  [tuto.transfer learning](https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html)
