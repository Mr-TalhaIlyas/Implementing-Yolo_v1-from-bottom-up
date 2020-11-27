Usually object detectors have a very complex pipline like RCNN family of 2-stage detectrs. In two-stage detectors first we pass the iamge throuhg a CNN to get sparce features and then a low level vision algorithm like 'Selective Search' or 'Edge based region detection algorithm' gets the ROIs from those featrue maps. These ROIs are then passed into a second CNN which classifies them accordingly and also regresses their bounding box coordinates in a seperate branch,.
The concept of Anchor boxes makes the understanding even more difficult for new-commers in the machine field. 

So befor diving into those complex and highly accurate OD let's start with the simplest multi-object detector namely YOLO.
In this I try to implement YOLO-v1 from scratch without any pre-training or using any API.

First of all before going furhter I'd like to clearify some thing about **Darknet** which kept me confused for quite a while. **Darknet** is just an term which houses all the CNNs as a back bone feature extractors in YOLO family just like any othe backbone like ResNet, Inception, VGG-16, Hourglass etc.
So, don't get intimidated by the image below.

img1.png

Let's get started 

## Dataset Preparation
First step befor starting the coding is to aquire and process the dataset so let's just dicuss that first and get it out of the way.

### Public Datasets
There are many public datasets availabe you can find them here at RoboFlow (https://public.roboflow.com/), other bench marks are also available like PASCAL VOC, MS COCO etc.
So download the dataset you like and while downloading the dataset select the groung truth (GT) type you wanna download in our case download yolo .txt format. Following are some common data formats used in object detection networks

img2.png

We should be able to convert each datatype in any othe required by anyother in this repo I'll only cover the conversion between .txt and .xml formats

If you have downloaded the data from roboflow (https://public.roboflow.com/) then just put your dataset as shown in data dir in repo.

### Cusotm Datasets
If you have annotated your then it probably will be in .xml format. Or you can make your own detection dataset via labelImg tool for description and usage visit (https://pypi.org/project/labelImg/)
Each image will have one corresponding xml file.
If your data is in .xml format then you can run the ***voc2yolo.py*** script in the repo and give the label names as list and also change the path names where you wanna save the converted GT txt file.

```
# inside voc2yolo.py get these lines.


VOC_class_names = ['aeroplane', 'bicycle', 'bird','boat','bottle', 'bus', 'car',
                 'cat', 'chair', 'cow','diningtable','dog', 'horse', 'motorbike',
                 'person','pottedplant','sheep', 'sofa', 'train', 'tvmonitor']

images_dir = 'D:/pascal_voc/'

xml_filepaths = glob.glob( os.path.join( images_dir , '*.xml' ) )
```


This script which actually has three sections the 1st one will convert the all .xml files to one .txt file.

The second part starting with **Train_Val split** will raondmly split the data into train and val you can select the number of val data to be selected by editing,
```
In [95]: val_data = 4000
```

The third part starting form ***Train_Val Write*** will read the newly created .txt files and will seperate the images into train / val folders, by reading the names of the images form the newly created train/val .txt files.
You can give the desired location bu=y editin following line,
```
In [124]: with open('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/train.txt', 'r') as f:
```

Afer running the script you'll have one .txt file for all the images in train set and one .txt file for val set. The txt file will look something like

```
2007_000027.jpg 174,101,349,351,14
2007_000032.jpg 104,78,375,183,0 133,88,197,123,0 195,180,213,229,14 26,189,44,238,14
```
One line in txt file has annotations for one image and each entry is seperated by space, e.g. look at frist line we 

1st we have image name withe it's extension **i.e.** 2007_000027.jpg
2nd we have 4 coordinates in order **i.e.** (x_min, y_min, x_max, y_max) -> (174,101,349,351)
3rd we have the integer specifing the class name as given in the list (VOC_class_names) while covnverting the data form xml to txt **i.e.** 14 -> person

if we have more than one instance in the image then after a space the same oreder continues as **2nd and 3rd**

## Making YOLO tensor
