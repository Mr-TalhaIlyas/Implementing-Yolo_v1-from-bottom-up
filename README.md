# Getting Started
Usually object detectors have a very complex pipline like RCNN family of 2-stage detectrs. In two-stage detectors first we pass the image throuhg a CNN to get sparce features and then a low level vision algorithm like 'Selective Search' or 'Edge based region detection algorithm' gets the ROIs from those featrue maps. These ROIs are then passed into a second CNN which classifies them accordingly and also regresses their bounding box coordinates in a seperate branch.The concept of Anchor boxes makes the understanding even more difficult for new-commers in the machine field. 

So befor diving into those complex and highly accurate OD let's start with the simplest multi-object detector namely YOLO.
In this I try to implement YOLO-v1 from scratch without any pre-training or using any API. The key reason for working on this was to establish a concrete understanding of keyconcepts of object detection, for future works on object detection. High precision, then, is not the end game here.

First of all before going furhter I'd like to clearify some thing about **Darknet** which kept me confused for quite a while. **Darknet** is just a term which houses all the CNNs architecture that serve as a back bone feature extractors in YOLO family just like any othe backbone like ResNet, Inception, VGG-16, Hourglass etc.
So, don't get intimidated the term and the image below.

![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/img1.png?raw=true)

Let's get started 

## Dataset Preparation
First step befor starting the coding is to aquire and process the dataset so let's just dicuss that first and get it out of the way.

### Public Datasets
There are many public datasets availabe you can find them here at [RoboFlow](https://public.roboflow.com/), other bench marks are also available like PASCAL VOC, MS COCO etc.
So download the dataset you like and while downloading the dataset select the groung truth (GT) type you wanna download in our case download yolo `.txt` format. Following are some common data formats used in object detection networks

![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/img2.png?raw=true)

We should be able to convert each datatype in any othe required by anyother in this repo I'll only cover the conversion between .txt and .xml formats

If you have downloaded the data from [roboflow](https://public.roboflow.com/) then just put your dataset as shown in data dir in repo.

### Cusotm Datasets
If you have annotated your then it probably will be in .xml format. Or you can make your own detection dataset via labelImg tool for description and usage [visit](https://pypi.org/project/labelImg/)
Each image will have one corresponding xml file.
If your data is in .xml format then you can run the ```voc2yolo.py``` script in the repo and give the label names as list and also change the path names where you wanna save the converted GT txt file.

```python
# inside voc2yolo.py get these lines.


VOC_class_names = ['aeroplane', 'bicycle', 'bird','boat','bottle', 'bus', 'car',
                 'cat', 'chair', 'cow','diningtable','dog', 'horse', 'motorbike',
                 'person','pottedplant','sheep', 'sofa', 'train', 'tvmonitor']

images_dir = '/../pascal_voc/'

xml_filepaths = glob.glob( os.path.join( images_dir , '*.xml' ) )
```


This script which actually has three sections the 1st one will convert the all .xml files to one .txt file.

The second part starting with **Train_Val split** will raondmly split the data into train and val you can select the number of val data to be selected by editing,
```
In [95]: val_data = 4000
```

The third part starting form ***Train_Val Write*** will read the newly created .txt files and will seperate the images into train / val folders, by reading the names of the images form the newly created train/val `.txt` files.
You can give the desired location bu=y editin following line,
```
In [124]: with open('/../Desktop/paprika_label 20.08.25/paparika/train.txt', 'r') as f:
```

Afer running the script you'll have one `.txt` file for all the images in train set and one .txt file for val set. The txt file will look something like

```
2007_000027.jpg 174,101,349,351,14
2007_000032.jpg 104,78,375,183,0 133,88,197,123,0 195,180,213,229,14 26,189,44,238,14
```
One line in txt file has annotations for one image and each entry is seperated by space, e.g. look at frist line we 

1st we have image name withe it's extension **i.e.** 2007_000027.jpg
2nd we have 4 coordinates in order **i.e.** (x_min, y_min, x_max, y_max) -> (174,101,349,351)
3rd we have the integer specifing the class name as given in the list (VOC_class_names) while covnverting the data form xml to txt **i.e.** 14 -> person

if we have more than one instance in the image then after a space the same oreder continues as **2nd and 3rd**

## Making YOLO Data Generator

The the built-in data generators provided by the libraries like tf/keras won't be able to read and generate the ground truths in case of object detectors. Because those codes are written for reading the classification data or pandas dataframes (here input is images and output of a network is a label for classification). The mose they can do is to generate the semantic segmentation data
if you create two instances of keras image data generator, one for images and one for masks (where input and ouput of network is images)

In case of object detection like in yolo here we the input to network is an image and the output is a ```7x7x30``` tensor. So first we'll create a class which will read the data form the data_dir and will convert the labels written in text file into a format suitable for our OD netowrk pipline. In this case our 
custom data generator should convert the lines in .txt files to a ```7x7x30``` tensor as explained in yolo paper (for pascal voc dataset). You can also read a good artical on yolo [here](https://medium.com/@amrokamal_47691/yolo-yolov2-and-yolov3-all-you-want-to-know-7e3e92dc4899)

As yolo divides the image into a 7x7 grid so we need to decide where in the 7x7 grid to put the values. See the image below for visual explaination.

![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/Slide1.JPG?raw=true)

The coords need to be converted form (x_min, y_min, x_max, y_max)  to (x_center, y_center, w, h) format as expalined in original paper. Also according to authors;

* the center coordinated (x, y) should be normalized via grid size.
* while the size of box (w, h) should be normalized via original image size.

this is important.
The follwing codes does the same process after reading the images and data of corresponding image from .txt file and makes a yolo tensor with proper offsets.
```python
def make_yolo_tensor(img_name, line, modelip_img_w, modelip_img_h, grid_size, num_class):
    
    img = cv2.imread(img_name)
    orig_img_h, orig_img_w, _ = img.shape
    img = cv2.resize(img, (modelip_img_w, modelip_img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    b_c_all = line # B_box co-ordinates
    label_matrix = np.zeros([grid_size, grid_size, num_class+5])
    for j in range(len(b_c_all)):
        b_c = np.asarray(b_c_all[j].split(','), np.uint)
        x_min, y_min, x_max, y_max, class_index = b_c
        # if the resolution in which you labelles image is different from model_ip
        # then we need to scale th b_boxes to new resolution (i.e. model_ip)
        if modelip_img_w != orig_img_w or modelip_img_h != orig_img_h:
            x_min, x_max = [i * (modelip_img_w / orig_img_w) for i in (x_min, x_max)]
            y_min, y_max = [i * (modelip_img_h / orig_img_h) for i in (y_min, y_max)]
        # This gives center_point(x,y) and (w,h) normalized corresponding to image hight and width 
        x, y, w, h = xyminmax_2_xywh(modelip_img_w, modelip_img_h, box = np.array([x_min, y_min, x_max, y_max]))
        # Also find in which grid cell to put the values of b_box co-ordinates and confidences
        loc = [grid_size * x, grid_size * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        
        # offset w.r.t the gridcell
        # verified via github
        y_offset = loc[1] - loc_i
        x_offset = loc[0] - loc_j
        # b/c small deviations in large boxes matter less as compared to small boxes
        # wSqrt = math.sqrt(w)
        # hSqrt = math.sqrt(h)
        
        if label_matrix[loc_i, loc_j, num_class+4] == 0:
            
           label_matrix[loc_i, loc_j, class_index] = 1
           label_matrix[loc_i, loc_j, num_class:num_class+4] = [x_offset, y_offset, w, h]
           label_matrix[loc_i, loc_j, num_class+4] = 1  
    
    return img, label_matrix
```

One thing to note here is that in original paper the op tenosr is of shape 7x7x30,

![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/Slide2.JPG?raw=true)

but here we just made a tensor of 7x7x25.

![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/Slide3.JPG?raw=true)

The authors of paper decided to detect use two boxes in predictions instead of one so that they could detect object of different sizes and aspect ratios, because they aren't using anchore boxes in version one. So this dimension mismatch will be solved automatically while we are calculating the loss the following image will give you a basic idea.

![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/Slide4.JPG?raw=true)

Now that we can generate a trainable batch of iamge and it's label, all that's left to do is to pass this function to the keras `Sequence` calss and call it at each epoch to give us a batch of new data you can see the details in `YOLO_DataGenerator.py` .
 

## Non-Maximum-Suppression

Unlike the other CNNs (e.g. classification and semantic segmentation) we can't utilize the ouptput of an OD directly we have to apply NMS to suppress the less probable ouputs a very good slides are provided by the google you can go through them [here](https://www.slideshare.net/OhYoojin/you-only-look-once-unified-realtime-object-detection-100657954)
So I have wrote two different versions of NMS both does the same thing and there ouput is also same the only difference is the output of the functions.

### ***nms_v1***

```python
nms_v1(pred, iou_thresh, confd_thersh, grid_size, num_class, boxes_percell, modelip_img_w, modelip_img_h)
```

* this fucntions takes the yolo op tensor as input and also outputs a tensor of same shape
* it's running time is about 0.001 second

### ***nms_v2***

```python
nms_v2(pred, iou_thresh, confd_thersh, grid_size, num_class, classes_name, boxes_percell, modelip_img_w, modelip_img_h, use_numpy=True)
```

* this fucntions takes the yolo op tensor as input but outputs only the list of NMS boxes
* it's running time is about 0.0001 second (10 times faster approx)

## Training

I have used following backbones to train the yolo_v1, in contrast to original paper where only the *Darknet* is used.

* Xception Net
* MobileNet v2
* Darknet

I have trained the netowrk using Xception as back bone and the weighta are also provided in repo. The results are not as good as explained in the paper because I didn't pretrain the network on `ImageNet` first, also I didn't
follow their training reoutine to letter may be that's the reason. But the main reason of working on this was to understand the main idea behind the object detectors working. Next time I'll try to make some more accurate detectors like
Faster-RCNN, Efficent-Det etc.

## Results

### Visual Results

![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/op.jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/op%20(1).jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/op%20(2).jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/op%20(3).jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/op%20(4).jpg?raw=true)
![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/op%20(5).jpg?raw=true)

### PASCAL-VOC (mAP)

To understand this metric there is a very good repo can check out [here](https://github.com/rafaelpadilla/Object-Detection-Metrics)

```
| Class_Name   |   Total GT |   Total Pred. |   TP |   FP |   FN |       AP |
|--------------|------------|---------------|------|------|------|----------|
| aeroplane    |        216 |           153 |  135 |   18 |   81 | 0.619045 |
| bicycle      |        149 |            45 |   43 |    2 |  106 | 0.270613 |
| bird         |        219 |           138 |  113 |   25 |  106 | 0.509256 |
| boat         |        113 |            46 |   25 |   21 |   88 | 0.207459 |
| bottle       |        174 |            28 |   22 |    6 |  152 | 0.170455 |
| bus          |        109 |            54 |   52 |    2 |   57 | 0.447046 |
| car          |        405 |           168 |  123 |   45 |  282 | 0.319196 |
| cat          |        207 |           197 |  176 |   21 |   31 | 0.784733 |
| chair        |        298 |            48 |   41 |    7 |  257 | 0.173554 |
| cow          |         71 |            47 |   32 |   15 |   39 | 0.385995 |
| diningtable  |         89 |            18 |   13 |    5 |   76 | 0.131772 |
| dog          |        231 |           174 |  160 |   14 |   71 | 0.616674 |
| horse        |        129 |            50 |   44 |    6 |   85 | 0.337143 |
| motorbike    |        134 |            54 |   46 |    8 |   88 | 0.329522 |
| person       |       2635 |          1645 | 1313 |  332 | 1322 | 0.392537 |
| pottedplant  |         98 |            11 |   10 |    1 |   88 | 0.173554 |
| sheep        |        200 |            90 |   59 |   31 |  141 | 0.230769 |
| sofa         |         82 |            15 |   13 |    2 |   69 | 0.169697 |
| train        |         98 |            69 |   61 |    8 |   37 | 0.619555 |
| tvmonitor    |         95 |            43 |   37 |    6 |   58 | 0.348622 |
``` 

### COnfusion Matrix

For details on making the confucion matrix, you can visit my repo [here](https://github.com/Mr-TalhaIlyas/Confusion_Matrix_for_Objecti_Detection_Models)

![alt text](https://github.com/Mr-TalhaIlyas/Implementing-Yolo_v1-from-bottom-up/blob/master/images/CM.png?raw=true)

## References

Following repos helped me alot in understanding the workings of OD

* [lovish1234](https://github.com/lovish1234/YOLOv1)
* [Vivek Maskara](https://www.maskaravivek.com/post/yolov1/)
* [FMsunyh](https://github.com/FMsunyh/keras-yolo)

