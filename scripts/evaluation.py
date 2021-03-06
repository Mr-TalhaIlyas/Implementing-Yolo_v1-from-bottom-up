import numpy as np
import os
import xml.etree.cElementTree as ET
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm
import xmltodict
import numpy as np
import glob
import os, copy
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# predictedFolder = 'D:/Anaconda/Datasets/pascal_voc/VOC2012/Annotations_XML'
# groundTruthFolder = 'D:/Anaconda/Datasets/pascal_voc/VOC2012/Annotations_XML'

# numOfClasses = 20


# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
#             "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
#             "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# IoUThreshold = 0.5


def IoU(boxA, boxB):
    intersectionX = max(0, min(boxA[1], boxB[1]) - max(boxA[0], boxB[0]))
    intersectionY = max(0, min(boxA[3], boxB[3]) - max(boxA[2], boxB[2]))
    intersection = intersectionX * intersectionY
    union = (boxA[1] - boxA[0]) * (boxA[3] - boxA[2]) + \
        (boxB[1] - boxB[0]) * (boxB[3] - boxB[2]) - intersection
    # print(intersection, union, intersection * 1.0 / union)
    try:
        iou = intersection * 1. / union
    except ZeroDivisionError:
        iou = 0
    return iou



def calculate_mAP(groundTruthFolder, predictedFolder, numOfClasses, classes, IoUThreshold):
    # Init
    fileListGT = os.listdir(groundTruthFolder)
    fileListPredicted = os.listdir(predictedFolder)
    
    dictPredicted={}
    
    # Each index of values in key:value pair would consist
    # of [confidence, image_name, [x,y,w,h]],key:class 
    for classId in range(numOfClasses):
        dictPredicted[classId]=[]
    
    # total numbers of objects predicted can help in calcu
    # - lating recall as it equals TP + FN 
    totalPredicted = np.zeros(numOfClasses, dtype=int)
    totalGT = np.zeros(numOfClasses, dtype=int)
    
    for file in fileListPredicted:
        
        predictedFilePath = os.path.join(predictedFolder, file)
        predictedObject = ET.parse(predictedFilePath).findall('object')
        
        for item in predictedObject:
            itemClass = item.find('name').text
            classId = classes.index(itemClass)
            confidence = float(item.find('confidence').text)
            xmin = int(float(item.find('bndbox').find('xmin').text))
            xmax = int(float(item.find('bndbox').find('xmax').text))
            ymin = int(float(item.find('bndbox').find('ymin').text))
            ymax = int(float(item.find('bndbox').find('ymax').text))
             
    
            dictPredicted[classId].append([confidence,
                file,
                [xmin, xmax, ymin, ymax]])
    
            totalPredicted[classId]+=1
    
    # for each predicted box, sort according to confidence   
    for classId in range(numOfClasses):
        dictPredicted[classId].sort(key=lambda x: x[0], reverse=True)
    
    # dictionary of dictionary, key: class, nested key : file
    # eg { 'car' : {000001.xml: [[x,y,w,h],[a,b,c,d]], '0000002.xml': [] } }
    dictGT = {}
    dictMask = {}
    
    for classId in range(numOfClasses):
        dictGT[classId] = {}
        dictMask[classId] = {}
        for file in fileListGT:
            dictGT[classId][file] = []
            dictMask[classId][file] = []
    
    for file in fileListGT:
    
        GTFilePath = os.path.join(groundTruthFolder, file)
        groundTruthObject = ET.parse(GTFilePath).findall('object')
     
        for item in groundTruthObject:
            itemClass = item.find('name').text
            classId = classes.index(itemClass)
            xmin = int(float(item.find('bndbox').find('xmin').text))
            xmax = int(float(item.find('bndbox').find('xmax').text))
            ymin = int(float(item.find('bndbox').find('ymin').text))
            ymax = int(float(item.find('bndbox').find('ymax').text))
            # Append dictGT
            dictGT[classId][file].append([xmin, xmax, ymin, ymax])
            # To find out if a ground truth exists for an object of a class,
            # and if a prediction has been made corresponding to that object,
            # we shall append dictMask with 0 if GT exists,
            # and modify it to 1 when a prediction corresponds with it
            dictMask[classId][file].append(False)
            totalGT[classId]+=1
     
    truePositives = []
    falsePositives = []
    falseNegatives = np.zeros(numOfClasses)
       
    for classId in range(numOfClasses):
        numberOfPredictedObjectsInClass = totalPredicted[classId]
        truePositives.append(np.zeros(numberOfPredictedObjectsInClass,
            dtype=np.float64))
        falsePositives.append(np.zeros(numberOfPredictedObjectsInClass,
            dtype=np.float64))
    
        for predictedObjectIndex in range(len(dictPredicted[classId])):
        # To find the ground truth bounding box corresponding with the
        # predicted bounding box
    
            predictedItem = dictPredicted[classId][predictedObjectIndex]
            maxIoU = 0.0
            maxIndex = -1
            # If no item of classId predicted is present in the ground truth image
            if len(dictGT[classId][predictedItem[1]])==0:
                falsePositives[classId][predictedObjectIndex]=1
                continue
             # For each ground truth box            
            for GTObjectIndex in  range(len(dictGT[classId][predictedItem[1]])):
                # If particular GTbox has already been alloted to predicted box
                # move to the next box without considering it
                if dictMask[classId][predictedItem[1]][GTObjectIndex]==True:
                    continue
                GTItem = dictGT[classId][predictedItem[1]]
                areaMetric = IoU(GTItem[GTObjectIndex],
                                 predictedItem[2])
    
                # Record that GT bounding box which has maximum IoU with
                # the predicted bounding box
                if areaMetric > maxIoU:
    
                    maxIoU = areaMetric
                    maxIndex = GTObjectIndex
            
            # If all the GT box in a particular image are already alloted
            # to predictedBox, add the new predictedBox to fP
            if maxIndex==-1:
                falsePositives[classId][predictedObjectIndex]=1
                continue
    
            if maxIoU > IoUThreshold:
    
                # If the object has not been detected before
                if dictMask[classId][predictedItem[1]][maxIndex]==False:
                    
                    # Modify dictMask to indicate that object has been 
                    # detected
                    dictMask[classId][predictedItem[1]][maxIndex]=True
                    truePositives[classId][predictedObjectIndex]=1
                    
                    # predictObject has been attributed to GT Object
                    # move to next predictedObject
                else:   
                    
                    # Else if object has been detected before, since we know
                    # that the current prediction has lesser confidence than the
                    # previous one, so we will consider this a false positive
                    falsePositives[classId][predictedObjectIndex]=1
            else:
                falsePositives[classId][predictedObjectIndex]=1
    cumulativePrecision = []
    cumulativeRecall = []
    averagePrecision = np.zeros(numOfClasses)
    
    # For each class calculate Interpolated Average Precision
    # as given in PASCAL VOC handbook
    for classId in range(numOfClasses):
    
        for image in dictMask[classId]:
            falseNegatives[classId]+=sum([ not x for x in
                dictMask[classId][image]])
        cumulativePrecision.append(np.divide(np.cumsum(truePositives[classId]),
            np.cumsum(truePositives[classId])+np.cumsum(falsePositives[classId])))
        #cumulativePrecision.append(np.divide(np.cumsum(truePositives[classId]),
            #1 + np.arange(totalPredicted[classId])))
    
        cumulativeRecall.append(np.cumsum(truePositives[classId])/totalGT[classId])
    
        classPrecision = np.asarray(cumulativePrecision[-1], dtype=np.float64)
        classRecall = np.asarray(cumulativeRecall[-1], dtype=np.float64)
        for threshold in range(0,11,1):
            threshold = (threshold/10.0)
    
            # Get the maximum precision above a particular recall value 
            precisionValues = (classPrecision[classRecall>=threshold])
    
            # If precision is 0 for all the values 
            if precisionValues.shape[0]==0:
                p=0
            # Otherwise store the maximum
            else:
                p = np.amax(precisionValues)
    
            # Average precision would be mean of the precision values 
            # taken at these 11 points ( according to VOC handbook)
            averagePrecision[classId]+=(p/11)
    
    meanAveragePrecision = np.mean(averagePrecision)
    headers = [ "Class_Name",  
                "Total GT", 
                "Total Pred.", 
                "TP", 
                "FP",
                "FN",
                "AP"]
    table = []
    for classId in range(numOfClasses):
        table.append(   [classes[classId],
                        totalGT[classId],
                        len(dictPredicted[classId]),
                        np.sum(truePositives[classId]),
                        np.sum(falsePositives[classId]),
                        falseNegatives[classId],
                        averagePrecision[classId]])
    
    print(tabulate(table, headers, tablefmt="github"))
    print ("Mean Average Precision : %.3f" % meanAveragePrecision)

#calculate_mAP(groundTruthFolder, predictedFolder)
#%%
'''
This code is adapted from abvoe links to plot a confusion matrix  for object detection
model for details kinly visit them

Kindly read README.md for details.
'''
    
def box_iou_calc(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min = 0, a_max = None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD = 0.3, IOU_THRESHOLD = 0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
    
    def process_batch(self, detections, labels):
        '''
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        '''
        detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        gt_classes = labels[:, 0].astype(np.int16)
        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])
        
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0: # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index = True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index = True)[1]]


        for i, label in enumerate(labels):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                gt_class = gt_classes[i]
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[(gt_class), detection_class] += 1
            else:
                gt_class = gt_classes[i]
                self.matrix[(gt_class), self.num_classes] += 1
        
        for i, detection in enumerate(detections):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.matrix[self.num_classes ,detection_class] += 1
            

def plot_confusion_matrix(cm, class_names, normalize = True, show_text = True, show_fpfn = False):
    
    conf_mat = cm
    if normalize:
        row_sums = conf_mat.sum(axis=1)
        conf_mat = conf_mat / row_sums[:, np.newaxis]
        conf_mat = np.round(conf_mat, 3)
        
    if show_fpfn:
        conf_mat = conf_mat
        x_labels = copy.deepcopy(class_names)
        y_labels = copy.deepcopy(class_names)
        x_labels.append('FN')
        y_labels.append('FP')
    else:
        conf_mat = conf_mat[0:cm.shape[0]-1, 0:cm.shape[0]-1]
        x_labels = class_names
        y_labels = class_names
    my_cmap = 'CMRmap'# viridis, seismic, gray, ocean, CMRmap, RdYlBu, rainbow, jet, Blues, Greens, Purples
    

    c_m = conf_mat 
    
    
    
    print('*'*80)
    print('NOTE: In confusion_matrix the last coloumn "FP/FN" shows False Positives in Groundtruths \
          \nand False Negatives in Predictions')
    print('*'*80)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(c_m, cmap = my_cmap) 
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")#ha=right
    
    # Loop over data dimensions and create text annotations.
    def clr_select(i, j):
        if i==j:
            color="green"
        else:
            color="red"
        return color
    if show_text:
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                text = ax.text(j, i, c_m[i, j], color="k", ha="center", va="center")#color=clr_select(i, j)
    
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm)
    plt.show() 
    return fig     

def read_txt(txt_file, pred=True):
    '''
    Parameters
    ----------
    txt_file : txt file path to read
    pred : if your are raedinf prediction txt file than it'll have 5 values 
    (i.e. including confdience) whereas GT won't have confd value. So set it
    to False for GT file. The default is True.
    Returns
    -------
    info : a list haing 
        if pred=True => detected_class, confd, x_min, y_min, x_max, y_max
        if pred=False => detected_class, x_min, y_min, x_max, y_max
    '''
    x = []
    with open(txt_file, 'r') as f:
        info = []
        x = x + f.readlines()
        for item in x:
            item = item.replace("\n", "").split(" ")
            if pred == True:
                # for preds because 2nd value in preds is confidence
                det_class = item[0]
                confd = float(item[1])
                x_min = int(item[2])
                y_min = int(item[3])
                x_max = int(item[4])
                y_max = int(item[5])
                
                info.append((x_min, y_min, x_max, y_max, confd, det_class))
                            
            else:
                # for preds because 2nd value in preds is confidence
                det_class = item[0]
                x_min = int(item[1])
                y_min = int(item[2])
                x_max = int(item[3])
                y_max = int(item[4])
                
                info.append((det_class, x_min, y_min, x_max, y_max))
                
        return info
    
def IoU(target_boxes , pred_boxes):
    xA = np.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = np.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = np.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = np.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = np.maximum(0.0, xB - xA ) * np.maximum(0.0, yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / ( boxAArea + boxBArea - interArea )
    iou = np.nan_to_num(iou)
    return iou    

def process(x, class_names, gt = True):
    '''
    Parameters
    ----------
    x : class_name, x_min, y_min, x_max, y_max
    Returns
    -------
    x : class_index, x_min, y_min, x_max, y_max
    '''
    if gt:
        clas = x[:,0]
        temp = []
        for i in range(len(clas)):
            temp.append(class_names.index(clas[i]))
        temp = np.array(temp)
        x[:,0] = temp
        x = x.astype(np.int32)
    else:
        clas = x[:,-1]
        temp = []
        for i in range(len(clas)):
            temp.append(class_names.index(clas[i]))
        temp = np.array(temp)
        x[:,-1] = temp
        x = x.astype(np.float32)
    
    return x