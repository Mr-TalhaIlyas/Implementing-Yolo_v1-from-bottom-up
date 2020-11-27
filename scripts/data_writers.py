import xml.etree.ElementTree as gfg
from xml.dom import minidom
import xmltodict
import numpy as np
import cv2
import os
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['figure.dpi'] = 300

from data_processors import IoU, xyminmax_2_xywh, xywh_2_xyminmax

#%%     
def show_results(img_in, yolo_op_tensor, classes_name, img_w, img_h, from_nms=True):
    '''
    Parameters
    ----------
    img : RGB images
    yolo_op_tensor : processed tensor of yolo after NMS or the ground turth tensor
                    this function will digest the yolo_tensor (7,7,25) and use draw_boxes
                    function to draw the b_boxes on image
    '''
    img = img_in
    modelip_img_w, modelip_img_h = img_w, img_h
    num_class = len(classes_name)
    gt = yolo_op_tensor
    class_probab = gt[:,:,0:num_class] # S * S * (num_class+5) => S * S * num_class
    s = class_probab.shape[0]
    # because for tensors coming from nms are already correcting offset inside nms
    if from_nms == False:
        ###### 
        # Removing grid offset
        # verified via github https://github.com/lovish1234/YOLOv1/blob/master/preprocess.py
        # in case of gt/nms tensor each grid cell will contain only one b_box so need for loop.
        offset_x = np.tile(np.arange(0,s), s).reshape(s,s)
        offset_y = offset_x.T
        gt[:,:,num_class] = np.where(gt[:,:,num_class] != 0, gt[:,:,num_class] + offset_x, 0) / s
        gt[:,:,num_class+1] = np.where(gt[:,:,num_class+1] != 0, gt[:,:,num_class+1] + offset_y, 0) / s
        # taking square of w and h to revrese the conversions we did for loss calculations
        # gt[:,:,num_class+2] = np.square(gt[:,:,num_class+2])
        # gt[:,:,num_class+3] = np.square(gt[:,:,num_class+3])
        ######
    row_idices = []
    col_idices = []
    obj_coloumns = []
    labels = []
    
    for i in range(s):
        for j in range(s):
            for k in range(num_class):
                if class_probab[i,j,k] > 0:
                    row_idices.append(i)
                    col_idices.append(j)
    
    for i in range(len(row_idices)):
        obj_coloumns.append(gt[row_idices[i], col_idices[i], :])
    obj_coloumns = np.reshape(obj_coloumns, (len(row_idices), -1))
    
    class_probab = obj_coloumns[:, 0:num_class]
    class_det = np.argmax(obj_coloumns[:, 0:num_class], axis = 1)
    b_box = obj_coloumns[:, num_class:num_class+4]
    obj_score = obj_coloumns[:, -1:]
    
    confidence_score = np.max(np.multiply(class_probab, obj_score),axis=1)
    
    for i in range(len(class_det)):
        labels.append(classes_name[class_det[i]])
    boxes_c = []
    for i in range(len(class_det)):
        boxes_c.append(xywh_2_xyminmax(modelip_img_w, modelip_img_h, b_box[i]))
    boxes_c = (np.asarray(boxes_c)).astype(np.uint32)
    
    op = draw_boxes(image_in=img, confidences=confidence_score, 
                     boxes=boxes_c, class_names= labels, all_classes = classes_name)
    
    return op

#%%  
    
def draw_boxes(image_in, confidences, boxes, class_names, all_classes):
    '''
    Parameters
    ----------
    image : RGB image
    confidences : confidence scores array, shape (None,)
    boxes : all the b_box coordinates array, shape (None, 4)
    classes : shape (None,), names  of classes detected
    '''
    image = image_in
    i = 1
    colors = sns.color_palette("bright")
    for result in zip(confidences, boxes, class_names, colors):
        conf = float(result[0])
        facebox = result[1].astype(np.int16)
        #print(facebox)
        name = result[2]
        color = result[3]
    
        
        cv2.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), color, 2)#255, 0, 0
        label = '{0}: {1:0.3f}'.format(name.strip(), conf)
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX   , 0.5, 1)

        cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),    # top left cornor
                     (facebox[0] + label_size[0], facebox[1] + base_line),# bottom right cornor
                     color, cv2.FILLED)#0, 0, 255
        op = cv2.putText(image, label, (facebox[0], facebox[1]),
                   cv2.FONT_HERSHEY_DUPLEX   , 0.5, (0, 0, 0)) 
        i = i+1
    return op
#%%
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = gfg.tostring(elem, 'utf-8')  # form byte to string 
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding = 'utf-8') # form string to byte
#%%
def write_xml(image_address, scores, boxes, classes, op_dir):
    '''
    Parameters
    ----------
    image_address : address for reading images to get hight and width for coord scaling
    scores : list of n elements
    boxes : list of (n x 4) elements, normalized in xywh format
    classes : list of n elements
    xml_dir : location where to save .xml file

    Returns
    -------
    None.
    '''
    # the inputs to this functio are scaled according to the model_ip width and hight
    # but while writing .xml files we need to write the coordinates w.r.t original 
    # image so normalize them again and then rescale them according to original image dimensions
    img = cv2.imread(image_address)
    h, w,_ = img.shape
    
    file_name = os.path.basename(image_address)[:-4]
    boxes_scaled = []
    for i in range(boxes.shape[0]):
        boxes_scaled.append(xywh_2_xyminmax(w, h, boxes[i]))
    boxes_scaled = (np.asarray(boxes_scaled)).astype(np.uint32)
    
    root = gfg.Element("annotation") 
    #element
    e1 = gfg.Element("filename") 
    e1.text = str(file_name)
    root.append (e1) 
    for i in range(len(classes)):
        e2 = gfg.Element('object')
        root.append(e2)
        #sub-element
        se1 = gfg.SubElement(e2, 'name')
        se1.text = str(classes[i])
        se2 = gfg.SubElement(e2, 'confidence')
        se2.text = str(scores[i])
        se3 = gfg.SubElement(e2, 'bndbox')
        #sub-sub-element
        sse1 = gfg.SubElement(se3, 'xmin')
        sse1.text = str((boxes_scaled[i][0]).astype(np.int16))
        sse2 = gfg.SubElement(se3, 'ymin')
        sse2.text = str((boxes_scaled[i][1]).astype(np.int16))
        sse3 = gfg.SubElement(se3, 'xmax')
        sse3.text = str((boxes_scaled[i][2]).astype(np.int16))
        sse4 = gfg.SubElement(se3, 'ymax')
        sse4.text = str((boxes_scaled[i][3]).astype(np.int16))
        
    xml_path = op_dir + file_name +'.xml'
    root = prettify(root)
    
    with open (xml_path, "wb") as files : 
       files.write(root)
       
     
#%%       
def yolotensor2xml(image_address, yolo_nms_tensor, grid_size, classes_name, num_class, op_dir):
    '''
    Parameters
    ----------
    image_address : address for reading images to get hight and width for coord scaling
    yolo_nms_tensor : yolo tensor (7,7,30) after nms or groundtruth 
    grid_size : int (e.g. 7)
    classes_name : str (e.g. list of classes )
    num_class : int (e.g. 20)
    op_dir : directory where to save the .xml files name of file will be extracted from the 
            given image address

    Returns
    -------
    None.
    '''
    img = cv2.imread(image_address)
    h, w,_ = img.shape
    
    num_class = num_class
    file_name = os.path.basename(image_address)[:-4]
    x = yolo_nms_tensor

    classes = x[...,0:num_class]
    b_box = x[...,num_class:num_class+4]
    obj_score = x[...,num_class+4:].squeeze()
    
    box_coord = []
    name = []
    confd = []
    for i in range(grid_size):
        for j in range(grid_size):
            if obj_score[i,j] > 0:
                #print(i,j) # get the indices which contain the objects
                cls_index = np.argmax(classes[i,j,:]) # form those indices get class index to get class name
                temp = classes_name[cls_index].strip()# get class name
                box_c = b_box[i,j,:] # get the b_box coord at same time
                box_c = xywh_2_xyminmax(w, h, box_c)
                temp2 = np.max(classes[i,j,:])# get obj_score/ confidence score
                
                name.append(temp)
                box_coord.append(box_c)
                confd.append(temp2)
                
    fileName = op_dir + file_name + '.xml'
          
    root = gfg.Element("annotation") 
     #element
    e1 = gfg.Element("filename") 
    e1.text = str(file_name)
    root.append (e1) 
    for i in range(len(name)):
        e2 = gfg.Element('object')
        root.append(e2)
        #sub-element
        se1 = gfg.SubElement(e2, 'name')
        se1.text = str(name[i])
        se2 = gfg.SubElement(e2, 'confidence')
        se2.text = str(confd[i])
        se3 = gfg.SubElement(e2, 'bndbox')
        #sub-sub-element
        sse1 = gfg.SubElement(se3, 'xmin')
        sse1.text = str((box_coord[i][0]).astype(np.int16))
        sse2 = gfg.SubElement(se3, 'ymin')
        sse2.text = str((box_coord[i][1]).astype(np.int16))
        sse3 = gfg.SubElement(se3, 'xmax')
        sse3.text = str((box_coord[i][2]).astype(np.int16))
        sse4 = gfg.SubElement(se3, 'ymax')
        sse4.text = str((box_coord[i][3]).astype(np.int16))
    
      
    root = prettify(root)
    with open (fileName, "wb") as files : 
        files.write(root)

#%%
pallet = np.array([[0, 0, 128], 
                    [0, 128, 0], 
                    [0, 128, 128],
                    [128, 0, 0], 
                    [128, 0, 128], 
                    [128, 128, 0],
                    [128, 12, 102],
                    [0, 0, 64], 
                    [0, 0, 192], 
                    [0, 128, 64],
                    [0, 128, 192],
                    [128, 0, 64], 
                    [128, 0, 192], 
                    [128, 128, 64],
                    [0, 250, 240], 
                    [0, 64, 0], 
                    [0, 64, 128],
                    [0, 192, 0],
                    [0, 192, 128], 
                    [128, 64, 0]], np.int)