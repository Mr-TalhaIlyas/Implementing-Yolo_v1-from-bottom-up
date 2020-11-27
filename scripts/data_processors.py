import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randint, seed
import tensorflow as tf
from tqdm import tqdm
import xmltodict, os, glob
#%%
def RTH(x, Thresh = 0.2, thresh = True):
    a,b,c = x.shape
    x = np.reshape(x, (c,1,1,a*b))
    x =  np.reshape(x, (a*b,c)).T
    if thresh:
        x = np.where(x<Thresh, 0, x)
    return x
#%%
def xywh_2_xyminmax(img_w, img_h, box):
    '''
    input_box : (x, y, w, h)
    output_box : (xmin, ymin, xmax, ymax) @ un_normalized
    '''
    xmin = box[0] - (box[2] / 2)
    ymin = box[1] - (box[3] / 2)
    xmax = box[0] + (box[2] / 2)
    ymax = box[1] + (box[3] / 2)
    
    box_minmax = np.array([xmin*img_w, ymin*img_h, xmax*img_w, ymax*img_h])
    
    return box_minmax
#%%
def xyminmax_2_xywh(img_w, img_h, box):
    '''
    input_box  : (xmin, ymin, xmax, ymax)
    output_box : (x, y, w, h) @ normalized
    '''
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = np.abs(box[2] - box[0])
    h = np.abs(box[3] - box[1])
    
    x = x * (1./img_w)
    w = w * (1./img_w)
    y = y * (1./img_h)
    h = h * (1./img_h)
    
    return (x,y,w,h)

#%%   
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

#%%
def nms_v1(pred, iou_thresh, confd_thersh, grid_size, num_class, boxes_percell, modelip_img_w, modelip_img_h):
    
    # iou_thresh = 0.5
    # confd_thersh = 0.2
    
    grid_square = grid_size * grid_size
    pred_class_pr = pred[:,:,0:num_class]   
    
    pred_class_pr = pred[:,:,0:num_class]                        # class probabilities
    pred_obj_score = pred[:,:,num_class:num_class+boxes_percell] # objectness score
    pred_box_coord = pred[:,:,num_class+boxes_percell:]          # b_box coordinates of both boxes
    ###### 
    #Removing offset
    # verfication ! after model is trained
    offset_x = np.tile(np.arange(0,grid_size), grid_size).reshape(grid_size, grid_size)
    offset_y = offset_x.T
    # as boxes are of the form (7,7,8) -> (x,y,w,h, x,y,w,h) so get 0,1 and 4,5 indexes to get both x's and y's.
    for i in range(0, pred_box_coord.shape[-1], 4):
        pred_box_coord[:,:,i] = np.where(pred_box_coord[:,:,i] != 0, pred_box_coord[:,:,i] + offset_x, 0) / grid_size
        pred_box_coord[:,:,i+1] = np.where(pred_box_coord[:,:,i+1] != 0, pred_box_coord[:,:,i+1] + offset_y, 0) / grid_size
        # taking square of w and h to revrese the conversions we did for loss calculations
        # pred_box_coord[:,:,i+2] = np.square(pred_box_coord[:,:,i+2])
        # pred_box_coord[:,:,i+3] = np.square(pred_box_coord[:,:,i+3])
    ######
    
    bb1_conf = np.multiply(pred_class_pr, pred_obj_score[:,:,0:1])# confidence score of box1
    bb2_conf = np.multiply(pred_class_pr, pred_obj_score[:,:,1:2])# confidence score of box2
    
    # setting negative values to zero
    pred_box_coord[pred_box_coord<0] = 0
    
    bb1_conf_r = RTH(bb1_conf, Thresh = confd_thersh) # reshaping and thresholding on class confidance scores 
    bb2_conf_r = RTH(bb2_conf, Thresh = confd_thersh) # _r => reshaped
    
    bb1_coord = RTH(pred_box_coord[:,:,0:4], thresh = False)
    bb2_coord = RTH(pred_box_coord[:,:,4:8], thresh = False)
    
    empty_girds1 = (np.max(bb1_conf_r, axis=0) > 0).astype(np.int16).reshape((1,grid_square))# setting the grid cells having no obj to zero
    survived_bb1_coord = np.multiply(empty_girds1, bb1_coord) # remaining coords which have confidence of containing object
    #same for b_box 2
    empty_girds2 = (np.max(bb2_conf_r, axis=0) > 0).astype(np.int16).reshape((1,grid_square))
    survived_bb2_coord = np.multiply(empty_girds2, bb2_coord)
    
    for clas in range(num_class):
        bb1n2_conf = np.concatenate((bb1_conf_r, bb2_conf_r), axis = -1)
        class_scores_sorted = np.sort(bb1n2_conf[clas,:])[::-1]     # [::-1] for descending order remove and get ascending order
        class_indices_sorted = np.argsort(bb1n2_conf[clas,:])[::-1] # getting the indices for sorted list so that 
                                                                  # we can get corresponding coordinates
        class_indices_sorted = np.multiply(class_indices_sorted,  # only keeping indices whoes corresponding scores are greater than 0.
                                          (class_scores_sorted > 0).astype(np.int16))                                                         
        for a in range(len(class_indices_sorted)):
            
            i = class_indices_sorted[a]
            k = class_scores_sorted[a]
            if k != 0:
                for b in range(np.count_nonzero(class_indices_sorted)):
                    j = class_indices_sorted[a+b+1]
                    # get the b_box with max confidence score that is sure to have max iou
                    if i < grid_square:
                        bb_max = bb1_coord[:, i]
                    elif i >= grid_square:
                        bb_max = bb2_coord[:, i%grid_square]
                    # get the next indice of the sorted list so that instead of looping over un-sorted list we can 
                    # loop over sorted one
                    if j < grid_square:
                        bb_nxt = bb1_coord[:, j]
                    elif j >= grid_square:
                        bb_nxt = bb2_coord[:, j%grid_square]
                    
                    # un_normalize and convert center_wh to min_max
                    bb_max =  np.abs(xywh_2_xyminmax(modelip_img_w, modelip_img_h, box=bb_max))
                    bb_nxt =  np.abs(xywh_2_xyminmax(modelip_img_w, modelip_img_h, box=bb_nxt))
                    
                    # calculate iou
                    iou = IoU(bb_max, bb_nxt)
                    #print(iou)
                    #print(i,j)
                    if iou > iou_thresh:
                        #print(iou)
                        if j < grid_square:
                            #print('i and j=',i, j)
                            bb1_conf_r[clas, j] = 0
                            survived_bb1_coord[:, j] = 0
                            #print('bb1 set zero',j)
                        elif j >= grid_square:
                            #print('i and j=',i, j)
                            bb2_conf_r[clas, j%grid_square] = 0
                            survived_bb2_coord[:, j%grid_square] = 0
                            #print('bb2 set zero',j)
                        
    #nms_bb_conf = bb1_conf_r + bb2_conf_r
    nms_bb_conf = np.empty((bb1_conf_r.shape))
    for i in range(bb1_conf_r.shape[0]):
        for j in range(bb1_conf_r.shape[1]):
            if bb1_conf_r[i,j] > 0 and bb2_conf_r[i,j] > 0:
                #print('A  Jujarr')
                nms_bb_conf[i,j] = (bb1_conf_r[i,j])# + bb2_conf_r[i,j])/2
            else:
                nms_bb_conf[i,j] = (bb1_conf_r[i,j] + bb2_conf_r[i,j])
                
    nms_bb_conf = nms_bb_conf.T
    nms_bb_conf = np.reshape(nms_bb_conf, (grid_size,grid_size,num_class))
    
    # doing this because still there might be some boxes who couldn't zeroed out 
    # so we take average of them like this
    nms_bb_coord = np.empty((survived_bb1_coord.shape))
    for i in range(survived_bb1_coord.shape[0]):
        for j in range(survived_bb1_coord.shape[1]):
            if survived_bb1_coord[i,j] > 0 and survived_bb2_coord[i,j] > 0:
                #print('B  Jujarr')
                nms_bb_coord[i,j] = (survived_bb1_coord[i,j])# + survived_bb2_coord[i,j])/2
            else:
                nms_bb_coord[i,j] = (survived_bb1_coord[i,j] + survived_bb2_coord[i,j])
                
    nms_bb_coord = nms_bb_coord.T
    nms_bb_coord = np.reshape(nms_bb_coord, (grid_size,grid_size,4))
    # just so that same function (show_results) can handle the both gt and preds.
    tobj_score = (np.sum(nms_bb_coord, axis= -1) > 0).astype(np.int16)[:,:,np.newaxis]
    # same order as the GT labels:
    nms_tensor = np.concatenate((nms_bb_conf, nms_bb_coord, tobj_score), axis = -1)   
    return nms_tensor

#%%  
def nms_v2(pred, iou_thresh, confd_thersh, grid_size, num_class, classes_name, boxes_percell, modelip_img_w, modelip_img_h, use_numpy=True):
    '''
    Parameters
    ----------
    pred : yolo_op of shape (grid_size, grid_size, num_class+(boxes_per_cell+(boxes_per_cell x 4)))
            (e.g. None,7,7,30)
    num_classes : int (e.g. 20)
    grid_size : int (e.g. 7)
    boxes_per_cell : int (e.g. 2)
    classes_name : str (e.g. list of classes )
    image_shape : tuple (e.g. 448,448)
    max_boxes : int (e.g. 10)
    score_threshold : float less than 1
    iou_threshold : float less than 1

    Returns
    -------
    scores :shape (None,)
    boxes : shape (None, 4)
    classes : shape (None,)
    '''
    
    class_pr = pred[...,:num_class]
    obj_score = pred[...,num_class:num_class+boxes_percell]
    box_coord = pred[..., num_class+boxes_percell:]
    ###### 
    #Removing offset
    # verfication ! after model is trained
    offset_x = np.tile(np.arange(0,grid_size), grid_size).reshape(grid_size, grid_size)
    offset_y = offset_x.T
    # as boxes are of the form (7,7,8) -> (x,y,w,h, x,y,w,h) so get 0,1 and 4,5 indexes to get both x's and y's.
    for i in range(0, box_coord.shape[-1], 4):
        box_coord[:,:,i] = np.where(box_coord[:,:,i] != 0, box_coord[:,:,i] + offset_x, 0) / grid_size
        box_coord[:,:,i+1] = np.where(box_coord[:,:,i+1] != 0, box_coord[:,:,i+1] + offset_y, 0) / grid_size
        # taking square of w and h to revrese the conversions we did for loss calculations
        # box_coord[:,:,i+2] = np.square(box_coord[:,:,i+2])
        # box_coord[:,:,i+3] = np.square(box_coord[:,:,i+3])
    ######
    # concating the class probab. twice b/c boxes_percell = 2
    class_pr = np.concatenate((class_pr, class_pr), axis = -1)
    # reshaping into (e.g. 7,7,2,20)
    class_pr = np.reshape(class_pr, (grid_size,grid_size,boxes_percell,num_class))
    # reshaping into (e.g. 7,7,2,4)
    box_coord = np.reshape(box_coord, (grid_size,grid_size,boxes_percell,4))
    # reshaping into (e.g. 7,7,2,1)
    obj_score = obj_score[..., np.newaxis]
    # getting confidence scores
    confd_scores = np.multiply(class_pr, obj_score)

    # returns the indices of cell which highest confidence of having an object
    grid_cell_classes = np.argmax(confd_scores, axis=-1)
    # returns the confidence score of a class in a cell
    grid_cell_class_scores = np.max(confd_scores, axis=-1)
    
    # zeroing out the less confident cells
    mask = (grid_cell_class_scores >= confd_thersh)
    
    # applying mask to the grid_cell_class_scores and 
    grid_cell_classes = grid_cell_classes[mask]
    grid_cell_class_scores = grid_cell_class_scores[mask]
    box_coord = box_coord[mask] # boxes are in xy_wh normalized format
    
    # scaling the coord backe to model input dimensions:
    # ***** we will use these scaled boxes only for iou calculation ***
    # these are in xy_minmax scaled to model input format
    box_coord_scaled = np.empty((box_coord.shape))
    for i in range(box_coord.shape[0]):
        box_coord_scaled[i,:] = xywh_2_xyminmax(modelip_img_w, modelip_img_h, box_coord[i,:])

    '''choose one of the following'''
    if use_numpy:
        ##################################
        # NMS using numpy 
        ##################################
        # get the indices in descending order
        sort_ind = np.argsort(grid_cell_class_scores)[::-1]
        # sort all the arrays to be used accordingly
        sorted_class_score = grid_cell_class_scores[sort_ind]
        sorted_cell_classes = grid_cell_classes[sort_ind]
        sorted_box_coord = box_coord_scaled[sort_ind,:]
        sorted_box_coord_norm = box_coord[sort_ind,:]
        
        for i in range(sorted_box_coord.shape[0]):
            for j in range(i+1, sorted_box_coord.shape[0], 1):
                box_max = sorted_box_coord[i]
                box_cur = sorted_box_coord[j]
                iou = IoU(box_max, box_cur)
                #print(iou)
                if iou > iou_thresh:
                    sorted_box_coord[j,:] = 0
                    sorted_class_score[j] = 0
                    sorted_cell_classes[j] = 0
        # here for simplicity we will use the non_zero indices produced by the sorted 
        # class score b/c of the correspondence between them
        non_zero_ind = np.nonzero(sorted_class_score)
        nms_scores = sorted_class_score[non_zero_ind]
        nms_boxes = sorted_box_coord[non_zero_ind]
        nms_classes = sorted_cell_classes[non_zero_ind]
        nms_box_coord_norm = sorted_box_coord_norm[non_zero_ind]
        nms_classes_names = np.take(classes_name, nms_classes)
        
    else:
        ##################################
        # NMS using tf function
        ##################################
        # using tf built in function to supress the boxes, it'll return the indices of the surviving boxes
        print('Using TF')
        max_boxes_tensor = 10
        nms_ind = tf.image.non_max_suppression(box_coord_scaled, grid_cell_class_scores, max_boxes_tensor, iou_threshold=iou_thresh)
        sess = tf.compat.v1.Session()
        nms_ind = sess.run(nms_ind)
        
        # get the values of arrays at the survived indices
        nms_boxes = box_coord_scaled[nms_ind,:]
        nms_scores = grid_cell_class_scores[nms_ind]
        nms_classes = grid_cell_classes[nms_ind].astype(np.int)
        nms_classes_names = np.take(classes_name,nms_classes)
        nms_box_coord_norm = box_coord[nms_ind]
    
    return nms_boxes, nms_scores, nms_classes_names, nms_box_coord_norm
#%%
'''
COnvert xml files to txt files to plot the txt files because the function requires it
'''
def remove_duplicate(s):
    '''
    I created this function b/c i am looping over all the keys in the xml file
    and in case if xmldict have only one object the loop sitll consider the 
    <part> keys to be part of object so i am just looping over same object again 
    and again. So i just rmove those duplication with this
    '''
    x = s.split(' ')
    y = list(set(x))
    y = ' '.join(map(str, y))
    return y
def convert_xml2txt(xml_filepaths, op_dir, gt=True):
    
    xml_filepaths = glob.glob( os.path.join( xml_filepaths , '*.xml' ) )
    for filepath in tqdm(xml_filepaths, desc='Converting'):
     
        full_dict = xmltodict.parse(open( filepath , 'rb' ))
        
        obj_boxnnames = full_dict[ 'annotation' ][ 'object' ] # names and boxes
        file_name = os.path.basename(filepath)[:-4]#full_dict[ 'annotation' ][ 'filename' ]
        all_bounding_boxnind = []
        for i in range(len(obj_boxnnames)):
            # 1st get the name and indices of the class
            try:
                obj_name = obj_boxnnames[i]['name']
                try:
                    obj_confd = obj_boxnnames[i]['confidence']
                except:
                    pass
            except:
                obj_name = obj_boxnnames['name']  # if the xml file has only one object key
                try:
                    obj_confd = obj_boxnnames['confidence']
                except:
                    pass
            
            # 2nd get tht bbox coord and append the class name at the end
            try:
                obj_box = obj_boxnnames[i]['bndbox']
            except:
                obj_box = obj_boxnnames['bndbox'] # if the xml file has only one object key
            if gt:
                bounding_box = [0.0] * 5# 5 or 6                     # creat empty list
                bounding_box[0] = obj_name
                #bounding_box[1] = obj_confd
                bounding_box[1] = int(float(obj_box['xmin']))# two times conversion is for handeling exceptions 
                bounding_box[2] = int(float(obj_box['ymin']))# so that if coordinates are given in float it'll
                bounding_box[3] = int(float(obj_box['xmax']))# still convert them to int
                bounding_box[4] = int(float(obj_box['ymax']))
            else:
                
                bounding_box = [0.0] * 6                     # creat empty list
                bounding_box[0] = obj_name
                bounding_box[1] = obj_confd
                bounding_box[2] = int(float(obj_box['xmin']))# two times conversion is for handeling exceptions 
                bounding_box[3] = int(float(obj_box['ymin']))# so that if coordinates are given in float it'll
                bounding_box[4] = int(float(obj_box['xmax']))# still convert them to int
                bounding_box[5] = int(float(obj_box['ymax']))
            #bounding_box.append(obj_ind)                # append the class ind in same list (YOLO format)
            bounding_box = str(bounding_box)[1:-1]      # remove square brackets
            bounding_box = bounding_box.replace("'",'')# removing inverted commas around class name
            bounding_box = "".join(bounding_box.split())# remove spaces in between **here dont give space inbetween the inverted commas "".
            all_bounding_boxnind.append(bounding_box)
        all_bounding_boxnind = ' '.join(map(str, all_bounding_boxnind))# convert list to string
        all_bounding_boxnind = remove_duplicate(all_bounding_boxnind) # remove duplicates
        all_bounding_boxnind=list(all_bounding_boxnind.split(' ')) # convert strin to list
        # replacing commas with spaces
        for i in range(len( all_bounding_boxnind)):
            all_bounding_boxnind[i] = all_bounding_boxnind[i].replace(',',' ')      
        for i in range(len(all_bounding_boxnind)):
        # check if file exiscts else make new
            with open(op_dir +  file_name + ".txt", "a+") as file_object:
                # Move read cursor to the start of file.
                file_object.seek(0)
                # If file is not empty then append '\n'
                data = file_object.read(100)
                if len(data) > 0 :
                    file_object.write("\n")
                # Append text at the end of file
                file_object.write(all_bounding_boxnind[i])