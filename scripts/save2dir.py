#%%
#from main import *
'''
Carefully select the confd_th for the both try_except conditions
For exception values are o.4 and 0.2
For orig are 0.2 and 0.1
'''
from tqdm import trange, tqdm
iou_thresh = 0.3
confd_thersh1 = 0.3
confd_thersh2 = 0.2
a = 0
b = 0
for i in trange(len(X_val)):
    n = X_val[i]
    
    img_name = os.path.basename(n)
    img = cv2.imread(n)
    img = cv2.resize(img, (modelip_img_w, modelip_img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pred = model.predict(img[np.newaxis,:,:,:]/255, verbose=0)#Normalizing
    pred = pred.squeeze()
    #%
    try:
        # # version 1
        # nms_pred = nms_v1(pred, iou_thresh, confd_thersh1, grid_size, num_class, boxes_percell, modelip_img_w, modelip_img_h)
        # op = show_results(img, nms_pred, classes_name, modelip_img_w, modelip_img_h)
        # version 2
        nms_boxes, nms_scores, nms_classes_names, nms_box_norm = nms_v2(pred, iou_thresh, confd_thersh1, grid_size, num_class, 
                                                          classes_name, boxes_percell, modelip_img_w, modelip_img_h, use_numpy=True)
        op = draw_boxes(img, nms_scores, nms_boxes, nms_classes_names, classes_name)
        
        op = cv2.cvtColor(op, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/home/user01/data_ssd/Talha/yolo_data/op/{0}'.format(img_name), op)    
    except ValueError:
        try:
            # # version 1
            # nms_pred = nms_v1(pred, iou_thresh, confd_thersh1, grid_size, num_class, boxes_percell, modelip_img_w, modelip_img_h)
            # op = show_results(img, nms_pred, classes_name, modelip_img_w, modelip_img_h)
            # version 2
            nms_boxes, nms_scores, nms_classes_names, nms_box_norm = nms_v2(pred, iou_thresh, confd_thersh1, grid_size, num_class, 
                                                              classes_name, boxes_percell, modelip_img_w, modelip_img_h, use_numpy=True)
            op = draw_boxes(img, nms_scores, nms_boxes, nms_classes_names, classes_name)
            
            op = cv2.cvtColor(op, cv2.COLOR_RGB2BGR)
            cv2.imwrite('/home/user01/data_ssd/Talha/yolo_data/op/{0}'.format(img_name), op)  
        except:
            #print("Couldn't predict")
            a+=1
    except:
        b+=1
        pass#print("Overflow Error")