import numpy as np
import matplotlib.pyplot as plt
import cv2, math
import matplotlib as mpl
from random import randint, seed
from tensorflow.python.keras.utils.data_utils import Sequence
from data_writers import draw_boxes, show_results
from data_processors import IoU, xywh_2_xyminmax, RTH, xyminmax_2_xywh
mpl.rcParams['figure.dpi'] = 300
# read_from_directory_train = '/home/user01/data_ssd/Talha/yolo_data/synth_fruit/train/'
# classes_name = []
# with open(read_from_directory_train + '_classes.txt', 'r') as f:
#         classes_name = classes_name + f.readlines()

def make_data_list(read_from_directory):
    
    train_datasets = []
    X_train = []
    Y_train = []
    
    with open(read_from_directory + '_annotations.txt', 'r') as f:
        train_datasets = train_datasets + f.readlines()
    
    for item in train_datasets:
      item = item.replace("\n", "").split(" ")
      tt = read_from_directory + item[0]
      X_train.append(tt)
      arr = []
      for i in range(1, len(item)):
        arr.append(item[i])
      Y_train.append(arr)
  
    return X_train, Y_train

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

class My_Custom_Generator(Sequence) :
  
  def __init__(self, images, labels, batch_size, modelip_img_w, modelip_img_h,
               grid_size, num_class, shuffle=True):
    
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.modelip_img_w = modelip_img_w
    self.modelip_img_h = modelip_img_h
    self.grid_size = grid_size
    self.num_class = num_class
    self.shuffle = shuffle
    self.indices = np.arange(len(self.images))
    self.i = 0
    
  def on_epoch_end(self):
      # shuffling the indices
      if self.shuffle == True:
          np.random.shuffle(self.indices)
          # print('\n Shuffling Data...')
      
  def __len__(self) :
    # getting the total no. of iterations in one epoch
    return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx):
    # from shuffled indices get the indices which will make the next batch 
    inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
    
    batch_x = []
    batch_y = []
    # loading data from those indices to arrays
    for i in inds:
        batch_x.append(self.images[i])
        batch_y.append(self.labels[i])
    
    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
      img_path = batch_x[i]
      label = batch_y[i]
      image, label_matrix = make_yolo_tensor(img_path, label, self.modelip_img_w, self.modelip_img_h,
                                             self.grid_size, self.num_class)
      train_image.append(image)
      train_label.append(label_matrix)
      
      # this  is just to save data in directory to see if generator is working
      # op = show_results((np.array(train_image)[0,::])*255, np.array(train_label)[0,::], classes_name, self.modelip_img_w, self.modelip_img_h)
      # cv2.imwrite('/home/user01/data_ssd/Talha/data_gen_test/img_{}.jpg'.format(self.i+1), op)
      # self.i = self.i+1
      
    return np.array(train_image), np.array(train_label)

#%%
# read_from_directory_train = 'C:/Users/Talha/Desktop/yolo_data/synth_fruit/train/'
# read_from_directory_val = 'C:/Users/Talha/Desktop/yolo_data/synth_fruit/valid/'
# grid_size = 7
# orig_img_w = 416 # original image resolution you used for labelling
# orig_img_h = 550
# modelip_img_w = 448
# modelip_img_h = 448
# num_class = 63
# batch_size = 4
# classes_name = []
# with open(read_from_directory_train + '_classes.txt', 'r') as f:
#         classes_name = classes_name + f.readlines()

# X_train, Y_train = make_data_list(read_from_directory_train)
# X_val, Y_val = make_data_list(read_from_directory_val)
# my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size, modelip_img_w, modelip_img_h,
#                                                   orig_img_w, orig_img_h, grid_size, num_class)
# my_validation_batch_generator = My_Custom_Generator(X_val, Y_val, batch_size, modelip_img_w, modelip_img_h,
#                                                   orig_img_w, orig_img_h, grid_size, num_class)

# x_train, y_train = my_training_batch_generator.__getitem__(0)
# x_val, y_val = my_training_batch_generator.__getitem__(0)
# print(x_train.shape)
# print(y_train.shape)

# print(x_val.shape)
# print(y_val.shape)

# #%%
# i = 3
# img = x_train[i, ...]  # batch * img_w * img_h * 3 => img_w * img_h * 3
# gt = y_train[i, ...] # batch * S * S * (num_class + 5) => S * S * (num_class+5)
# op = show_results(img, gt, classes_name, modelip_img_w, modelip_img_h)

# plt.imshow(op)

















