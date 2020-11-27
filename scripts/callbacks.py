import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os 
import random
from IPython.display import clear_output
from data_writers import draw_boxes, show_results
from data_processors import IoU, xywh_2_xyminmax, RTH, xyminmax_2_xywh, nms_v1
from YOLO_DataGenerator import make_yolo_tensor
import copy

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import tensorflow as tf
from collections import defaultdict
if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
    from keras import backend as K
    from keras.callbacks import Callback
    from keras.layers import DepthwiseConv2D, SeparableConv2D, Conv2D, Dropout
if tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0' or tf.__version__ == '2.2.0-rc2':
    import tensorflow.keras.backend as K
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D, Conv2D, Dropout


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.lr = []
        self.dropout = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.metric1 = self.model.metrics_names[0]
        self.metric2 = self.model.metrics_names[0]
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get(self.metric1))
        self.val_losses.append(logs.get('val_' + self.metric1))
        self.acc.append(logs.get(self.metric2))
        self.val_acc.append(logs.get('val_' + self.metric2))
        self.lr.append(float(K.get_value(self.model.optimizer.lr)))
        if len(self.model.layers) > 10:
            for layer in self.model.layers:
                if isinstance(layer, Dropout) == True:
                    self.dropout.append(K.get_value(layer.rate))
                    break
        else:
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, Dropout) == True:
                    self.dropout.append(K.get_value(layer.rate))
                    break
        
        self.i += 1
        f, ax = plt.subplots(2, 2, figsize = (10,10), sharex=False)
        
        clear_output(wait=True)
        ax[0,0].plot(self.x, self.losses, 'g--*', label=self.metric1)
        ax[0,0].plot(self.x, self.val_losses, 'r-.+', label='val_' + self.metric1)
        ax[0,0].legend()
        ax[0,0].axis(xmin=0, ymin=0, ymax=100)

        ax[0,1].plot(self.x, self.acc, 'g--*', label=self.metric2)
        ax[0,1].plot(self.x, self.val_acc, 'r-.+', label='val_' + self.metric2)
        ax[0,1].legend()
        
        ax[1,0].plot(self.x, self.lr, 'm-*', label='Learning Rate')
        ax[1,0].legend()
        
        ax[1,1].plot(self.x, self.dropout, 'c-*', label='Dropout')
        ax[1,1].legend()
        
        plt.show();

class PredictionCallback(Callback):
    '''
    Decalre Arguments Input Here
    '''
    def __init__(self, address_list, label_list, im_height, im_width, classes_name, grid_size, num_class, boxes_percell):
            self.address_list = address_list
            self.label_list = label_list
            self.im_height = im_height
            self.im_width = im_width
            self.classes_name = classes_name
            self.grid_size = grid_size
            self.num_class = num_class
            self.boxes_percell = boxes_percell
            
    def on_epoch_end(self, epoch, logs={}):
            
            idx = random.randint(0, len(self.address_list)-1)
            address = self.address_list[idx]
            label = self.label_list[idx]
            _, label_matrix = make_yolo_tensor(address, label, self.im_width, self.im_height,
                                                   self.grid_size, self.num_class)
            img_o = cv2.imread(address)     
            img_o = cv2.resize(img_o, (self.im_width, self.im_height))
            img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB) /255
            img1 = copy.deepcopy(img_o)
            img2 = copy.deepcopy(img_o)
            
            orig = show_results(img1, label_matrix, self.classes_name, self.im_width, self.im_height, from_nms=False)
            
            pred = self.model.predict(img_o[np.newaxis,:,:,:]/255, verbose=0)#Normalizing
            pred = pred.squeeze()
          
            try:
              nms_pred = nms_v1(pred, 0.3, 0.4, self.grid_size, self.num_class, self.boxes_percell, self.im_width, self.im_height)
              op = show_results(img2, nms_pred, self.classes_name, self.im_width, self.im_height)
            except ValueError:
              try: 
                  nms_pred = nms_v1(pred, 0.3, 0.3, self.grid_size, self.num_class, self.boxes_percell, self.im_width, self.im_height)
                  op = show_results(img2, nms_pred, self.classes_name, self.im_width, self.im_height)
              except:
                  op = img2
                  op = cv2.putText(op, 'Nothind Detected', (200, 200),
                                   cv2.FONT_HERSHEY_DUPLEX   , 0.5, (255, 0, 0), 2) 
            except:
               op = img2
               op = cv2.putText(op, 'Nothind Detected', (200, 200),
                                   cv2.FONT_HERSHEY_DUPLEX   , 0.5, (255, 0, 0), 2) 
            
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,4), sharex=False)
            f.suptitle('Pred after {} Epoch(s)'.format(epoch+1))
            clear_output(wait=True)
            
            ax1.imshow(img_o)
            ax1.axis("off")
            
            ax2.imshow(orig)
            ax2.axis("off")
            
            ax3.imshow(op)
            ax3.axis("off")
            
            plt.show();
            
class CustomLearningRateScheduler(Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule, initial_lr, lr_decay, total_epochs, drop_epoch, power):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.total_epochs = total_epochs
        self.drop_epoch = drop_epoch
        self.power = power
        
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
            
        if self.schedule == 'step_decay':
            self.schedule = step_decay
        if self.schedule == 'polynomial_decay':
            self.schedule = polynomial_decay
        if self.schedule == 'K_decay':  
            self.schedule = K_decay
        if self.schedule == 'yolo_decay':
            self.schedule = yolo_decay
        lr = self.initial_lr
        if lr is None:
            # Get the current learning rate from model's optimizer.
            lr = float(K.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr, self.lr_decay, self.drop_epoch, self.total_epochs, self.power)
        # Set the value back to the optimizer before this epoch starts
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch {}: Learning rate is {}".format(epoch+1, scheduled_lr))
        
class CustomDropoutScheduler(Callback):
    def __init__(self, schedule, dropout_after_epoch):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.DAE = dropout_after_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if len(self.model.layers) > 10: # for handeling the multi gpu strategy
            for i in range(len(self.DAE)):
                if epoch == self.DAE[i]:
                    print('Updating Dropout Rate...')
                    for layer in self.model.layers:
                        if isinstance(layer, Dropout):
                            new_drop_out = self.schedule[i]
                            K.set_value(layer.rate, new_drop_out)
            for layer in self.model.layers:
                if isinstance(layer, Dropout) == True:
                    print('Epoch %05d: Dropout Rate is %6.4f' % (epoch+1, K.get_value(layer.rate)))
                    break
        else:
            for i in range(len(self.DAE)):
                if epoch == self.DAE[i]:
                    print('Updating Dropout Rate...')
                    for layer in self.model.layers[-2].layers:
                        if isinstance(layer, Dropout):
                            new_drop_out = self.schedule[i]
                            K.set_value(layer.rate, new_drop_out)
            for layer in self.model.layers[-2].layers:
                if isinstance(layer, Dropout) == True:
                    print('Epoch %05d: Dropout Rate* is %6.4f' % (epoch+1, K.get_value(layer.rate)))
                    break
       
class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
        Code: https://gist.github.com/jeremyjordan/5a222e04bb78c242f5763ad40626c452
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the maximum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())
        
    def on_epoch_begin(self, epoch, logs=None):
        print("Epoch %05d: Learning rate is %6.2e"  % (epoch+1, K.get_value(self.model.optimizer.lr)))
        
    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)      
#%%
class WRWDScheduler(Callback):
    """Warm restart scheduler for optimizers with decoupled weight decay
    
    This Keras callback should be used with TensorFlow optimizers 
    with decoupled weight decay, such as tf.contrib.opt.AdamWOptimizer
    or tf.contrib.opt.MomentumWOptimizer. Warm restarts include 
    cosine annealing with periodic restarts for both learning rate 
    and weight decay. Normalized weight decay is also included.
    
    from collections import defaultdict
    import tensorflow_addons as tfa
    tfa.optimizers.AdamW(weight_decay=0.03,learning_rate= 0.001)
    
    # Example
    ```python
    lr = 0.001
    wd = 0.01
    optimizer = tf.contrib.opt.AdamWOptimizer(
        learning_rate=lr,
        weight_decay=wd)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    cb_wrwd = WRWDScheduler(steps_per_epoch=100, lr=lr, wd_norm=wd)
    model.fit(x_train, y_train, callbacks=[cb_wrwd])
    figure(1);plt.plot(cb_wrwd.history['lr'])
    figure(2);plt.plot(cb_wrwd.history['wd'])
    ```
    # Arguments
        steps_per_epoch: number of training batches per epoch
        lr: initial learning rate
        wd_norm: normalized weight decay
        eta_min: minimum of the multiplier
        eta_max: maximum of the multiplier
        eta_decay: decay rate of eta_min/eta_max after each restart
        cycle_length: number of epochs in the first restart cycle
        cycle_mult_factor: rate to increase the number of epochs 
            in a cycle after each restart
    # Reference
        arxiv.org/abs/1608.03983
        arxiv.org/abs/1711.05101
        jeremyjordan.me/nn-learning-rate
        https://gist.github.com/chaohuang/e1c624027e16a0428489163ceb7b1f06
    """
    
    def __init__(self,
                 steps_per_epoch,
                 lr=0.001,
                 wd_norm=0.03,
                 eta_min=0.0,
                 eta_max=1.0,
                 eta_decay=1.0,
                 cycle_length=10,
                 cycle_mult_factor=1.5):
        
        self.lr = lr
        self.wd_norm = wd_norm

        self.steps_per_epoch = steps_per_epoch

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_decay = eta_decay

        self.steps_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.cycle_mult_factor = cycle_mult_factor

        self.wd = wd_norm / (steps_per_epoch*cycle_length)**0.5

        self.history = defaultdict(list)

    def cal_eta(self):
        '''Calculate eta'''
        fraction_to_restart = self.steps_since_restart / (self.steps_per_epoch * self.cycle_length)
        eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + np.cos(fraction_to_restart * np.pi))
        return eta

    def on_train_batch_begin(self, batch, logs={}):
        '''update learning rate and weight decay'''
        eta = self.cal_eta()
        self.model.optimizer.optimizer._learning_rate = eta * self.lr
        self.model.optimizer.optimizer._weight_decay = eta * self.wd

    def on_train_batch_end(self, batch, logs={}):
        '''Record previous batch statistics'''
        logs = logs or {}
        self.history['lr'].append(self.model.optimizer.optimizer._learning_rate)
        self.history['wd'].append(self.model.optimizer.optimizer._weight_decay)
        for k, v in logs.items():
            self.history[k].append(v)

        self.steps_since_restart += 1

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary'''
        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.cycle_mult_factor)
            self.next_restart += self.cycle_length
            self.eta_min *= self.eta_decay
            self.eta_max *= self.eta_decay
            self.wd = self.wd_norm / (self.steps_per_epoch*self.cycle_length)**0.5
#%%
def step_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power):
    initial_lrate = initial_lr
    drop = lr_decay
    epochs_drop = drop_epoch
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def polynomial_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power):
    initial_lrate = initial_lr
    lrate = initial_lrate * math.pow((1-(epoch/Epoch)),power)
    return lrate

# x = np.arange(0,50)# current epoch 
# Epoch  = 50
# k = 0.4
# N = 1
# inint_lr = 0.002
# final_lr = 0

def K_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power, Le=1e-7, N=4, k=3):
    t = epoch
    L0 = initial_lr
    T = Epoch
    lr = (L0 - Le) * (1 - t**k / T**k)**N + Le
    return lr

def yolo_decay(epoch, initial_lr, lr_decay, drop_epoch, Epoch, power, Le=1e-7, N=4, k=3):

    if epoch <= 1:
        lr = 0.0001
    elif epoch == 2:
        lr = 0.00015
    elif epoch <= 75:
        lr = 0.0002
    elif epoch <=  95:
        lr = 0.00015
    elif epoch <= Epoch:
        lr = 0.0001
    return lr