import tensorflow as tf
import numpy as np
if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0':
    from keras.layers.merge import concatenate
    from keras.layers import Activation, Layer
    import keras.backend as K
if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0-rc2':
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Activation, concatenate, Layer
    import tensorflow_addons as tfa
    
#%%
grid_size = 13
modelip_img_w = 448
modelip_img_h = 448
num_class = 20
boxes_percell = 2
# Loss functions hyper parameters
lambda_coord = 7
lambda_noobj = 0.4

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * modelip_img_w
    box_wh = feats[..., 2:4] * modelip_img_w

    return box_xy, box_wh


def YOLO_LOSS(y_true, y_pred):
    # tensor made by make_yolo_tensor function
    # in labels the tensor order is:
    #                       [class_prob, b_boxes, obj_score]
    label_class = y_true[..., 0:num_class]  # ? * 7 * 7 * 20
    label_box = y_true[..., num_class:num_class+4]  # ? * 7 * 7 * 4
    response_mask = y_true[..., num_class+4:]  # ? * 7 * 7 * 1
    
    
    # in predicted tesnsor the tensor order is:
    #                       [class_prob, obj_score, b_boxes]
    predict_class = y_pred[..., 0:num_class]  # ? * 7 * 7 * 20
    predict_trust = y_pred[..., num_class:num_class + boxes_percell]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., num_class + boxes_percell:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, grid_size, grid_size, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, grid_size, grid_size, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = lambda_noobj * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, grid_size, grid_size, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, grid_size, grid_size, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = lambda_coord * box_mask * response_mask * K.square((label_xy - predict_xy) / modelip_img_w)
    box_loss += lambda_coord * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / modelip_img_w)
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss
#%%
lambdaNoObject = 0.5
lambdaCoordinate = 5
def iou_train_unit(boxA, realBox):
        """
        Calculate IoU between boxA and realBox
        """
        # make sure that the representation of box matches input
        intersectionX = tf.minimum(
            boxA[:, :, :, 0] + 0.5*boxA[:, :, :, 2],
            realBox[:, :, :, 0] + 0.5*realBox[:, :, :, 2]
            ) - tf.maximum(
                    boxA[:, :, :, 0] - 0.5*boxA[:, :, :, 2],
                    realBox[:, :, :, 0] - 0.5*realBox[:, :, :, 2]
                    )
        intersectionY = tf.minimum(
            boxA[:, :, :, 1] + 0.5*boxA[:, :, :, 3],
            realBox[:, :, :, 1] + 0.5*realBox[:, :, :, 3]
            ) - tf.maximum(
                        boxA[:, :, :, 1] - 0.5*boxA[:, :, :, 3],
                        realBox[:, :, :, 1] - 0.5*realBox[:, :, :, 3]
                        )
        intersection = tf.multiply(
            tf.maximum(0., intersectionX), tf.maximum(0., intersectionY)
            )
        union = tf.subtract(
                    tf.multiply(
                        boxA[:, :, :, 1], boxA[:, :, :, 3]) + tf.multiply(
                            realBox[:, :, :, 1], realBox[:, :, :, 3]
                            ),
                        intersection
                        )
        iou = tf.divide(intersection, union)
        return iou
    
def iou_train(boxA, boxB, realBox):
        """
        Calculate IoU between boxA and realBox
        Calculate the IoU in training phase, to get the box (out of N
            boxes per grid) responsible for ground truth box
        """
        iou1 = tf.reshape(iou_train_unit(boxA, realBox), [-1, 7, 7, 1])
        iou2 = tf.reshape(iou_train_unit(boxB, realBox), [-1, 7, 7, 1])
        return tf.concat([iou1, iou2], 3)
    
def YOLO_LOSS_Updated(groundTruth, predicted):
        """
        Calculate the total loss for gradient descent.
        For each ground truth object, loss needs to be calculated.
        It is assumed that each image consists of only one object.
        Predicted
        0-19 CLass prediction
        20-21 Confidence that objects exist in bbox1 or bbox2 of grid
        22-29 Coordinates for bbo1, followed by those of bbox2
        Real
        0-19 Class prediction (One-Hot Encoded)
        20-23 Ground truth coordinates for that box
        24-72 Cell has an object/no object (Only one can be is 1)
        """
        
        predictedParameters = predicted
        predictedClasses = predictedParameters[:, :, :, :20]
        predictedObjectConfidence = predictedParameters[:, :, :, 20:22]
        predictedBoxes = predictedParameters[:, :, :, 22:]
        # seperate both boxes
        predictedFirstBoxes = predictedBoxes[:, :, :, 0:4]
        predictedSecondBoxes = predictedBoxes[:, :, :, 4:]
        
        groundTruthClasses = groundTruth[:, :, :, :20]
        groundTruthBoxes = groundTruth[:, :, :, 20:24]
        groundTruthGrid = groundTruth[:, :, :, 24:]
        
        # Calulate loss along the 4th axis, localFirstBoxes -1x7x7x1
        # Think there should be a simpler method to do this
        lossFirstBoxes = tf.reduce_sum(
            tf.square(predictedFirstBoxes - groundTruthBoxes), 3)
        lossSecondBoxes = tf.reduce_sum(
            tf.square(predictedSecondBoxes - groundTruthBoxes), 3)
        # Computing which box (bbox1 or bbox2) is responsible for
        # detection
        IOU = iou_train(predictedFirstBoxes,
                       predictedSecondBoxes, groundTruthBoxes)
        responsibleBox = tf.greater(IOU[:, :, :, 0], IOU[:, :, :, 1])
        # Suppose it is known which iou is greater,
        # coordinate loss (loss due to difference in coordinates of
        # predicted-responsible and real box)
        coordinateLoss = tf.where(
            responsibleBox, lossFirstBoxes, lossSecondBoxes)
        # reshape it
        coordinateLoss = tf.reshape(coordinateLoss, [-1, 7, 7, 1])
        # count the loss only if the object is in the groundTruth grid
        # gives a sparse -1x7x7x1 matrix, only one element would be nonzero in
        # each slice
        coordinateLoss = lambdaCoordinate * \
            tf.multiply(groundTruthGrid, coordinateLoss)
        # object loss (loss due to difference in object confidence)
        # only take the objectLoss of the predicted grid with higher IoU is
        # responsible for the object
        objectLoss = tf.square(predictedObjectConfidence - groundTruthGrid)
        objectLoss = tf.where(responsibleBox, objectLoss[
                              :, :, :, 0], objectLoss[:, :, :, 1])
        tempObjectLoss = tf.reshape(objectLoss, [-1, 7, 7, 1])
        objectLoss = tf.multiply(groundTruthGrid, tempObjectLoss)
        # class loss (loss due to misjudgement in class of the object
        # detected
        classLoss = tf.square(predictedClasses - groundTruthClasses)
        classLoss = tf.reduce_sum(
            tf.multiply(groundTruthGrid, classLoss), axis=3)
        classLoss = tf.reshape(classLoss, [-1, 7, 7, 1])
        # no-object loss, decrease the confidence where there is no
        # object in the ground truth
        noObjectLoss = lambdaNoObject * \
            tf.multiply(1 - groundTruthGrid, tempObjectLoss)
        # total loss
        totalLoss = coordinateLoss + objectLoss + classLoss + noObjectLoss
        totalLoss = tf.reduce_mean(tf.reduce_sum(
            totalLoss, axis=[1, 2, 3]), axis=0)
        return totalLoss
    
def YOLO_LOSS_Updated2(y_true, y_pred):
        """
        Calculate the total loss for gradient descent.
        For each ground truth object, loss needs to be calculated.
        It is assumed that each image consists of only one object.
        Predicted
        0-19 CLass prediction
        20-21 Confidence that objects exist in bbox1 or bbox2 of grid
        22-29 Coordinates for bbo1, followed by those of bbox2
        Real
        0-19 Class prediction (One-Hot Encoded)
        20-23 Ground truth coordinates for that box
        24-72 Cell has an object/no object (Only one can be is 1)
        """
        groundTruth = y_true
        predicted = y_pred
        
        predictedParameters = predicted
        predictedClasses = predictedParameters[:, :, :, 0:num_class]
        predictedObjectConfidence = predictedParameters[:, :, :, num_class:num_class + boxes_percell]
        predictedBoxes = predictedParameters[:, :, :, num_class + boxes_percell:]
        # seperate both boxes
        predictedFirstBoxes = predictedBoxes[:, :, :, 0:4]
        predictedSecondBoxes = predictedBoxes[:, :, :, 4:]
        
        groundTruthClasses = groundTruth[:,:,:, :num_class]
        groundTruthBoxes = groundTruth[:,:,:, num_class:num_class+4]
        groundTruthGrid = groundTruth[:,:,:, num_class+4:]
        
        # Calulate loss along the 4th axis, localFirstBoxes -1x7x7x1
        # Think there should be a simpler method to do this
        lossFirstBoxes = tf.reduce_sum(
            tf.square(predictedFirstBoxes - groundTruthBoxes), 3)
        lossSecondBoxes = tf.reduce_sum(
            tf.square(predictedSecondBoxes - groundTruthBoxes), 3)
        # Computing which box (bbox1 or bbox2) is responsible for
        # detection
        IOU = iou_train(predictedFirstBoxes,
                       predictedSecondBoxes, groundTruthBoxes)
        responsibleBox = tf.greater(IOU[:, :, :, 0], IOU[:, :, :, 1])
        # Suppose it is known which iou is greater,
        # coordinate loss (loss due to difference in coordinates of
        # predicted-responsible and real box)
        coordinateLoss = tf.where(
            responsibleBox, lossFirstBoxes, lossSecondBoxes)
        # why do we need to reshape it
        coordinateLoss = tf.reshape(coordinateLoss, [-1, grid_size, grid_size, 1])
        # count the loss only if the object is in the groundTruth grid
        # gives a sparse -1x7x7x1 matrix, only one element would be nonzero in
        # each slice
        coordinateLoss = lambda_coord * \
            tf.multiply(groundTruthGrid, coordinateLoss)
        # object loss (loss due to difference in object confidence)
        # only take the objectLoss of the predicted grid with higher IoU is
        # responsible for the object
        objectLoss = tf.square(predictedObjectConfidence - groundTruthGrid)
        objectLoss = tf.where(responsibleBox, objectLoss[
                              :, :, :, 0], objectLoss[:, :, :, 1])
        tempObjectLoss = tf.reshape(objectLoss, [-1, grid_size, grid_size, 1])
        objectLoss = tf.multiply(groundTruthGrid, tempObjectLoss)
        # class loss (loss due to misjudgement in class of the object
        # detected
        classLoss = tf.square(predictedClasses - groundTruthClasses)
        classLoss = tf.reduce_sum(
            tf.multiply(groundTruthGrid, classLoss), axis=3)
        classLoss = tf.reshape(classLoss, [-1, grid_size, grid_size, 1])
        # no-object loss, decrease the confidence where there is no
        # object in the ground truth
        noObjectLoss = lambda_noobj * \
            tf.multiply(1 - groundTruthGrid, tempObjectLoss)
        # total loss
        totalLoss = coordinateLoss + objectLoss + classLoss + noObjectLoss
        totalLoss = tf.reduce_mean(tf.reduce_sum(
            totalLoss, axis=[1, 2, 3]), axis=0)
        return totalLoss
#%%
class Loss(Layer):
    def __init__(self, num_classes=num_class, cell_size=grid_size, boxes_per_cell=boxes_percell, *args, **kwargs):
        self.num_classes = num_classes
        self.cell_size   = cell_size
        self.boxes_per_cell   = boxes_per_cell

        self.object_scale = 1.0
        self.noobject_scale = 0.5
        self.class_scale = 2.0
        self.coord_scale = 5.0

        super(Loss, self).__init__(*args, **kwargs)

    def classification_loss(self,labels,  classification, response):
        class_delta = response * (labels - classification)
        cls_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='cls_loss') * self.class_scale

        return cls_loss

    def regression_loss(self, regression_target, regression,object_mask):
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (regression_target - regression)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                    name='coord_loss') * self.coord_scale
        reg_loss = tf.constant(0,dtype=tf.float32)
        return reg_loss

    def confidence_loss(self,predict_scales, iou_predict_truth, object_mask):
        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        # object_loss
        object_delta = object_mask * (iou_predict_truth - predict_scales)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),  name='object_loss') * self.object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

        return [object_loss, noobject_loss]


    def call(self, inputs):
        '''
        :param inputs:
        predicts: shape(None, 1470)
        labels shape(None, 7, 7, 25)
        :return:
        '''

        predicts, labels, image_shape = inputs

        index_classification = tf.multiply(tf.pow(self.cell_size, 2), self.num_classes)
        index_confidence = tf.multiply(tf.pow(self.cell_size, 2), self.num_classes + self.boxes_per_cell)


        predict_classes = tf.reshape(predicts[:, :index_classification], [-1, self.cell_size, self.cell_size, self.num_classes])
        predict_scales = tf.reshape(predicts[:, index_classification:index_confidence], [-1, self.cell_size, self.cell_size, self.boxes_per_cell])
        predict_boxes = tf.reshape(predicts[:, index_confidence:], [-1, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        response = tf.reshape(labels[:, :, :, 0], [-1, self.cell_size, self.cell_size, 1])
        regression_labels = tf.reshape(labels[:, :, :, 1:5], [-1, self.cell_size, self.cell_size, 1, 4])
        regression_labels = tf.math.truediv(tf.tile(regression_labels, [1, 1, 1, self.boxes_per_cell, 1]), tf.cast(image_shape[0], dtype=tf.float32))
        classification_labels = labels[:, :, :, 5:]

        offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        offset = tf.constant(offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])

        regression = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                       (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                       tf.square(predict_boxes[:, :, :, :, 2]),
                                       tf.square(predict_boxes[:, :, :, :, 3])])
        regression = tf.transpose(regression, [1, 2, 3, 4, 0])

        iou_predict_truth = self.calc_iou(regression, regression_labels)

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        regression_target = tf.stack([regression_labels[:, :, :, :, 0] * self.cell_size - offset,
                               regression_labels[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(regression_labels[:, :, :, :, 2]),
                               tf.sqrt(regression_labels[:, :, :, :, 3])])
        regression_target = tf.transpose(regression_target, [1, 2, 3, 4, 0])

        # regression loss (localization loss) coord_loss
        coord_loss = self.regression_loss(regression_target, predict_boxes, object_mask)

        # confidence loss
        object_loss, noobject_loss = self.confidence_loss(predict_scales, iou_predict_truth, object_mask)

        # classification loss
        cls_loss = self.classification_loss(classification_labels, predict_classes, response)

        self.add_loss(cls_loss)
        self.add_loss(object_loss)
        self.add_loss(noobject_loss)
        self.add_loss(coord_loss)

        return [coord_loss, object_loss, noobject_loss, cls_loss]

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                      (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                      (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def compute_output_shape(self, input_shape):
        return [(1,), (1,), (1,), (1,)]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]

    def get_config(self):
        return {
            'num_classes' : self.num_classes,
            'cell_size'   : self.cell_size,
        }