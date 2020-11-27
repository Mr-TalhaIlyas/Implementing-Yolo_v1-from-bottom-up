import tensorflow as tf

if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0':
    import keras.backend as K
    from keras.layers import Layer, InputSpec, MaxPool2D, MaxPooling2D
if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0-rc2':
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Layer, InputSpec, MaxPool2D, MaxPooling2D
from keras.utils.generic_utils import get_custom_objects


class Dropout(Layer):
    
    '''
    it also acts as a parent class for the SpatialDropout
     https://github.com/keras-team/keras/issues/8826
    '''
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = K.variable(min(1., max(0., rate)))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < K.get_value(self.rate) < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs, training=training)#inputs
        return inputs

    def get_config(self):
        config = {'rate': K.get_value(self.rate),
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

#%%
#%%
class SpatialDropout2D(Dropout):

    def __init__(self, rate, data_format=None, **kwargs):
        super(SpatialDropout2D, self).__init__(rate, **kwargs)
        if data_format is None:
          data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
          raise ValueError('data_format must be in '
                           '{"channels_last", "channels_first"}')
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=4)
    
    def _get_noise_shape(self, inputs):
        #input_shape = array_ops.shape(inputs)
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
          return (input_shape[0], input_shape[1], 1, 1)
        elif self.data_format == 'channels_last':
          return (input_shape[0], 1, 1, input_shape[3])
#%%
class DropBlock2D(Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 rate,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
         Replace it with the rate = 1 - keep_prob
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.rate = K.variable(rate)
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.supports_masking = True
        self.height = self.width = self.ones = self.zeros = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':# b x ch x h x w 
            self.height, self.width = input_shape[2], input_shape[3]
        else:# {channel last} b x h x w x ch
            self.height, self.width = input_shape[1], input_shape[2]
        self.ones = K.ones((self.height, self.width), name='ones')
        self.zeros = K.zeros((self.height, self.width), name='zeros')
        super().build(input_shape)

    def get_config(self):
        config = {'block_size': self.block_size,
                  'rate': self.rate,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self):
        """Get the number of activation units to drop"""
        height, width = K.cast(self.height, K.floatx()), K.cast(self.width, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())# can be replaces by tf.variable with trainable == False
        return ((self.rate) / (block_size ** 2)) *\
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self):# so that dropblock is completely inside the image
        positions = K.concatenate([
            K.expand_dims(K.tile(K.expand_dims(K.arange(self.height), axis=1), [1, self.width]), axis=-1),
            K.expand_dims(K.tile(K.expand_dims(K.arange(self.width), axis=0), [self.height, 1]), axis=-1),
        ], axis=-1)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < self.height - half_block_size,
                        positions[:, :, 1] < self.width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            self.ones,
            self.zeros,
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        '''
        The K.random_binomial will creat a tensor of given shape having 1 at random locations 
        adopted from binomial distribution and the probability of those locations having 1 will 
        be decided  by p. Everywhere else is will be 0
        '''
        mask = K.random_binomial(shape, p=self._get_gamma())
        mask *= self._compute_valid_seed_region()           
        mask = MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask # so that 0's are only at places where we wanna remove inputs

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            if self.sync_channels: # if wanna use same dropblock for all channesl
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])# sends (b_size, h, w, 1)
            else:
                mask = self._compute_drop_mask(shape)# sends (b_size, h, w, ch)
            outputs = outputs * mask *\
                (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask)) # normalizes the features
            if self.data_format == 'channels_first': # for rearranging in channal first format after processing
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)#inputs
    
    
