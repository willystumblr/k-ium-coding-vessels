
from keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Input, LSTM, Dense, TimeDistributed, Add, Reshape, Multiply, Activation, Lambda



# Define the spatial attention mechanism as a custom layer
class SpatialAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = Conv2D(1, (1, 1), activation='sigmoid')
        super(SpatialAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        conv = self.conv(inputs)
        multiplied = Multiply()([inputs, conv])
        return multiplied

    def compute_output_shape(self, input_shape):
        return input_shape
