from keras.layers import (
    Conv2D,
    Conv3D,
    Input,
    AveragePooling2D,
    AveragePooling3D,
    multiply,
    Dense,
    Dropout,
    Flatten
)

from keras.models import Model
from .attention_mask import AttentionMask


def can_2d(
        nb_filters1,
        nb_filters2,
        input_shape: tuple[int, int, int],
        kernel_size: tuple[int, int] = (3, 3),
        dropout_rate1: float = 0.25,
        dropout_rate2: float = 0.5,
        pool_size: tuple[int, int] = (2, 2),
        nb_dense: int = 128) -> Model:
    """
    Convolutional Attention Network (CAN) model for 2D data.
    :param nb_filters1:
    :param nb_filters2:
    :param input_shape:
    :param kernel_size:
    :param dropout_rate1:
    :param dropout_rate2:
    :param pool_size:
    :param nb_dense:
    :return:
    """

    # Define shared convolutional layers
    conv_layer1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')
    conv_layer2 = Conv2D(nb_filters1, kernel_size, activation='tanh')
    conv_layer3 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')
    conv_layer4 = Conv2D(nb_filters2, kernel_size, activation='tanh')

    # Input layers
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    # First convolutional block (shared for both inputs)
    d1 = conv_layer1(diff_input)
    d2 = conv_layer2(d1)
    r1 = conv_layer1(rawf_input)
    r2 = conv_layer2(r1)

    # Attention mechanism and gating
    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = AttentionMask()(g1)
    gated1 = multiply([d2, g1])

    # Shared pooling and dropout
    pool_dropout = lambda x: Dropout(dropout_rate1)(AveragePooling2D(pool_size)(x))
    d4 = pool_dropout(gated1)
    r4 = pool_dropout(r2)

    # Second convolutional block (shared for both inputs)
    d5 = conv_layer3(d4)
    d6 = conv_layer4(d5)
    r5 = conv_layer3(r4)
    r6 = conv_layer4(r5)

    # Attention mechanism and gating
    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = AttentionMask()(g2)
    gated2 = multiply([d6, g2])

    # Shared pooling and dropout
    d8 = pool_dropout(gated2)

    # Fully connected layers
    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(1)(d11)

    # Create and return the model
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model


def can_3d(
        n_frame,
        nb_filters1,
        nb_filters2,
        input_shape,
        kernel_size=(3, 3, 3),
        dropout_rate1=0.25,
        dropout_rate2=0.5,
        pool_size=(2, 2, 2),
        nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(d1)

    # Appearance Branch
    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    g1 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r2)
    g1 = AttentionMask()(g1)
    gated1 = multiply([d2, g1])
    d3 = AveragePooling3D(pool_size)(gated1)

    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(d5)

    r3 = AveragePooling3D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(r5)
    g2 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r6)
    g2 = AttentionMask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling3D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)
    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(n_frame)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model
