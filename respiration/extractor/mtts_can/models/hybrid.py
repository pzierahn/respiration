from tensorflow.keras import backend as K
from keras.layers import (
    Conv2D,
    Conv3D,
    Input,
    AveragePooling2D,
    multiply,
    Dense,
    Dropout,
    Flatten,
    AveragePooling3D
)
from keras.models import Model

from .mtts_can import AttentionMask


def hybrid_can_2d(
        n_frame,
        nb_filters1,
        nb_filters2,
        input_shape_1,
        input_shape_2,
        kernel_size_1=(3, 3, 3),
        kernel_size_2=(3, 3),
        dropout_rate1=0.25,
        dropout_rate2=0.5,
        pool_size_1=(2, 2, 2),
        pool_size_2=(2, 2),
        nb_dense=128):
    diff_input = Input(shape=input_shape_1)
    rawf_input = Input(shape=input_shape_2)

    # Motion branch
    d1 = Conv3D(nb_filters1, kernel_size_1, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size_1, activation='tanh')(d1)

    # App branch
    r1 = Conv2D(nb_filters1, kernel_size_2, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size_2, activation='tanh')(r1)

    # Mask from App (g1) * Motion Branch (d2)
    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = AttentionMask()(g1)
    g1 = K.expand_dims(g1, axis=-1)
    gated1 = multiply([d2, g1])

    # Motion Branch
    d3 = AveragePooling3D(pool_size_1)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size_1, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size_1, activation='tanh')(d5)

    # App branch
    r3 = AveragePooling2D(pool_size_2)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv2D(nb_filters2, kernel_size_2, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size_2, activation='tanh')(r5)

    # Mask from App (g2) * Motion Branch (d6)
    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = AttentionMask()(g2)
    g2 = K.repeat_elements(g2, d6.shape[3], axis=-1)
    g2 = K.expand_dims(g2, axis=-1)
    gated2 = multiply([d6, g2])

    # Motion Branch
    d7 = AveragePooling3D(pool_size_1)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    # Motion Branch
    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(n_frame)(d11)

    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model


def mt_hybrid_can(n_frame, nb_filters1, nb_filters2, input_shape_1, input_shape_2, kernel_size_1=(3, 3, 3),
                  kernel_size_2=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5, pool_size_1=(2, 2, 2),
                  pool_size_2=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape_1)
    rawf_input = Input(shape=input_shape_2)

    # Motion branch
    d1 = Conv3D(nb_filters1, kernel_size_1, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size_1, activation='tanh')(d1)

    # App branch
    r1 = Conv2D(nb_filters1, kernel_size_2, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size_2, activation='tanh')(r1)

    # Mask from App (g1) * Motion Branch (d2)
    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = AttentionMask()(g1)
    g1 = K.expand_dims(g1, axis=-1)
    gated1 = multiply([d2, g1])

    # Motion Branch
    d3 = AveragePooling3D(pool_size_1)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size_1, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size_1, activation='tanh')(d5)

    # App branch
    r3 = AveragePooling2D(pool_size_2)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv2D(nb_filters2, kernel_size_2, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size_2, activation='tanh')(r5)

    # Mask from App (g2) * Motion Branch (d6)
    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = AttentionMask()(g2)
    g2 = K.repeat_elements(g2, d6.shape[3], axis=-1)
    g2 = K.expand_dims(g2, axis=-1)
    gated2 = multiply([d6, g2])

    # Motion Branch
    d7 = AveragePooling3D(pool_size_1)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    # Motion Branch
    d9 = Flatten()(d8)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(n_frame, name='output_1')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(n_frame, name='output_2')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model
