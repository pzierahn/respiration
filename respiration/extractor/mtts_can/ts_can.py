from keras.layers import Conv2D, Conv3D, Input, AveragePooling2D, \
    multiply, Dense, Dropout, Flatten, AveragePooling3D
from keras.models import Model

from respiration.extractor.mtts_can import AttentionMask, tsm_cov_2d


def TS_CAN(
        n_frame,
        nb_filters1,
        nb_filters2,
        input_shape,
        kernel_size=(3, 3),
        dropout_rate1=0.25,
        dropout_rate2=0.5,
        pool_size=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = tsm_cov_2d(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = tsm_cov_2d(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = AttentionMask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = tsm_cov_2d(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = tsm_cov_2d(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = AttentionMask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(1)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model
