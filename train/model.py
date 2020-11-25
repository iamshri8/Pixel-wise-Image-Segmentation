from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

def dice_coefficient(labels, logits):

    """ Calculate dice coefficient metric for model accuracy."""

    return (2. * K.sum(labels * logits)) / (K.sum(labels) + K.sum(logits))


def unet(num_classes, in_shape, lrate, decay_rate, vgg_path, dropout_rate):

    """
    Model Definition for U-Net architecture for semantic segmentation.

    Arguments:
    num_classes -- number of classes
    in_shape -- shape of the input image
    lrate -- initial learning rate
    decay_rate -- learning rate decay

    Returns:
    model -- returns the defined Keras model object

    """
    in_img = Input(in_shape)

    # Down sampling block 1
    x = Conv2D(64, (3, 3), padding='same')(in_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    block1 = Activation('relu')(x)

    x = MaxPooling2D()(block1)
    x = Dropout(dropout_rate)(x)

    # Down sampling block 2
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    block2 = Activation('relu')(x)

    x = MaxPooling2D()(block2)
    x = Dropout(dropout_rate)(x)

    # Down sampling block 3
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    block3 = Activation('relu')(x)

    x = MaxPooling2D()(block3)
    x = Dropout(dropout_rate)(x)

    # Down sampling block 4
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    block4 = Activation('relu')(x)

    x = MaxPooling2D()(block4)
    x = Dropout(dropout_rate)(x)

    # Bottleneck block
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for_pretrained_weight = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_path is not None:
        vgg16 = Model(in_img, for_pretrained_weight)
        vgg16.load_weights(vgg_path, by_name=True)

    # Up sampling 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block4])
    x = Dropout(dropout_rate)(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Up sampling 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block3])
    x = Dropout(dropout_rate)(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Up sampling 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block2])
    x = Dropout(dropout_rate)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Up sampling 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block1])
    x = Dropout(dropout_rate)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last convolution
    output = Conv2D(num_classes, (3, 3), activation='softmax', padding='same', name='output')(x)

    # instantiate the model variables
    model = Model(inputs=in_img, outputs=output)

    #configure the model for training
    model.compile(optimizer=Adam(lr=lrate, decay=decay_rate),
                  loss='categorical_crossentropy',
                  metrics = [dice_coefficient])

    return model
