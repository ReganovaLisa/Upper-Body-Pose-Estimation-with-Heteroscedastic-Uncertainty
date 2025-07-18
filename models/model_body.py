import tensorflow as tf


def hhp_net_body(mean = 0, std = 0.05, alpha = 1, only_hands = True):
    """
    Create a model based on the experiment parameter passed as input

    Args:
        :mean (float): Mean value of the confidences in the training set;
            this value is used to normalize the confidence values
        :std (float): Standard deviation value of the confidences in the training set;
            this value is used to normalize the confidence values
    Returns:
        :model_exp (tf.keras.models.Model): the model built accordingly to the experiment number
    """
    #let's first consider only hands keypoints: shoulders, elbows and wrists; otherwise also neck
    if only_hands:
        shape_of_input = (18,1)
    else:
        shape_of_input = (21,1)
    
    c_start = 2*shape_of_input[0]/3    

    input = tf.keras.layers.Input(shape=shape_of_input)

    num_filters = 5

    input_x = tf.keras.layers.Lambda(lambda k: tf.cast(k[:, 0:c_start:2], tf.float32), input_shape=shape_of_input)(input)
    model_x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=1, strides=1, use_bias=True, kernel_initializer='random_normal')(input_x)
    x1 = tf.keras.layers.LeakyReLU(0.1)(model_x)
    x1 = tf.keras.layers.Flatten()(x1)

    input_y = tf.keras.layers.Lambda(lambda k: tf.cast(k[:, 1:c_start:2], tf.float32), input_shape=shape_of_input)(input)
    model_y = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=1, strides=1, use_bias=True, kernel_initializer='random_normal')(input_y)
    y1 = tf.keras.layers.LeakyReLU(0.1)(model_y)
    y1 = tf.keras.layers.Flatten()(y1)

    input_c = tf.keras.layers.Lambda(lambda k: tf.cast(1 / std * (k[:, c_start:shape_of_input[0]] - mean), tf.float32), input_shape=shape_of_input)(input)
    input_c = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=1, strides=1, use_bias=False, kernel_initializer='random_normal')(input_c)
    c1 = tf.nn.sigmoid(input_c)
    c1 = tf.keras.layers.Flatten()(c1)

    x = [x1, c1]
    y = [y1, c1]

    x_mul = tf.keras.layers.Multiply()(x)
    y_mul = tf.keras.layers.Multiply()(y)

    xy_merge = tf.keras.layers.Concatenate()([x_mul, y_mul])

    d0 = tf.keras.layers.Dense(250 * alpha, kernel_initializer='random_normal')(xy_merge)
    d0 = tf.keras.layers.LeakyReLU(0.1)(d0)

    d0 = tf.keras.layers.Dense(200 * alpha, kernel_initializer='random_normal')(d0)
    d0 = tf.keras.layers.LeakyReLU(0.1)(d0)

    d0 = tf.keras.layers.Dense(150 * alpha, kernel_initializer='random_normal')(d0)
    d0 = tf.keras.layers.LeakyReLU(0.1)(d0)

    yaw = tf.keras.layers.Dense(name='yaw', units=2)(d0)
    pitch = tf.keras.layers.Dense(name='pitch', units=2)(d0)
    roll = tf.keras.layers.Dense(name='roll', units=2)(d0)

    model_exp = tf.keras.models.Model(inputs=input, outputs=[yaw, pitch, roll])

    return model_exp
