import sys
sys.path.append('/content/drive/MyDrive')
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201,ResNet152V2
from tensorflow.keras import layers

def make_nn_model_efficientb0(image_size, trainb=False):
    print('Making our deep cnn model.....')

    # Girdi tanımı
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    # EfficientNetV2-B0 base model
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Transfer learning: katmanları dondur veya aç
    base_model.trainable = trainb

    # Ek katmanlar
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4, name='dropout')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Model tanımı
    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='efficientb0_model')

    cnn_model.summary()
    print('model built!')
    return cnn_model
def make_nn_model_efficientb7(image_size, trainb=False):
    print('Making our deep cnn model EFFICIENT B6.....')

    # Girdi tanımı
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    # EfficientNetV2-B0 base model
    base_model = tf.keras.applications.EfficientNetB6(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Transfer learning: katmanları dondur veya aç
    base_model.trainable = trainb

    # Ek katmanlar
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4, name='dropout')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Model tanımı
    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='efficientb6_model')

    cnn_model.summary()
    print('model built!')
    return cnn_model
def make_nn_model_mobileNet(image_size, trainb=False):
    print('Making our deep cnn model MobileNet.....')

    # Girdi tanımı
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    # EfficientNetV2-B0 base model
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Transfer learning: katmanları dondur veya aç
    base_model.trainable = trainb

    # Ek katmanlar
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4, name='dropout')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Model tanımı
    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mobilenetv2_model')

    cnn_model.summary()
    print('model built!')
    return cnn_model
def make_nn_model_mobileNetV3(image_size, trainb=False):
    print('Making our deep cnn model MobileNet.....')

    # Girdi tanımı
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    # EfficientNetV2-B0 base model
    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Transfer learning: katmanları dondur veya aç
    base_model.trainable = trainb

    # Ek katmanlar
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4, name='dropout')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    # Model tanımı
    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mobilenetv2_model')

    cnn_model.summary()
    print('model built!')
    return cnn_model
def make_nn_model(image_size,model_handle,trainb=False):
    print('Making our deep cnn model.....')
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
        hub.KerasLayer(model_handle, trainable=trainb, name='base'), #önceden eğitilmiş model kullanılmış trainable true yapılarak transfer learning kullanılmış oldu.Transfer öğreniminde, iki farklı veri türü için aynı mimari (EfficientNetB0 temelli) ama ayrı modeller (PNG ve DICOM için) kullanılmıştır.
        tf.keras.layers.Dense(512, activation='relu', name='fc1'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4, name='dropout'),
        tf.keras.layers.Dense(128, activation='relu', name='fc2'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ], name='cnn_model')

    cnn_model.build((None, image_size, image_size, 3))
    cnn_model.summary()
    print('model built!')
    return cnn_model

  #densenet
def make_nn_model_dense(image_size,trainb=False):
    print('Making DenseNet201 model...')

    base_model = DenseNet201(
    include_top=False,
    weights='imagenet',
    input_shape=(image_size, image_size, 3),
    name="base",
    pooling='avg'  # <-- Burası önemli!
    )
    base_model.trainable = trainb

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=trainb)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    #model.get_layer(index=1)._name = 'base'  # base_model'a isim verme hilesi
    model.summary()
    return model

def make_nn_model_resnet152(image_size, trainb=False):
    print('Making ResNet152V2 model...')

    base_model = ResNet152V2(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3),
        name="base",
        pooling='avg'
    )
    base_model.trainable = trainb

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=trainb)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    #model.get_layer(index=1)._name = 'base'  # base_model'a isim verme hilesi
    model.summary()
    return model
