import sys
sys.path.append('/content/drive/MyDrive')
import tensorflow as tf
import os
from tensorflow.keras.applications.efficientnet import preprocess_input
# Ortak prepare_dataset fonksiyonu
def load_and_preprocess_image(img_path, label=None, img_size=(224, 224)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)  # PNG yükleme
    img = tf.image.resize(img, img_size)       # 224x224 boyutlandırma
    img = preprocess_input(img)                # EfficientNetB0 normalizasyonu
    return (img, label) if label is not None else img
# Roboflow veri kümesinin işlemesi
def prepare_roboflow_dataset(image_paths, labels=None, img_size=(224, 224), batch_size=32):
    # Etiket varsa (labels argümanı sağlanmışsa), image_paths ve labels ile dataset oluştur
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Etiket yoksa sadece image_paths ile dataset oluştur
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda x: load_and_preprocess_image(x), num_parallel_calls=tf.data.AUTOTUNE)

    # Dataset'i karıştır, batch'le ve prefetch ile optimize et
    #dataset = dataset.shuffle(len(image_paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
#RSNA
def prepare_rsna_dataset(data, images_dir, img_size=(224, 224), batch_size=32):
    # Görüntü yollarını ve etiketleri çıkarma
    image_paths = data.apply(lambda row: os.path.join(images_dir, f"{row['patient_id']}_{row['image_id']}.png"), axis=1)
    labels = data["cancer"].values  # 'cancer' sütunundaki etiketleri al

    # TensorFlow Dataset oluşturma
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
#BUSI
def prepare_busi_dataset(paths, labels, image_size,batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda x, y: process_busi_image(x, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 6. Görseli okuma ve boyutlandırma
def process_busi_image(file_path, label, image_size):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = image / 255.0  # Normalize et
    return image, tf.cast(label, tf.int32)

#BCFPP
def prepare_bcfpp_dataset(images, labels,image_size,batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: (
    tf.image.grayscale_to_rgb(tf.image.resize(x, [image_size, image_size])),
      tf.reshape(tf.cast(y, tf.float32), [-1])  # tek boyutlu
    ), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
