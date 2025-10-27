#data_loader.py
import sys
sys.path.append('/content/drive/MyDrive')
import tensorflow as tf
import os
import glob
from sklearn.model_selection import train_test_split
from functools import partial
import pandas as pd
from preprocess import prepare_roboflow_dataset,prepare_rsna_dataset,load_and_preprocess_image,prepare_busi_dataset,prepare_bcfpp_dataset
from config import IMG_SIZE,BATCH_SIZE,test_split,val_split,seed
import matplotlib.pyplot as plt
import numpy as np


#breakhis starts
def load_breakhis_data(path):
  # Eğitim ve test CSV dosyalarının tam yollarını oluşturun
  data_path = os.path.join(path, "Folds.csv")
  class_names = ['benign', 'malignant'] #benign=0, malignan=1 oluyor
  # Eğitim ve test verilerini yükleyin
  dataBreak = pd.read_csv(data_path)

  dataBreak = dataBreak.rename(columns={'filename': 'path'})
  dataBreak['label'] = dataBreak.path.apply(lambda x: x.split('/')[3])
  dataBreak['label_int'] = dataBreak.label.apply(lambda x: class_names.index(x))
  dataBreak['filename'] = dataBreak.path.apply(lambda x: x.split('/')[-1])

  test_images = dataBreak.groupby(by='label').sample(frac=test_split, random_state=seed) #yüzde 10unu teste aldım
  train_images = dataBreak.drop(test_images.index).reset_index(drop=True)
  test_images = test_images.reset_index(drop=True)
  test_images['set'] = 'test'

  train_images["path"] = train_images["path"].astype(str).apply(lambda x: os.path.join(path, "BreaKHis_v1", x))
  test_images["path"] = test_images["path"].astype(str).apply(lambda x: os.path.join(path, "BreaKHis_v1", x))
  # Pandas'ın satırda gösterdiği maksimum uzunluğu arttır
  pd.set_option('display.max_colwidth', None)
  validation_images, train_images = make_validation(train_images) #genel
  trainBreak_dataset, valBreak_dataset = load_trains(train_images, validation_images)

   # Test datasetini oluştur
  testBreak_dataset = tf.data.Dataset.from_tensor_slices((test_images.path, test_images.label_int))
  testBreak_dataset = (
      testBreak_dataset
      .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      .map(image_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      .batch(BATCH_SIZE)
      .prefetch(tf.data.experimental.AUTOTUNE)
  )


  return trainBreak_dataset,valBreak_dataset,testBreak_dataset

def make_validation(client_dataset):
   client_dataset = client_dataset.reset_index(drop=True)
   val_images = client_dataset.sample(frac=val_split, random_state=seed)
   client_dataset = client_dataset.drop(val_images.index).reset_index(drop=True)
   val_images = val_images.reset_index(drop=True)
   client_dataset['set'] = 'train' # yeni bir column tanımlıyoruz ::set:: adında. Veriyi tanımlıyor
   val_images['set'] = 'validation'
   # unsampling data
   # Bu kısım önemli: Bias ve oversampling önlemek için ekledim. Yani azınlık sınıfındaki örneklerin
   # sayısını çoğunluk sınıfıyla dengelemek için azınlığı artırmayı içerir. Her iki sınıfta benzer temsile
   # sahip olana kadar azınlık sınıfındaki örnekler rastgele çoğaltılır (veya tersi)
   max_count = np.max(client_dataset.label.value_counts())
   min_count = np.min(client_dataset.label.value_counts())
   client_dataset = client_dataset.groupby('label_int').sample(n=max_count, replace=True)
   client_dataset = client_dataset.reset_index(drop=True)
   return val_images,client_dataset
# Görüntüyü yükleyen fonksiyon
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    return image, label

# Görüntüyü uygun boyuta getirme ve TensörFlow'un hazır ön işleme fonksiyonlarını kullanma
def image_reshape(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  # 224x224 boyutuna getir
    image = tf.keras.applications.efficientnet.preprocess_input(image)  # TensorFlow'un EfficientNet preprocess fonksiyonunu kullan
    return image, label

# Eğitim ve doğrulama veri setini yükleyen fonksiyon
def load_trains(train_images, validation_images):
    # TensorFlow veri kümelerini oluştur
    load_tr = tf.data.Dataset.from_tensor_slices((train_images.path, train_images.label_int))
    load_val = tf.data.Dataset.from_tensor_slices((validation_images.path, validation_images.label_int))

    # Eğitim veri setini işle
    train_images = (
        load_tr
        .shuffle(len(train_images))  # Karıştır
        .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Görüntüyü oku
        .map(image_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Boyutlandır ve normalize et
        .batch(BATCH_SIZE)  # Batch işle
        .prefetch(tf.data.experimental.AUTOTUNE)  # Performans için prefetch
    )

    # Doğrulama veri setini işle
    validation_images = (
        load_val
        .shuffle(len(validation_images))  # Karıştır
        .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Görüntüyü oku
        .map(image_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Boyutlandır ve normalize et
        .batch(BATCH_SIZE)  # Batch işle
        .prefetch(tf.data.experimental.AUTOTUNE)  # Performans için prefetch
    )

    return train_images, validation_images

#end breakhis
#rsna start
def load_rsna_data(path):
  # Eğitim ve test CSV dosyalarının tam yollarını oluşturun
  train_csv_path = os.path.join(path, "train.csv")
  test_csv_path = os.path.join(path, "test.csv")

  # Eğitim ve test verilerini yükleyin
  train_data = pd.read_csv(train_csv_path)
  test_data = pd.read_csv(test_csv_path)

  # Görüntü dosyalarının yolu
  train_images_dir = os.path.join(path, "output/train")  # 'output/train' içindeki görüntüler
  test_images_dir = os.path.join(path, "output/test")   # 'output/test' içindeki görüntüler

  # train.csv'deki image_id sütunundan tam yolları oluştur
  train_data["image_path"] = train_data.apply(
      lambda row: os.path.join(train_images_dir, f"{row['patient_id']}_{row['image_id']}.png"), axis=1
  )
  # Eğitim ve doğrulama verilerini bölme
  train_df, val_df = train_test_split(train_data, test_size=test_split,stratify=train_data['cancer'], random_state=seed) #random state hep aynı test verisini seçer %20 test
  train_df = train_df.reset_index(drop=True)
  val_df = val_df.reset_index(drop=True)
  # Sınıf başına eşit örnek sayısı belirle
  sample_per_class = 5490  # Toplam 8.500 olacak (3500 'cancer' + 3500 'non-cancer')
  total_train_samples=sample_per_class * 2
  # Örnek sayıları
  val_sample_total = int(total_train_samples * 0.20)  # %20 validation
  test_sample_total = int(total_train_samples * 0.10) # %10 test
  train_sample_total = total_train_samples - val_sample_total - test_sample_total  # Kalan %70 eğitim

  # Her sınıf için örnek sayısı (eğitim, val, test)
  # Öncelikle sınıf dağılımına göre eşit paylaştırıyoruz
  num_classes = 2  # cancer 0 ve 1 için
  val_sample_per_class = val_sample_total // num_classes
  test_sample_per_class = test_sample_total // num_classes
  train_sample_per_class = train_sample_total // num_classes

  # Sınıflara göre örnek seç
  cancer_samples = train_df[train_df["cancer"] == 1].sample(n=train_sample_per_class, replace=True, random_state=42)
  non_cancer_samples = train_df[train_df["cancer"] == 0].sample(n=train_sample_per_class, replace=False, random_state=42)

  # Dengeli dataset oluştur
  balanced_train_df = pd.concat([cancer_samples, non_cancer_samples]).sample(frac=1, random_state=seed)  # Karıştır
  # Validation seti için de dengeleme train yüzde 20si 1680
  val_cancer_samples = val_df[val_df["cancer"] == 1].sample(n=val_sample_per_class, replace=True, random_state=seed)
  val_non_cancer_samples = val_df[val_df["cancer"] == 0].sample(n=val_sample_per_class, replace=False, random_state=seed)

  balanced_val_df = pd.concat([val_cancer_samples, val_non_cancer_samples]).sample(frac=1, random_state=seed).reset_index(drop=True)
  #test
  # Test verisini ayır 1500 yaklaşık yüzde 20
  test_cancer_samples = balanced_train_df[balanced_train_df["cancer"] == 1].sample(n=test_sample_per_class, random_state=seed)
  test_non_cancer_samples = balanced_train_df[balanced_train_df["cancer"] == 0].sample(n=test_sample_per_class, random_state=seed)

  # Dengeli test verisini birleştir
  balanced_test_df = pd.concat([test_cancer_samples, test_non_cancer_samples]).sample(frac=1, random_state=seed)

  #testRSNA_dataset = testRSNA_dataset.map(lambda x: load_and_preprocess_image(x, 0)).batch(32)  # Eskisi Etiket 0 çünkü test
  # Test veri etiketlerini elde et

  test_data = balanced_test_df[["image_id", "cancer"]]   

  # Test verisini çıkar (eğitim için kalan kısmı)
  remaining_train_df = balanced_train_df.drop(balanced_test_df.index)

  # Eğitim verisini oluştur
  trainRSNA_dataset = prepare_rsna_dataset(remaining_train_df, train_images_dir)
  # Dataset oluşturma
  #eskisi#trainRSNA_dataset = prepare_rsna_dataset(balanced_train_df, train_images_dir)
  valRSNA_dataset = prepare_rsna_dataset(balanced_val_df, train_images_dir)
  testRSNA_dataset = prepare_rsna_dataset(balanced_test_df, train_images_dir)

  return trainRSNA_dataset,valRSNA_dataset,testRSNA_dataset,test_data
#rsna ends
#roboflow starts
# Test datasetini augmentation’lı oluştur
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label
def load_roboflow_data(path_ds):
  # Görüntü dosyalarının yolu
  train_images_dir = os.path.join(path_ds, "train")  # 'output/train' içindeki görüntüler
  test_images_dir = os.path.join(path_ds, "test")   # 'output/test' içindeki görüntüler
  val_images_dir = os.path.join(path_ds, "valid")   # 'output/test' içindeki görüntüler

  #  Eğitim ve doğrulama veri kümesini oluştur
  train_image_paths, train_labels = get_image_paths_and_labelsRobo(train_images_dir)
  val_image_paths, val_labels = get_image_paths_and_labelsRobo(val_images_dir)

  # Eğitim ve doğrulama veri kümesini böl
  train_paths, test_paths, train_labels_split, test_labels = train_test_split(train_image_paths, train_labels, test_size=test_split, random_state=seed)

  # Datasetleri oluştur
  trainRobo_dataset = prepare_roboflow_dataset(train_paths, train_labels_split)
  valRobo_dataset = prepare_roboflow_dataset(val_image_paths, val_labels)
  testRobo_dataset=prepare_roboflow_dataset(test_paths, test_labels)

  return trainRobo_dataset,valRobo_dataset,testRobo_dataset

def get_label_from_pathRobo(image_path):
    # Resmin etiketini, dosya yolundaki üst klasörden alıyoruz (0 veya 1)
    return int(image_path.split(os.sep)[-2])
def get_image_paths_and_labelsRobo(directory):
    # 'train' dizinindeki alt klasörleri (0 ve 1) tarıyoruz
    image_paths = glob.glob(os.path.join(directory, "*", "*.jpg"))  # Tüm JPG dosyaları (alt klasörler dahil:0 ve 1 alt klasörü var)
    labels = [get_label_from_pathRobo(img) for img in image_paths]
    return image_paths, labels

#load BUSI Dataset
def load_busi_data(path_ds):
  data_dir = os.path.join(path_ds, "Dataset_BUSI_with_GT")
  # Sınıflar ve etiketleri
  class_map = {"normal": 0,"benign": 0,"malignant": 1,}
  image_paths = []
  labels = []
  image_size=224
  batch_size=32
  # 3. Her klasördeki görüntüleri listele ve etiketle
  for class_name, label in class_map.items():
      class_dir = os.path.join(data_dir, class_name)
      for fname in os.listdir(class_dir):
          fpath = os.path.join(class_dir, fname)
          fname_lower = fname.lower()
          if os.path.isfile(fpath) and fname_lower.endswith(".png") and "mask" not in fname_lower:
              image_paths.append(fpath)
              labels.append(label)
    # Eğitim, doğrulama, test olarak ayır
  X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_split, random_state=seed, stratify=labels)
  X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=seed, stratify=y_temp)

  train=prepare_busi_dataset(X_train, y_train,image_size,batch_size)
  val= prepare_busi_dataset(X_val, y_val,image_size,batch_size)
  test= prepare_busi_dataset(X_test, y_test,image_size,batch_size)
  return train,val,test
#BCFPP DATASET
def load_bcfpp_data(jlb_path,data):
    image_size=224
    batch_size=32
    # Veriyi yükle
    images, labels = data[0], data[1]

    labels = np.where(labels == 0, 1, 0) # dönüşüm yaptım sitesinde etiket dağılımları:
    #The second one is for labels which contains 3 classes (0 for Cancer, 1 for Benign and 2 for normal)
    #benimki cancer:1 diğerleri 0
    
    # Normalize et ve kanal ekle
    images = np.expand_dims(images, axis=-1).astype('float32') / 255.0
    labels = labels.astype('int32')

    # Eğitim, test, val böl
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_split, random_state=seed, stratify=labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=seed, stratify=y_temp)

    # Dataset'leri hazırla
    train_ds = prepare_bcfpp_dataset(X_train, y_train,image_size,batch_size)
    val_ds = prepare_bcfpp_dataset(X_val, y_val,image_size,batch_size)
    test_ds = prepare_bcfpp_dataset(X_test, y_test,image_size,batch_size)

    return train_ds, val_ds, test_ds
