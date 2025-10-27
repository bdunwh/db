import sys
sys.path.append('/content/drive/MyDrive')
from modelbuilder import make_nn_model_mobileNet,make_nn_model_mobileNetV3,make_nn_model_efficientb0,make_nn_model_efficientb7,make_nn_model_dense,make_nn_model_resnet152
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback,ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from config import IMG_SIZE,BATCH_SIZE
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryFocalCrossentropy

def train_deep_learning_model_genel(model_name,model_handle,train_images, val_train_images, model_checkpoint, csv_logger,test_dataset,test_labels, epochs=10, trainb=False):
    metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')]

    # Test callback
    test_cb = TestEvaluationCallback(test_dataset, test_labels)
    # EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='val_accuracy',     # Gelişmenin izleneceği metrik
        patience=10,             # 10 epoch boyunca gelişme olmazsa durur
        restore_best_weights=True  # En iyi model ağırlıklarını geri yükler
    )
    
    learn = 1e-3
      # İstemci modeli
    if model_name == 'EFFB0' :
      train_model = make_nn_model_efficientb0(IMG_SIZE, trainb)
    elif model_name == 'EFFB7':
      train_model = make_nn_model_efficientb7(IMG_SIZE, trainb)
    elif model_name == 'DENSE':
      train_model = make_nn_model_dense(IMG_SIZE, trainb)
    elif model_name == 'REST':
      train_model = make_nn_model_resnet152(IMG_SIZE,trainb)
    elif model_name == 'MOBILE':
      learn=1e-4
      train_model = make_nn_model_mobileNet(IMG_SIZE, trainb)
    elif model_name == 'MOBV3':
      learn=1e-4
      train_model = make_nn_model_mobileNetV3(IMG_SIZE, trainb)


    if model_name=='DENSE' or model_name=='REST':
      # 2. aşama: baz modelin son katmanları açılmış, kalan epochlar
      base_model = train_model.get_layer('base')
      for layer in base_model.layers[:-30]:
          layer.trainable = False
      for layer in base_model.layers[-30:]:
          layer.trainable = True
    # İstemci modeli
    train_model.compile(optimizer=AdamW(learning_rate=learn, weight_decay=1e-4), loss='binary_crossentropy', metrics=metrics)
    # Modeli eğitme
    train_history = train_model.fit(train_images, epochs=epochs, batch_size=BATCH_SIZE, verbose=1,
                                      callbacks=[model_checkpoint, csv_logger,ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6), F1ScoreCallback(),test_cb,early_stop],
                                      validation_data=val_train_images)
    return train_model, [train_history]

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Eğitim F1-score hesaplama
        precision = logs.get('precision', 0)
        recall = logs.get('recall', 0)
        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

        # Validation (Doğrulama) F1-score hesaplama
        val_precision = logs.get('val_precision', 0)
        val_recall = logs.get('val_recall', 0)
        val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + tf.keras.backend.epsilon())

        # Ekrana bas
        print(f"Epoch {epoch + 1}: Train F1-score = {f1_score:.4f}, Val F1-score = {val_f1_score:.4f}")

        # Loglara ekle
        logs['f1_score'] = f1_score
        logs['val_f1_score'] = val_f1_score

class TestEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset,test_labels):
        super(TestEvaluationCallback, self).__init__()
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        results  = self.model.evaluate(self.test_dataset, verbose=0)
        test_acc=results[1];
        test_loss=results[0];

          # Log ekleme
        logs['test_accuracy'] = test_acc
        logs['test_loss'] = test_loss

        print(f"Epoch {epoch+1}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
def train_deep_learning_model(model_handle,train_images, val_train_images, model_checkpoint, csv_logger,test_dataset,test_labels, epochs=10, trainb=False):
    #metrics
    metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')]
    # Test callback
    test_cb = TestEvaluationCallback(test_dataset, test_labels)
    # EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='accuracy',     # Gelişmenin izleneceği metrik
        patience=10,             # 10 epoch boyunca gelişme olmazsa durur
        restore_best_weights=True  # En iyi model ağırlıklarını geri yükler
    )

    # İstemci modeli
    train_model = make_nn_model(IMG_SIZE,model_handle, trainb) #fine tuning için true yapılacak
    train_model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss='binary_crossentropy', metrics=metrics)
    
    # Modeli eğitme
    train_history = train_model.fit(train_images, epochs=epochs, batch_size=BATCH_SIZE, verbose=1,
                                     callbacks=[model_checkpoint, csv_logger,ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6), F1ScoreCallback(),test_cb,early_stop],
                                     validation_data=val_train_images)


    return train_model, [train_history]
      
#densenet
def train_deep_learning_model_dense(train_images, val_train_images, model_checkpoint, csv_logger,test_dataset,test_labels, epochs=10, trainb=False):
    #metrics
    metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')]
    image_size=224
    # Test callback
    test_cb = TestEvaluationCallback(test_dataset, test_labels)
     # EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='accuracy',     # Gelişmenin izleneceği metrik
        patience=10,             # 5 epoch boyunca gelişme olmazsa durur
        restore_best_weights=True  # En iyi model ağırlıklarını geri yükler
    )
    # İstemci modeli
    train_model = make_nn_model_dense(image_size, trainb)
    train_model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss='binary_crossentropy', metrics=metrics)
    train_history1 = train_model.fit(train_images, epochs=2, batch_size=BATCH_SIZE, verbose=0,
                                     callbacks=[model_checkpoint, csv_logger,ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6), F1ScoreCallback(),test_cb,early_stop],
                                     validation_data=val_train_images)

    # 2. aşama: baz modelin son katmanları açılmış, kalan epochlar
    base_model = train_model.get_layer('base')
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    train_model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss='binary_crossentropy', metrics=metrics)
   
    # Modeli eğitme
    train_history = train_model.fit(train_images, epochs=epochs+2,initial_epoch=2, batch_size=BATCH_SIZE, verbose=1,
                                     callbacks=[model_checkpoint, csv_logger,ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6), F1ScoreCallback(),test_cb,early_stop],
                                     validation_data=val_train_images)


    return train_model, [train_history]

#restnet
def train_deep_learning_model_restv2(train_images, val_train_images, model_checkpoint, csv_logger,test_dataset,test_labels, epochs=10, trainb=True):
    #metrics
    metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')]
    image_size=224
    # Focal loss objesi
    focal_loss = BinaryFocalCrossentropy(
    gamma=2.0,   # odaklanma parametresi
    alpha=0.25   # pozitif sınıfa ağırlık
    )
    # Test callback
    test_cb = TestEvaluationCallback(test_dataset, test_labels)
     # EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',     # Gelişmenin izleneceği metrik
        patience=10,             # 5 epoch boyunca gelişme olmazsa durur
        restore_best_weights=True  # En iyi model ağırlıklarını geri yükler
    )
    # İstemci modeli
    train_model = make_nn_model_resnet152(image_size, trainb)
    train_model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss=focal_loss, metrics=metrics)
    train_history1 = train_model.fit(train_images, epochs=2, batch_size=BATCH_SIZE, verbose=0,
                                     callbacks=[model_checkpoint, csv_logger,ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6), F1ScoreCallback(),test_cb,early_stop],
                                     validation_data=val_train_images)

    # 2. aşama: baz modelin son katmanları açılmış, kalan epochlar
    base_model = train_model.get_layer('base')
    for layer in base_model.layers[:-30]:
         layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    train_model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss=focal_loss, metrics=metrics)
     
    # Modeli eğitme
    train_history = train_model.fit(train_images, epochs=epochs+2, initial_epoch=2, batch_size=BATCH_SIZE, verbose=1,
                                     callbacks=[model_checkpoint, csv_logger,ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6), F1ScoreCallback(),test_cb,early_stop],
                                     validation_data=val_train_images)


    return train_model, [train_history]


