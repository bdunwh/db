import sys
import numpy as np
sys.path.append('/content/drive/MyDrive')
from modelbuilder import make_nn_model_mobileNetV3,make_nn_model_mobileNet,make_nn_model_efficientb0,make_nn_model_efficientb7,make_nn_model_dense,make_nn_model_resnet152
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback,ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from config import IMG_SIZE,BATCH_SIZE
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
#**Federated Learning Modeli Eğitme**
def train_fedcycle_model(model_handle, client_images, val_images, test_images, test_labels, model_checkpoint, csv_logger,model_name, total_samples,global_model=None, epochs=10, trainb=True):
    metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')]

    # Test callback
    test_cbs = []
    for i in range (5):  
     test_cb = TestEvaluationCallback(test_images[i], test_labels[i])
     test_cbs.append(test_cb)

    # EarlyStopping callback
    early_stops = []
    for _ in range(5):
      early_stop = EarlyStopping(
          monitor='val_accuracy',     # Gelişmenin izleneceği metrik
          patience=10,             # 10 epoch boyunca gelişme olmazsa durur
          restore_best_weights=True  # En iyi model ağırlıklarını geri yükler
      )
      early_stops.append(early_stop)
    
    learn = 1e-3
     # İstemci modeli
    client_models = []
    for _ in range(5):
      if model_name == 'EFFB0' :
        client_model = make_nn_model_efficientb0(IMG_SIZE, trainb)
      elif model_name == 'EFFB7':
        client_model = make_nn_model_efficientb7(IMG_SIZE, trainb)
      elif model_name == 'DENSE':
        client_model = make_nn_model_dense(IMG_SIZE, trainb)
      elif model_name == 'REST':
        client_model = make_nn_model_resnet152(IMG_SIZE, trainb)
      elif model_name == 'MOBILE':
        learn=1e-4
        client_model = make_nn_model_mobileNet(IMG_SIZE, trainb)
      elif model_name == 'MOBV3':
        learn=1e-4
        client_model = make_nn_model_mobileNetV3(IMG_SIZE, trainb)

      if model_name=='DENSE' or model_name=='REST':
        # 2. aşama: baz modelin son katmanları açılmış, kalan epochlar
        base_model = client_model.get_layer('base')
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True
      
      client_model.compile(optimizer=AdamW(learning_rate=learn, weight_decay=1e-4), loss='binary_crossentropy', metrics=metrics)
      client_models.append(client_model)
        
    
    # Modeli eğitme
    train_histories = [[] for _ in range(5)]
    client_name=["Breakhis","Busi","Roboflow","Rsna","Bcfpp"]
    cnt_completed=0
    stopped = [False, False, False, False, False]
    n_clients = [sum(1 for _ in c.unbatch()) for c in client_images]
    for ind in range(10):
      total_samples = 0 #her epochda yeniden total hesaplanır
      client_weights = [] 
      print(f"Epoch {ind+1}/10 - FedCycle Training")
      for inCli in range(5):
        total_samples += n_clients[inCli]
        print(f"\n>>> {client_name[inCli]} Train Başladı <<<\n")
        if global_model is not None:
          client_models[inCli].set_weights(global_model.get_weights())
        # İstemcileri eğit
        history=client_models[inCli].fit(client_images[inCli], epochs=1, batch_size=BATCH_SIZE, verbose=1,
                                    callbacks=[model_checkpoint, csv_logger,ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6), F1ScoreCallback(),test_cbs[inCli],early_stops[inCli]],
                                    validation_data=val_images[inCli])
        
        train_histories[inCli].append(history)
        # Ağırlıkları topla (henüz global model güncellenmeyecek)
        client_weights.append(client_models[inCli].get_weights())
      
      if global_model is None:
       global_model = client_model[0]
       global_model = tf.keras.models.clone_model(client_models[0])
       global_model.set_weights(client_weights[0])
      else:
       global_weights = global_model.get_weights()

      n_global = total_samples

      new_weights = global_weights
      for n_client, c_weights in zip(n_clients, client_weights):
        new_weights = [
          (n_client * cw + n_global * gw) / (n_client + n_global)
          for cw, gw in zip(c_weights, new_weights)]

      global_model.set_weights(new_weights)
      global_model.save_weights(f"{model_name}.weights.h5")

         
      if cnt_completed==5:
        break
      else:
        continue
          
    
    return global_model,total_samples,train_histories


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





