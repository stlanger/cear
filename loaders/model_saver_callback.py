import keras
import os

class ModelSaverCallback(keras.callbacks.Callback):
    
    def __init__(self, model, folder, filename, save_each_epoch=False):
        self.folder = folder
        self.filename = filename
        self.model = model
        self.save_each_epoch = save_each_epoch
        
    def on_epoch_end(self, epoch, logs=None):
        if self.save_each_epoch:
            filepath = os.path.join(self.folder, f"epoch_{str(epoch.rjust(3, '0'))}_{self.filename}")
            self.model.save(filepath)
        
    def on_train_end(self, logs=None):
        output_path = os.path.join(self.folder, self.filename)
        print("saving model to:", output_path)
        self.model.save_pretrained(output_path)

# class model_per_epoch(keras.callbacks.Callback):
#     def __init__(self, model,filepath,save_best_only):
#         self.filepath=filepath
#         self.model=model
#         self.save_best_only=save_best_only
#         self.lowest_loss=np.inf
#         self.best_weights=self.model.get_weights()
#     def on_epoch_end(self,epoch, logs=None):
#         v_loss=logs.get('val_loss')
#         if v_loss< self.lowest_loss:
#             self.lowest_loss =v_loss
#             self.best_weights=self.model.get_weights()
#             self.best_epoch=epoch +1
#         if self.save_best_only==False:
#             name= str(epoch) +'-' + str(v_loss)[:str(v_loss).rfind('.')+3] + '.h5'
#             file_id=os.path.join(self.filepath, name)
#             self.model.save(file_id)
#     def on_train_end(self, logs=None):
#         if self.save_best_only == True:
#             self.model.set_weights(self.best_weights)
#             name= str(self.best_epoch) +'-' + str(self.lowest_loss)[:str(self.lowest_loss).rfind('.')+3] + '.h5'
#             file_id=os.path.join(self.filepath, name)
#             self.model.save(file_id)
#             print(' model is returned with best weiights from epoch ', self.best_epoch)
            
# save_dir=r'c:\Temp\models'
# save_best_only= True
# callbacks=[model_per_epoch(model, save_dir, save_best_only)]  