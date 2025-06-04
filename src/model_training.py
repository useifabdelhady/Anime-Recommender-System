import joblib
import mlflow
import mlflow.keras
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from src.logger import get_logger
from src.custom_exception import CustomException
from src.base_model import BaseModel
from config.paths_config import *

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, data_path):
        self.data_path = data_path
        
        # Initialize MLflow
        mlflow.set_experiment("anime_recommender")
        logger.info("Model Training & MLflow initialized..")
    
    def load_data(self):
        try:
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)

            logger.info("Data loaded successfully for Model Training")
            return X_train_array, X_test_array, y_train, y_test
        except Exception as e:
            raise CustomException("Failed to load data", e)
        
    def train_model(self):
        try:
            # Start MLflow run
            with mlflow.start_run():
                X_train_array, X_test_array, y_train, y_test = self.load_data()

                n_users = len(joblib.load(USER2USER_ENCODED))
                n_anime = len(joblib.load(ANIME2ANIME_ENCODED))

                base_model = BaseModel(config_path=CONFIG_PATH)
                model = base_model.RecommenderNet(n_users=n_users, n_anime=n_anime)

                # Hyperparameters
                start_lr = 0.00001
                min_lr = 0.0001
                max_lr = 0.00005
                batch_size = 10000
                epochs = 20
                ramup_epochs = 5
                sustain_epochs = 0
                exp_decay = 0.8

                # Log hyperparameters
                mlflow.log_params({
                    "start_lr": start_lr,
                    "min_lr": min_lr,
                    "max_lr": max_lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "ramup_epochs": ramup_epochs,
                    "sustain_epochs": sustain_epochs,
                    "exp_decay": exp_decay,
                    "n_users": n_users,
                    "n_anime": n_anime
                })

                def lrfn(epoch):
                    if epoch < ramup_epochs:
                        return (max_lr - start_lr) / ramup_epochs * epoch + start_lr
                    elif epoch < ramup_epochs + sustain_epochs:
                        return max_lr
                    else:
                        return (max_lr - min_lr) * exp_decay ** (epoch - ramup_epochs - sustain_epochs) + min_lr
                
                lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)
                model_checkpoint = ModelCheckpoint(
                    filepath=CHECKPOINT_FILE_PATH,
                    save_weights_only=True,
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True
                )
                early_stopping = EarlyStopping(
                    patience=3,
                    monitor="val_loss",
                    mode="min",
                    restore_best_weights=True
                )

                my_callbacks = [model_checkpoint, lr_callback, early_stopping]

                # Create directories
                os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH), exist_ok=True)
                os.makedirs(MODEL_DIR, exist_ok=True)
                os.makedirs(WEIGHTS_DIR, exist_ok=True)

                try:
                    history = model.fit(
                        x=X_train_array,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_test_array, y_test),
                        callbacks=my_callbacks
                    )
                    
                    model.load_weights(CHECKPOINT_FILE_PATH)
                    logger.info("Model training completed.....")

                    # Log metrics for each epoch
                    for epoch in range(len(history.history['loss'])):
                        train_loss = history.history["loss"][epoch]
                        val_loss = history.history["val_loss"][epoch]
                        
                        mlflow.log_metrics({
                            'train_loss': train_loss,
                            'val_loss': val_loss
                        }, step=epoch)
                    
                    # Log final metrics
                    final_train_loss = history.history["loss"][-1]
                    final_val_loss = history.history["val_loss"][-1]
                    best_val_loss = min(history.history["val_loss"])
                    
                    mlflow.log_metrics({
                        'final_train_loss': final_train_loss,
                        'final_val_loss': final_val_loss,
                        'best_val_loss': best_val_loss
                    })
                
                except Exception as e:
                    raise CustomException("Model training failed.....")
                
                self.save_model_weights(model)

        except Exception as e:
            logger.error(str(e))
            raise CustomException("Error during Model Training Process", e)
        
    def extract_weights(self, layer_name, model):
        try:
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0]
            weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
            logger.info(f"Extracting weights for {layer_name}")
            return weights
        except Exception as e:
            logger.error(str(e))
            raise CustomException("Error during Weight Extraction Process", e)
    
    def save_model_weights(self, model):
        try:
            # Save the model
            model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")

            # Extract and save weights
            user_weights = self.extract_weights('user_embedding', model)
            anime_weights = self.extract_weights('anime_embedding', model)

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            # Log model and artifacts to MLflow
            mlflow.keras.log_model(model, "model")
            mlflow.log_artifact(MODEL_PATH, "saved_model")
            mlflow.log_artifact(ANIME_WEIGHTS_PATH, "weights")
            mlflow.log_artifact(USER_WEIGHTS_PATH, "weights")
            
            # Log model info
            mlflow.log_params({
                "model_path": MODEL_PATH,
                "user_weights_shape": user_weights.shape,
                "anime_weights_shape": anime_weights.shape
            })

            logger.info("User and Anime weights saved successfully....")
            
        except Exception as e:
            logger.error(str(e))
            raise CustomException("Error during saving model and weights Process", e)


if __name__ == "__main__":
    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()