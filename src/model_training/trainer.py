import logging
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

from tensorflow import keras
import numpy as np
import pandas as pd

# from src.model_training 
import data, model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

def train(
    train_data_dir,
    eval_data_dir,
    hyperparams,
    base_model_dir=None,
):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Loading dataset from {train_data_dir}")

    train_dataset = data.get_dataset(
        train_data_dir,
        hyperparams,
    )
    
    eval_dataset = data.get_dataset(
        eval_data_dir,
        hyperparams,
    )
    
    def dice_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

    def dice_p_bce(in_gt, in_pred):
        return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

    def true_positive_rate(y_true, y_pred):
        return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
    
    
    weight_path="{}_weights.best.ctph".format('seg_model')

    checkpoint = ModelCheckpoint(
        weight_path, 
        monitor='val_dice_coef', 
        verbose=1, 
        save_best_only=True, 
        mode='max', 
        save_weights_only = True)

    optimizer = keras.optimizers.Adam(
        learning_rate=hyperparams["learning_rate"], 
        decay=hyperparams["decay_rate"])
    
    metrics = [dice_coef, 'binary_accuracy', true_positive_rate]
    
    reduceLROnPlat = ReduceLROnPlateau(
        monitor='val_dice_coef', 
        factor=0.5, 
        patience=10,                       
        verbose=1, 
        mode='max', 
        epsilon=0.0001, 
        cooldown=2, 
        min_lr=1e-6
    )
    
    early_stopping = EarlyStopping(
        monitor="val_dice_coef", 
        mode="max", 
        patience=30)
    
    callbacks_list = [checkpoint, reduceLROnPlat, early_stopping]

    seg_model = model.create_model(hyperparams)
    if base_model_dir:
        try:
            seg_model = keras.load_model(base_model_dir)
        except:
            pass

    seg_model.compile(
        optimizer = optimizer, 
        loss = dice_p_bce, 
        metrics = metrics
    )

    logging.info("Model training started...")
    loss_history = [
        seg_model.fit(
            train_dataset,
            epochs=hyperparams["num_epochs"], 
            validation_data=eval_dataset.take(1),
            callbacks=callbacks_list,
            verbose=1,
            workers=1 # the generator is not very thread safe
        )
    ]
    
    logging.info("Model training completed.")

    return seg_model

def evaluate(model, data_dir, raw_schema_location, hyperparams):
    pass
