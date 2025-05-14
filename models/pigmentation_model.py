import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, ResNet50V2

class PigmentationSegmentationModel:
    """
    Deep learning model for skin pigmentation segmentation.
    Uses a U-Net architecture with a pre-trained encoder
    """
    def __init__(self, 
                 input_shape=(224, 224, 3),
                 encoder_backbone='mobilenetv2',
                 num_classes=1,
                 learning_rate=0.0001,
                 batch_size=16):
        """
        Initialize the pigmentation segmentation model
        
        Parameters:
        ----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        encoder_backbone : str
            Backbone to use for the encoder ('mobilenetv2' or 'resnet50v2')
        num_classes : int
            Number of classes to segment (1 for binary segmentation)
        learning_rate : float
            Learning rate for training
        batch_size : int
            Batch size for training
        """
        self.input_shape = input_shape
        self.encoder_backbone = encoder_backbone.lower()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        
    def build_model(self):
        """
        Build and compile the U-Net model with a pre-trained encoder
        
        Returns:
        -------
        model : tensorflow.keras.Model
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Create encoder based on selected backbone
        if self.encoder_backbone == 'mobilenetv2':
            encoder = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            # Define encoder output layers for skip connections
            e1 = encoder.get_layer('input_1').output  # 224x224x3
            e2 = encoder.get_layer('block_1_expand_relu').output  # 112x112x96
            e3 = encoder.get_layer('block_3_expand_relu').output  # 56x56x144
            e4 = encoder.get_layer('block_6_expand_relu').output  # 28x28x192
            e5 = encoder.get_layer('block_13_expand_relu').output  # 14x14x576
            
        elif self.encoder_backbone == 'resnet50v2':
            encoder = ResNet50V2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            # Define encoder output layers for skip connections
            e1 = encoder.get_layer('input_1').output  # 224x224x3
            e2 = encoder.get_layer('conv1_conv').output  # 112x112x64
            e3 = encoder.get_layer('conv2_block3_1_relu').output  # 56x56x64
            e4 = encoder.get_layer('conv3_block4_1_relu').output  # 28x28x128
            e5 = encoder.get_layer('conv4_block6_1_relu').output  # 14x14x256
        
        else:
            raise ValueError(f"Backbone {self.encoder_backbone} not supported. Use 'mobilenetv2' or 'resnet50v2'.")
        
        # Freeze encoder weights
        encoder.trainable = False
        
        # Bottleneck
        b = Conv2D(512, (3, 3), activation='relu', padding='same')(e5)
        b = BatchNormalization()(b)
        b = Dropout(0.3)(b)
        b = Conv2D(512, (3, 3), activation='relu', padding='same')(b)
        b = BatchNormalization()(b)
        
        # Decoder path
        # Upsampling block 1
        d1 = UpSampling2D((2, 2))(b)
        d1 = Concatenate()([d1, e4])
        d1 = Conv2D(256, (3, 3), activation='relu', padding='same')(d1)
        d1 = BatchNormalization()(d1)
        d1 = Dropout(0.3)(d1)
        d1 = Conv2D(256, (3, 3), activation='relu', padding='same')(d1)
        d1 = BatchNormalization()(d1)
        
        # Upsampling block 2
        d2 = UpSampling2D((2, 2))(d1)
        d2 = Concatenate()([d2, e3])
        d2 = Conv2D(128, (3, 3), activation='relu', padding='same')(d2)
        d2 = BatchNormalization()(d2)
        d2 = Dropout(0.3)(d2)
        d2 = Conv2D(128, (3, 3), activation='relu', padding='same')(d2)
        d2 = BatchNormalization()(d2)
        
        # Upsampling block 3
        d3 = UpSampling2D((2, 2))(d2)
        d3 = Concatenate()([d3, e2])
        d3 = Conv2D(64, (3, 3), activation='relu', padding='same')(d3)
        d3 = BatchNormalization()(d3)
        d3 = Dropout(0.3)(d3)
        d3 = Conv2D(64, (3, 3), activation='relu', padding='same')(d3)
        d3 = BatchNormalization()(d3)
        
        # Upsampling block 4
        d4 = UpSampling2D((2, 2))(d3)
        d4 = Concatenate()([d4, e1])
        d4 = Conv2D(32, (3, 3), activation='relu', padding='same')(d4)
        d4 = BatchNormalization()(d4)
        d4 = Dropout(0.3)(d4)
        d4 = Conv2D(32, (3, 3), activation='relu', padding='same')(d4)
        d4 = BatchNormalization()(d4)
        
        # Output layer
        if self.num_classes == 1:
            outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)
        else:
            outputs = Conv2D(self.num_classes, (1, 1), activation='softmax')(d4)
        
        # Create model
        model = Model(inputs=[encoder.input], outputs=[outputs])
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        
        if self.num_classes == 1:
            # Binary segmentation
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
            )
        else:
            # Multi-class segmentation
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=self.num_classes)]
            )
        
        self.model = model
        return model
    
    def train(self,
              train_generator,
              validation_generator,
              epochs=100,
              steps_per_epoch=None,
              validation_steps=None,
              fine_tune_encoder=True,
              fine_tune_at=0.3,
              model_dir='models/checkpoints'):
        """
        Train the model.
        
        Parameters:
        ----------
        train_generator : tensorflow.keras.utils.Sequence
            Generator for training data
        validation_generator : tensorflow.keras.utils.Sequence
            Generator for validation data
        epochs : int
            Number of training epochs
        steps_per_epoch : int, optional
            Number of steps per epoch (default: None, uses len(train_generator))
        validation_steps : int, optional
            Number of validation steps (default: None, uses len(validation_generator))
        fine_tune_encoder : bool
            Whether to fine-tune the encoder backbone after initial training
        fine_tune_at : float
            Percentage of the encoder to fine-tune (0.3 means fine-tune the last 30% of the encoder)
        model_dir : str
            Directory to save model checkpoints
            
        Returns:
        -------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        if self.model is None:
            self.build_model()
            
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(model_dir, 'pigmentation_model_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Initial training with frozen encoder
        print("Initial training with frozen encoder...")
        history = self.model.fit(
            train_generator,
            epochs=epochs // 2 if fine_tune_encoder else epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        # Fine-tune encoder if specified
        if fine_tune_encoder and epochs > 1:
            print("Fine-tuning encoder...")
            
            # Get the encoder part of the model
            encoder = None
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.Model):
                    encoder = layer
                    break
                    
            if encoder is not None:
                # Calculate how many layers to unfreeze
                if 0 < fine_tune_at < 1:
                    # Percentage of layers
                    num_layers = len(encoder.layers)
                    fine_tune_from = int(num_layers * (1 - fine_tune_at))
                else:
                    # Specific layer index
                    fine_tune_from = int(fine_tune_at)
                
                # Unfreeze layers for fine-tuning
                encoder.trainable = True
                for layer in encoder.layers[:fine_tune_from]:
                    layer.trainable = False
                    
                # Recompile model with lower learning rate
                optimizer = Adam(learning_rate=self.learning_rate / 10)
                
                if self.num_classes == 1:
                    self.model.compile(
                        optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
                    )
                else:
                    self.model.compile(
                        optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=self.num_classes)]
                    )
                
                # Continue training with fine-tuned encoder
                fine_tune_history = self.model.fit(
                    train_generator,
                    epochs=epochs // 2,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=callbacks
                )
                
                # Combine histories
                for key in fine_tune_history.history:
                    history.history[key].extend(fine_tune_history.history[key])
        
        # Save final model
        self.model.save(os.path.join(model_dir, 'pigmentation_model_final.h5'))
        
        return history
    
    def predict(self, image):
        """
        Predict pigmentation segmentation on a preprocessed image
        
        Parameters:
        ----------
        image : numpy.ndarray
            Preprocessed image of shape (height, width, channels)
            
        Returns:
        -------
        segmentation : numpy.ndarray
            Segmentation mask
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() or load_model() first.")
            
        # Ensure image has the right shape
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Predict
        prediction = self.model.predict(image)
        
        # Remove batch dimension
        if prediction.shape[0] == 1:
            prediction = prediction[0]
            
        # Binarize if binary segmentation
        if self.num_classes == 1:
            prediction = (prediction > 0.5).astype(np.uint8)
            
        return prediction
    
    def load_model(self, model_path):
        """
        Load a pre-trained model
        
        Parameters:
        ----------
        model_path : str
            Path to the pre-trained model file
            
        Returns:
        -------
        model : tensorflow.keras.Model
            Loaded Keras model
        """
        self.model = tf.keras.models.load_model(model_path)
        return self.model


class PigmentationClassifier:
    """
    Deep learning model for classifying pigmentation severity
    """
    def __init__(self, 
                 input_shape=(224, 224, 3),
                 base_model='mobilenetv2',
                 num_classes=5,
                 learning_rate=0.0001):
        """
        Initialize the pigmentation classifier
        
        Parameters:
        ----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        base_model : str
            Base model to use ('mobilenetv2' or 'resnet50v2')
        num_classes : int
            Number of severity classes
        learning_rate : float
            Learning rate for training
        """
        self.input_shape = input_shape
        self.base_model = base_model.lower()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self):
        """
        Build and compile the classification model
        
        Returns:
        -------
        model : tensorflow.keras.Model
            Compiled Keras model
        """
        # Create base model
        if self.base_model == 'mobilenetv2':
            base = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.base_model == 'resnet50v2':
            base = ResNet50V2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Base model {self.base_model} not supported. Use 'mobilenetv2' or 'resnet50v2'.")
            
        # Freeze the base model
        base.trainable = False
        
        # Create classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        if self.num_classes == 1:  # Regression
            outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        else:  # Classification
            outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
            
        # Create model
        model = tf.keras.Model(inputs=base.input, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        
        if self.num_classes == 1:  # Regression
            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )
        else:  # Classification
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
        self.model = model
        return model
    
    def train(self,
              train_generator,
              validation_generator,
              epochs=50,
              steps_per_epoch=None,
              validation_steps=None,
              fine_tune=True,
              model_dir='models/checkpoints'):
        """
        Train the model
        
        Parameters:
        ----------
        train_generator : tensorflow.keras.utils.Sequence
            Generator for training data
        validation_generator : tensorflow.keras.utils.Sequence
            Generator for validation data
        epochs : int
            Number of training epochs
        steps_per_epoch : int, optional
            Number of steps per epoch (default: None, uses len(train_generator))
        validation_steps : int, optional
            Number of validation steps (default: None, uses len(validation_generator))
        fine_tune : bool
            Whether to fine-tune the base model after initial training
        model_dir : str
            Directory to save model checkpoints
            
        Returns:
        -------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        if self.model is None:
            self.build_model()
            
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(model_dir, 'pigmentation_classifier_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Initial training with frozen base model
        print("Initial training with frozen base model...")
        history = self.model.fit(
            train_generator,
            epochs=epochs // 2 if fine_tune else epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        # Fine-tune base model if specified
        if fine_tune and epochs > 1:
            print("Fine-tuning base model...")
            
            # Get the base model
            base_model = None
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.Model):
                    base_model = layer
                    break
                    
            if base_model is not None:
                # Unfreeze the top layers of the base model
                base_model.trainable = True
                
                # Freeze all the layers except the last 4 blocks
                if self.base_model == 'mobilenetv2':
                    for layer in base_model.layers[:-20]:  # MobileNetV2 has ~155 layers
                        layer.trainable = False
                elif self.base_model == 'resnet50v2':
                    for layer in base_model.layers[:-30]:  # ResNet50V2 has ~190 layers
                        layer.trainable = False
                
                # Recompile model with lower learning rate
                optimizer = Adam(learning_rate=self.learning_rate / 10)
                
                if self.num_classes == 1:  # Regression
                    self.model.compile(
                        optimizer=optimizer,
                        loss='mean_squared_error',
                        metrics=['mean_absolute_error']
                    )
                else:  # Classification
                    self.model.compile(
                        optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                
                # Continue training with fine-tuned base model
                fine_tune_history = self.model.fit(
                    train_generator,
                    epochs=epochs // 2,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=callbacks
                )
                
                # Combine histories
                for key in fine_tune_history.history:
                    history.history[key].extend(fine_tune_history.history[key])
        
        # Save final model
        self.model.save(os.path.join(model_dir, 'pigmentation_classifier_final.h5'))
        
        return history
    
    def predict(self, image):
        """
        Predict pigmentation severity class
        
        Parameters:
        ----------
        image : numpy.ndarray
            Preprocessed image of shape (height, width, channels)
            
        Returns:
        -------
        severity_class : int
            Predicted severity class (0 to num_classes-1)
        confidence : float
            Confidence score for the prediction
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() or load_model() first.")
            
        # Ensure image has the right shape
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Predict
        prediction = self.model.predict(image)
        
        if self.num_classes == 1:  # Regression
            # Return the predicted value and 1.0 confidence (arbitrary for regression)
            return prediction[0][0], 1.0
        else:  # Classification
            # Get the class with highest probability
            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]
            
            return class_idx, confidence
    
    def load_model(self, model_path):
        """
        Load a pre-trained model
        
        Parameters:
        ----------
        model_path : str
            Path to the pre-trained model file
            
        Returns:
        -------
        model : tensorflow.keras.Model
            Loaded Keras model
        """
        self.model = tf.keras.models.load_model(model_path)
        return self.model 