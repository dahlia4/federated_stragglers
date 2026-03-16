import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Input, Flatten,\
Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# Check GPU availibility-
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    # Get number of available GPUs-
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print(f"number of GPUs available = {num_gpus}")
    
    print(f"GPU: {gpu_devices}")
    details = tf.config.experimental.get_device_details(gpu_devices[0])
    print(f"GPU details: {details.get('device_name', 'Unknown GPU')}")
else:
    print("No GPU found")



# input image dimensions
img_rows, img_cols = 32, 32

# Load CIFAR-10 dataset-
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# classes we want
classes = [3, 8]  # cat, dog

# filter train
train_mask = np.isin(y_train, classes).flatten()
X_train, y_train = X_train[train_mask], y_train[train_mask]

# filter test
test_mask = np.isin(y_test, classes).flatten()
X_test, y_test = X_test[test_mask], y_test[test_mask]

if tf.keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# Specify hyper-parameters-
batch_size = 64
num_classes = 10
num_epochs = 100


# Convert datasets to floating point types-
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize the training and testing datasets-
X_train /= 255.0
X_test /= 255.0

print("\nDimensions of training and testing sets are:")
print(f"X_train.shape: {X_train.shape} & X_test.shape: {X_test.shape}")

# Create TF datasets-
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(50000).batch(batch_size = batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(X_test).shuffle(10000).batch(batch_size = batch_size)

# Needed by 'tf.GradientTape()' API for custom training and testing code.

class ConvBlock(Model):
    def __init__(
        self, num_filters,
        kernel_size, stride_length,
        pooling_size, pooling_stride,
        padding_type = 'same'
    ):
        super(ConvBlock, self).__init__()
        
        self.conv1 = Conv2D(
            filters = num_filters, kernel_size = kernel_size,
            strides = stride_length, padding = padding_type,
            activation = None, use_bias = False,
        )
        self.bn = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.conv2 = Conv2D(
            filters = num_filters, kernel_size = kernel_size,
            strides = stride_length, padding = padding_type,
            activation = None, use_bias = False
        )
        self.bn2 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.pool = MaxPooling2D(
            pool_size = pooling_size,
            strides = pooling_stride
        )
        
    
    def call(self, x):
        x = tf.keras.activations.relu(self.bn(self.conv1(x)))
        x = tf.keras.activations.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        return x
    
class Conv6_Encoder(Model):
    def __init__(self, latent_dim = 10):
        super(Conv6_Encoder, self).__init__()

        self.latent_dim = latent_dim
        
        self.conv_block1 = ConvBlock(
            num_filters = 64, kernel_size = 3,
            stride_length = 1, pooling_size = 2,
            pooling_stride = 2, padding_type = 'valid'
            )

        self.conv_block2 = ConvBlock(
            num_filters = 128, kernel_size = 3,
            stride_length = 1, pooling_size = 2,
            pooling_stride = 2, padding_type = 'valid'
            )
        
        self.conv_block3 = ConvBlock(
            num_filters = 256, kernel_size = 3,
            stride_length = 1, pooling_size = 2,
            pooling_stride = 2, padding_type = 'same'
            )

        self.flatten = Flatten()
        
        self.output_layer = Dense(
            units = self.latent_dim, activation = None
            )
        self.bn = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
    
    def call(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = tf.keras.activations.relu(self.bn(self.output_layer(x)))
        return x
    def shape_computation(self, x):
        print(f"Input shape: {x.shape}")
        x = self.conv_block1(x)
        print(f"conv_block1.shape: {x.shape}")
        x = self.conv_block2(x)
        print(f"conv_block2.shape: {x.shape}")
        x = self.conv_block3(x)
        print(f"conv_block3.shape: {x.shape}")
        x = self.flatten(x)
        print(f"flattened shape: {x.shape}")
        x = tf.keras.activations.relu(self.bn(self.output_layer(x)))
        print(f"Encoder output shape: {x.shape}")
        '''
        Input shape: (64, 32, 32, 3)
        conv_block1.shape: (64, 14, 14, 64)
        conv_block2.shape: (64, 5, 5, 128)
        conv_block3.shape: (64, 2, 2, 256)
        flattened shape: (64, 1024)
        Encoder output shape: (64, 100)
        '''
        
        return None

class Conv6_Decoder(Model):
    def __init__(self, latent_dim = 10):
        super(Conv6_Decoder, self).__init__()

        self.latent_dim = latent_dim
        
        # self.inp_layer = InputLayer(input_shape = self.latent_dim)
        
        self.dense0 = Dense(
            units = self.latent_dim, activation = None
            )
        self.bn0 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.dense = Dense(
            units = 1024, activation = None
        )
        self.bn = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.dense2 = Dense(
            units = 4 * 4 * 256, activation = None
        )
        self.bn2 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.reshape = Reshape((4, 4, 256))
        
        self.conv_transpose_layer1 = Conv2DTranspose(
            filters = 256, kernel_size = 3,
            strides = 2, padding = 'same',
            activation = None
            )
        self.bn3 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        self.conv_transpose_layer2 = Conv2DTranspose(
            filters = 256, kernel_size = 3,
            strides = 1, padding = 'same',
            activation = None
            )
        
        self.bn4 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.conv_transpose_layer3 =  Conv2DTranspose(
            filters = 128, kernel_size = 3,
            strides = 2, padding = 'same',
            activation = None
            )
        self.bn5 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.conv_transpose_layer4 = Conv2DTranspose(
            filters = 128, kernel_size = 3,
            strides = 1, padding = 'same',
            activation = None
            )
        
        self.bn6 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        self.conv_transpose_layer5 = Conv2DTranspose(
            filters = 64, kernel_size = 3,
            strides = 2, padding = 'same',
            activation = None
            )
        self.bn7 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
       
        self.conv_transpose_layer6 = Conv2DTranspose(
            filters = 64, kernel_size = 3,
            strides = 1, padding = 'same',
            activation = None
            )
        
        self.bn8 = BatchNormalization(
            axis = -1, momentum = 0.99,
            epsilon = 0.001
            )
        
        self.final_conv_layer = Conv2DTranspose(
            filters = 3, kernel_size = 3,
            strides = 1, padding = 'same',
            activation = None
            )
    def call(self, X):
        # X = self.inp_layer(X)
        X = tf.keras.activations.relu(self.bn0(self.dense0(X)))
        X = tf.keras.activations.relu(self.bn(self.dense(X)))
        X = tf.keras.activations.relu(self.bn2(self.dense2(X)))
        X = self.reshape(X)
        X = tf.keras.activations.relu(self.bn3(self.conv_transpose_layer1(X)))
        X = tf.keras.activations.relu(self.bn4(self.conv_transpose_layer2(X)))
        X = tf.keras.activations.relu(self.bn5(self.conv_transpose_layer3(X)))
        X = tf.keras.activations.relu(self.bn6(self.conv_transpose_layer4(X)))
        X = tf.keras.activations.relu(self.bn7(self.conv_transpose_layer5(X)))
        X = tf.keras.activations.relu(self.bn8(self.conv_transpose_layer6(X)))
        X = tf.keras.activations.sigmoid(self.final_conv_layer(X))
        #X = self.final_conv_layer(X)

        return X

    def shape_computation(self, x):
        print(f"Input shape: {x.shape}")
        x = tf.nn.relu(self.bn0(self.dense0(x)))
        print(f"first dense layer shape: {x.shape}")
        x = tf.nn.relu(self.bn(self.dense(x)))
        print(f"second dense layer shape: {x.shape}")
        x = tf.nn.relu(self.bn2(self.dense2(x)))
        print(f"third dense layer shape: {x.shape}")
        x = self.reshape(x)
        print(f"reshape: {x.shape}")
        x = tf.nn.relu(self.bn3(self.conv_transpose_layer1(x)))
        print(f"conv transpose layer1 shape: {x.shape}")
        x = tf.nn.relu(self.bn4(self.conv_transpose_layer2(x)))
        print(f"conv transpose layer2 shape: {x.shape}")
        x = tf.nn.relu(self.bn5(self.conv_transpose_layer3(x)))
        print(f"conv transpose layer3 shape: {x.shape}")
        x = tf.nn.relu(self.bn6(self.conv_transpose_layer4(x)))
        print(f"conv transpose layer4 shape: {x.shape}")
        x = tf.nn.relu(self.bn7(self.conv_transpose_layer5(x)))
        print(f"conv transpose layer5 shape: {x.shape}")
        x = tf.nn.relu(self.bn8(self.conv_transpose_layer6(x)))
        print(f"conv transpose layer6 shape: {x.shape}")
        x = self.final_conv_layer(x)
        print(f"Decoder output shape: {x.shape}")
        
        '''
        Input shape: (64, 100)
        first dense layer shape: (64, 100)
        second dense layer shape: (64, 1024)
        third dense layer shape: (64, 4096)
        reshape: (64, 4, 4, 256)
        conv transpose layer1 shape: (64, 8, 8, 256)
        conv transpose layer2 shape: (64, 8, 8, 256)
        conv transpose layer3 shape: (64, 16, 16, 128)
        conv transpose layer4 shape: (64, 16, 16, 128)
        conv transpose layer5 shape: (64, 32, 32, 64)
        conv transpose layer6 shape: (64, 32, 32, 64)
        Decoder output shape: (64, 32, 32, 3)
        '''
        
        return None



encoder = Conv6_Encoder(latent_dim = 100)

decoder = Conv6_Decoder(latent_dim = 100)


x = next(iter(train_dataset))

print(x.shape)


x_enc = encoder(x)

x_recon = decoder(x_enc)

print(x_enc.shape)

print(x.shape, x_recon.shape)

print(encoder.shape_computation(x))

print(decoder.shape_computation(x_enc))


del encoder, decoder, x, x_enc, x_recon




class Sampling(tf.keras.layers.Layer):
    """
    Create a sampling layer.
    Uses (mu, log_var) to sample latent vector 'z'.
    """
    def call(self, mu, log_var):
    # def call(self, inputs):
        # z_mean, z_log_var = inputs

        # Get batch size-
        batch = tf.shape(mu)[0]

        # Get latent space dimensionality-
        dim = tf.shape(mu)[1]

        # Add stochasticity by sampling from a multivariate standard 
        # Gaussian distribution-
        epsilon = tf.keras.backend.random_normal(
            shape = (batch, dim), mean = 0.0,
            stddev = 1.0
        )

        return mu + (tf.exp(0.5 * log_var) * epsilon)

class VAE(Model):
    def __init__(self, latent_space = 100):
        super(VAE, self).__init__()
        
        self.latent_space = latent_space
        
        self.encoder = Conv6_Encoder(latent_dim = self.latent_space)
        self.decoder = Conv6_Decoder(latent_dim = self.latent_space)
        
        # Define fully-connected layers for computing mean & log variance-
        self.mu = Dense(units = self.latent_space, activation = None)
        self.log_var = Dense(units = self.latent_space, activation = None)


    def reparameterize(self, mu, logvar):
        # Sample from a multivariate Gaussian distribution.
        # Adds stochasticity or variation-
        eps = tf.random.normal(
            shape = mu.shape, mean = 0.0,
            stddev = 1.0
        )
        return (eps * tf.exp(logvar * 0.5) + mu)
        
    
    def call(self, x):
        x = self.encoder(x)
        # print(f"x.shape: {x.shape}")
        # x.shape: (batch_size, 100)
        
        mu = self.mu(x)
        log_var = self.log_var(x)
        # z = self.reparameterize(mu, log_var)
        # z = Sampling()([mu, log_var])
        z = Sampling()(mu, log_var)
        '''
        print(f"mu.shape: {mu.shape}, log_var.shape: {log_var.shape}"
              f" & z.shape: {z.shape}")
        # mu.shape: (batch_size, 100), log_var.shape: (batch_size, 100) & z.shape: (batch_size, 100)
        '''
        x_recon = self.decoder(z)
        #x = tf.keras.activations.sigmoid(self.decoder(z))
        return x_recon, mu, log_var
    def model(self):
        '''
        Overrides 'model()' call.
        Output shape is not well-defined when using sub-classing. As a
        workaround, this method is implemeted.
        '''
        x = Input(shape = (32, 32, 3))
        return Model(inputs = [x], outputs = self.call(x))


if __name__ == "__main__":
    # Initialize VAE model
    model = VAE(latent_space = 128)
    
    
    # Sanity check-
    x = next(iter(train_dataset))
    # Forward pass using input data-
    x_recon, mu, log_var = model(x)
    
    print(x.shape, x_recon.shape)
    print(mu.shape, log_var.shape)
    print(mu.numpy().mean(), mu.numpy().std())
    print(log_var.numpy().mean(), log_var.numpy().std())
    
    del x, x_recon, mu, log_var
    
    
    # Get model summary-
    model.model().summary()
    
    # Count layer-wise number of trainable parameters-
    tot_params = 0
    
    for layer in model.trainable_weights:
        loc_params = tf.math.count_nonzero(layer, axis = None).numpy()
        tot_params += loc_params
        print(f"layer: {layer.shape} has {loc_params} parameters")
        
    print(f"VAE has {tot_params} trainable parameters")
    
    
    # Define an optimizer-
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)

    def compute_loss(data, reconstruction, mu, log_var, kl_weight=0.01):#alpha = 1):
    
        # Reconstruction loss-
        # recon_loss = tf.keras.losses.mean_squared_error(K.flatten(data), K.flatten(reconstruction))

        recon_loss = tf.reduce_mean(
            #tf.reduce_loss(
            tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction),
                #tf.keras.losses.mean_squared_error(data, reconstruction),
                axis = (1, 2)
            #)
            )
        )
    
        # KL-divergence loss-    
        kl_loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                kl_loss,
                axis = 1
            )
        )
        print(recon_loss.shape)
        print(kl_loss.shape)
        #total_loss = (recon_loss * alpha) + kl_loss
        total_loss = recon_loss + (kl_weight * kl_loss)
        return total_loss, recon_loss, kl_loss


    @tf.function
    def train_one_step(model, optimizer, data, kl_weight):
        # Function to perform one step/iteration of training
        
        with tf.GradientTape() as tape:
            # Make predictions using defined model-
            data_recon, mu, log_var = model(data)
            
            # Compute loss-
            total_loss, recon_loss, kl_loss = compute_loss(
                data = data, reconstruction = data_recon,
                mu = mu, log_var = log_var,
                #alpha = alpha
                kl_weight = kl_weight
            )
            
        # Compute gradients wrt defined loss and weights and biases-
        grads = tape.gradient(total_loss, model.trainable_variables)
        
        # type(grads)
        # list
        
        # Apply computed gradients to model's weights and biases-
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return total_loss, recon_loss, kl_loss
    
    @tf.function
    def test_step(model, optimizer, data, kl_weight):
        '''
        Function to test model performance
        on testing dataset
        '''
        # Make predictions using defined model-
        data_recon, mu, log_var = model(data)
        
        # Compute loss-
        total_loss, recon_loss, kl_loss = compute_loss(
            data = data, reconstruction = data_recon,
            mu = mu, log_var = log_var,
            #alpha = alpha
            kl_weight = kl_weight
        )
        
        return total_loss, recon_loss, kl_loss

    def get_kl_weight(epoch, max_weight=0.001, anneal_epochs=50):
        return min(max_weight, (epoch / anneal_epochs) * max_weight)

    print(f"Train VAE model for {num_epochs} epochs")
    
    # Python3 dict to contain training metrics-
    training_metrics = {}
    val_metrics = {}
    # Specify hyper-parameter for reconstruction loss vs. kl-divergence-
    alpha = 10
    for epoch in range(1, num_epochs + 1):
        """
        # Manual early stopping implementation-
        if loc_patience >= patience:
        print("\n'EarlyStopping' called!\n")
        break
        """

        # Epoch train & validation losses-
        train_loss = 0.0
        train_r_loss = 0.0
        train_kl_l = 0.0
        val_loss = 0.0
        val_r_loss = 0.0
        val_kl_l = 0.0
        kl_weight = get_kl_weight(epoch)
        num_train_batches = len(train_dataset)
        num_val_batches = len(test_dataset)
        for data in train_dataset:
            train_total_loss, train_recon_loss, train_kl_loss = train_one_step(
                model = model, optimizer = optimizer,
                data = data, kl_weight = kl_weight
            )
            
            train_loss += train_total_loss.numpy()
            train_r_loss += train_recon_loss.numpy()
            train_kl_l += train_kl_loss.numpy()

        train_loss /= num_train_batches
        train_r_loss /= num_train_batches
        train_kl_l /= num_train_batches

        for test_data in test_dataset:
            test_total_loss, test_recon_loss, test_kl_loss = test_step(
                model = model, optimizer = optimizer,
                data = test_data, kl_weight = kl_weight)
            
            val_loss += test_total_loss.numpy()
            val_r_loss += test_recon_loss.numpy()
            val_kl_l += test_kl_loss.numpy()
        val_loss /= num_val_batches
        val_r_loss /= num_val_batches
        val_kl_l /= num_val_batches
        # vae_train_loss.append(train_loss)
        # vae_val_loss.append(val_loss)
        
        training_metrics[epoch] = {
            'total_loss': train_loss, 'recon_loss': train_r_loss,
            'kl_loss': train_kl_l
        }
    
        val_metrics[epoch] = {
            'total_loss': val_loss, 'recon_loss': val_r_loss,
            'kl_loss': val_kl_l
        }

        print(f"epoch = {epoch}; total train loss = {train_loss:.4f},"
              f" train recon loss = {train_r_loss:.4f}, train kl loss = {train_kl_l:.4f};"
              f" total val loss = {val_loss:.4f}, val recon loss = {val_r_loss:.4f} &"
              f" val kl loss = {val_kl_l:.4f}"
              )
        # Sanity check for Python3 dicts containing training metrics-
        print(training_metrics.keys())
        
        print(val_metrics.keys())
        
        
        #print(training_metrics[5].keys())
        
        #print(val_metrics[5].keys())
        
        
        # Save trained model at the end of training-
        model.save_weights("VAE_CIFAR10_last_epoch_32.h5", overwrite = True)
        
        import pickle
        
        # Save training metrics as pickle file-
        with open("VAE_CIFAR10_training_metrics.pkl", "wb") as file:
            pickle.dump(training_metrics, file)
            
        # Save validation metrics as pickle file-
        with open("VAE_CIFAR10_training_metrics.pkl", "wb") as file:
            pickle.dump(val_metrics, file)
            
        # Save validation metrics as pickle file-
        with open("VAE_CIFAR10_training_metrics.pkl", "wb") as file:
            pickle.dump(val_metrics, file)
            
        #print(X_train[:1000, :].shape, X_train_reconstruced.shape)
        
        #print(X_train_reconstruced.numpy().shape)
        
        # Visualize one of reconstructed CIFAR-10 image-
        #img_idx = 200
        
        #plt.imshow(X_train_reconstruced[img_idx])
        #plt.title(f"CIFAR-10 Reconstructed Image: {img_idx + 1}")
        #plt.show()
        
        # Visualize 20 reconstruced images-
        #plt.figure(figsize = (9, 7))
        #for i in range(20):
            # 4 rows & 5 columns-
            #    plt.subplot(4, 5, i + 1)
            #    plt.imshow(X_train_reconstruced[i + 850])
            
        #plt.suptitle("CIFAR-10 Reconstructed images")
        #plt.savefig('VAE_CIFAR10_Reconstructed_Images4.png')
        plt.show()
