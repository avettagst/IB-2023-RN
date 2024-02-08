import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# for reproducibility 
#-------------------------------------------------------------------
seed=2                           
np.random.seed(seed)
tf.random.set_seed(seed)


# accuracy compatible with tensorflow v1
#-------------------------------------------------------------------
from tensorflow.python.keras import backend as K
def v1_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


#Todos los patrones posibles
#-------------------------------------------------------------------
n_tot = 32
x_tot = np.zeros((n_tot,5), dtype=np.float32)
y_tot = np.zeros((n_tot,1), dtype=np.float32)

for i in range(n_tot):
    if(i%2 == 0):
        x_tot[i,0] = 1
    else:
        x_tot[i,0] = -1

    if(int(i/2)%2 == 0):
        x_tot[i,1] = 1
    else:
        x_tot[i,1] = -1

    if(int(i/4)%2 == 0):
        x_tot[i,2] = 1
    else:
        x_tot[i,2] = -1

    if(int(i/8)%2 == 0):
        x_tot[i,3] = 1
    else:
        x_tot[i,3] = -1

    if(int(i/16)%2 == 0):
        x_tot[i,4] = 1
    else:
        x_tot[i,4] = -1

    y = 1
    for k in range(5):
        y = y*x_tot[i, k]
    y_tot[i] = y

#Los junto para poder shufflearlos y elegir patrones al azar
patrones = np.column_stack((x_tot, y_tot))

#Defino cuántos patrones uso para entrenamiento y cuántos para validación
n_train = 28
n_test = n_tot - n_train

# Network architecture
#-------------------------------------------------------------------
N_in = 5
N_hids = [1, 3, 5, 7, 9, 11] #distintas dimensiones a probar

all_accuracies = []
all_losses = []
all_val_losses = []

#Mezclamos
np.random.shuffle(patrones)

# Patrones de entrenamiento
x_train = np.zeros((n_train,5), dtype=np.float32)
y_train = np.zeros((n_train,1), dtype=np.float32)
x_train = np.copy(patrones[0:n_train, :5])
y_train = np.copy(patrones[0:n_train, 5:])

#Patrones de validación
x_test = np.zeros((n_test,5), dtype=np.float32)
y_test = np.zeros((n_test,1), dtype=np.float32)
x_test = np.copy(patrones[n_train:, :5])
y_test = np.copy(patrones[n_train:, 5:])

#Epocas de entrenamiento
epocas = 500

for i in range(len(N_hids)):
    N_hid = N_hids[i]

    model=tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(N_in,)),
            tf.keras.layers.Dense(N_hid, activation="tanh"),
            tf.keras.layers.Dense(1, activation="tanh"),
        ]
    )
 
    opti=tf.keras.optimizers.SGD(learning_rate=0.5)

    model.compile(optimizer=opti,
                loss='MSE', metrics=[v1_accuracy])

    history=model.fit(x=x_train, y=y_train,
                    epochs=epocas,
                    batch_size=n_train,
                    shuffle=False,
                    validation_data=(x_test, y_test), verbose=True)
    
    all_accuracies.append(history.history['v1_accuracy'])
    all_losses.append(np.sqrt(history.history['loss']))
    all_val_losses.append(np.sqrt(history.history['val_loss']))

    
    W_Input_Hidden = model.layers[0].get_weights()[0]
    W_Output_Hidden = model.layers[1].get_weights()[0]
    B_Input_Hidden = model.layers[0].get_weights()[1]
    B_Output_Hidden = model.layers[1].get_weights()[1]

    print('INPUT-HIDDEN LAYER WEIGHTS:')
    print(W_Input_Hidden)
    print('HIDDEN-OUTPUT LAYER WEIGHTS:')
    print(W_Output_Hidden)

    print('INPUT-HIDDEN LAYER BIAS:')
    print(B_Input_Hidden)
    print('HIDDEN-OUTPUT LAYER BIAS:')
    print(B_Output_Hidden)

#Accuracie
for i in range(len(N_hids)):
    plt.plot(range(1, epocas+1), all_accuracies[i], label = "N' = " + str(N_hids[i]))

plt.xlabel("Epoca")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Training loss
for i in range(len(N_hids)):
    plt.plot(range(1, epocas+1), all_losses[i], label = "N' = " + str(N_hids[i]))

plt.xlabel("Epoca")
plt.ylabel("Training loss")
plt.legend()
plt.show()

#Validation loss
for i in range(len(N_hids)):
    plt.plot(range(1, epocas+1), all_val_losses[i], label = "N' = " + str(N_hids[i]))

plt.xlabel("Epoca")
plt.ylabel("Validation loss")
plt.legend()
plt.show()