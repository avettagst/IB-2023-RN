import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# for reproducibility 
seed=11                         
np.random.seed(seed)
tf.random.set_seed(seed)

#Funciones de activación de las neuronas de la red
def hidden_activation(x):
    return  1/(1+tf.math.exp(-x))

def output_activation(x):
    return  x


#Patrones
#--------------------------------------------------------------
n_tot = 501
x_ini = 0.2

xd = np.zeros(n_tot)
xd[0] = x_ini
for i in range(1, n_tot):
     xd[i] = 4*(xd[i-1]-(xd[i-1]**2))


#Mapeo logístico
plt.plot(xd[:-1], xd[1:], 'o', c = 'm', markersize = '1', label = "Función")
plt.xlabel("x(t)")
plt.ylabel("x(t+1)")
plt.title("x(0) = 0.2")
plt.show()

#Cantidad de patrones a utilizar para el entrenamiento
n_trains = [5, 10, 100]

#Patrones de testeo
n_test = n_tot - 1
x_test = np.zeros(n_test)
y_test = np.zeros(n_test)
x_test = np.copy(xd[:-1])
y_test = np.copy(xd[1:])



# Arquitectura de la red
#--------------------------------------------------------------
hidden_dim = 5 # Number of hidden units

inputs = tf.keras.layers.Input(shape=(1,))
x = tf.keras.layers.Dense(hidden_dim, activation=hidden_activation)(inputs)
merge=tf.keras.layers.concatenate([inputs,x],axis=-1)
predictions = tf.keras.layers.Dense(1, activation=output_activation)(merge)

#Epocas de entrenamiento
epocas = 1000

#Vectores donde almaceno los distintos resultados y métricas de cada caso
preds = np.zeros(shape=(len(n_trains),n_test))
train_loss = np.zeros(shape=(len(n_trains),epocas))
val_loss = np.zeros(shape=(len(n_trains),epocas))


for i in range(len(n_trains)):

    #Patrones de entrenamiento para cada caso
    n_train = n_trains[i]
    x_train = np.zeros((n_train,1), dtype=np.float32)
    y_train = np.zeros((n_train,1), dtype=np.float32)

    x_train = np.random.choice(xd, size=n_train, replace=False)

    for j in range(n_train):
        y_train[j] = 4*(x_train[j]-(x_train[j]**2))

    #Mapeo logístico y patrones de entrenamiento en cuestión
    plt.plot(x_train, y_train, 'o', color = 'r', markersize = '4', label = "Entrenamiento")
    plt.plot(xd[:-1], xd[1:], '*', color = 'm', markersize = '1', label = "Función")
    plt.xlabel("x(t)")
    plt.ylabel("x(t+1)")
    plt.title("x(0) = 0.2")
    plt.legend()
    plt.show()   
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    opti=tf.keras.optimizers.Adam(learning_rate=0.5)
    model.compile(optimizer=opti,
                loss='MSE')

    history=model.fit(x=x_train, y=y_train,
                    epochs=epocas,
                    batch_size=n_train,
                    shuffle=False,
                    validation_data=(x_test, y_test), verbose=True)

    encoded_log = model.predict(x_test, verbose=True)

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

    #Predicciones
    plt.plot(x_train, y_train, 'o', color = 'r', markersize = '4', label = "Train")
    plt.plot(x_test, y_test, '*', color = 'm', markersize = '1', label = "Función")
    plt.plot(x_test, encoded_log, '*', color = 'g', markersize = '1', label = "Prediccion")
    plt.legend()
    plt.show()

    #Loss
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.ylim(0)
    plt.ylabel('MSE')
    plt.xlabel('Epoca')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    #Almaceno para después comparar con otros casos
    preds[i, :] = np.copy(encoded_log.reshape(-1))
    train_loss[i, :] = np.copy(np.sqrt(history.history['loss']))
    val_loss[i, :] = np.copy(np.sqrt(history.history['val_loss']))

#Color para cada n_train distinto
colors = ['r', 'b', 'g']

#Predicciones
plt.plot(x_test, y_test, '*', color = 'm', markersize = '3', label = "Función deseada")
for i in range(len(n_trains)):
    plt.plot(x_test, preds[i], '*', color = colors[i], markersize = '1', label = "N = " + str(n_trains[i]))
plt.xlabel("x(t)")
plt.ylabel("x(t+1)")
plt.legend()
plt.show()

#Pérdidas de entrenamiento
for i in range(len(n_trains)):
    plt.plot(range(1, epocas+1), train_loss[i], color = colors[i], label = "N = " + str(n_trains[i]))
plt.xlabel("Epoca")
plt.ylabel("Training loss")
plt.legend()
plt.show()

#Pérdidas de validación
for i in range(len(n_trains)):
    plt.plot(range(1, epocas+1), val_loss[i], color = colors[i], label = "N = " + str(n_trains[i]))
plt.xlabel("Epoca")
plt.ylabel("Validation loss")
plt.legend()
plt.show()