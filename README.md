# Atari-Proyecto-Final
# Daian Alejandra Bermúdez Ceballos

Saqué de la pagina https://keras.io/examples/rl/deep_q_network_breakout/ el codigo proximo a explicarse:

## Setup: 
Este código importa las bibliotecas y módulos necesarios y configura los parámetros para un algoritmo de aprendizaje por refuerzo que juega al juego de Atari "Breakout". A continuación, se explica en detalle cada parte del código:

- Importación de bibliotecas y módulos:

```
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

Aquí, se importan las bibliotecas y módulos necesarios para el código. make_atari y wrap_deepmind son funciones proporcionadas por el paquete "baselines" que se utilizan para crear y envolver el entorno del juego Atari. numpy se importa como np para realizar operaciones numéricas, y tensorflow se importa como tf para construir y entrenar modelos de aprendizaje profundo.

- Parámetros de configuración:

```
seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 10000
```
Estos son los parámetros de configuración para el algoritmo de aprendizaje por refuerzo. seed es la semilla utilizada para la generación de números aleatorios, gamma es el factor de descuento que determina la importancia de las recompensas futuras, epsilon es el parámetro codicioso utilizado para la exploración y explotación del agente, epsilon_min y epsilon_max definen el rango en el que varía epsilon, y epsilon_interval calcula la diferencia entre epsilon_max y epsilon_min. batch_size define el tamaño de los lotes utilizados en el búfer de reproducción (replay buffer), y max_steps_per_episode establece el número máximo de pasos permitidos por episodio.

- Creación del entorno del juego:

```
env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)
```

Aquí, se crea el entorno del juego utilizando la función make_atari, que crea una instancia del juego "Breakout" sin cuadros omitidos y wrap_deepmind, que realiza una serie de transformaciones en los cuadros del juego para adaptarlos al modelo de aprendizaje profundo. Luego, se establece la semilla del entorno con el valor de seed.

## Implementar Deep Q-Network:  
Esta parte del código se encarga de definir la arquitectura del modelo de red neuronal utilizado para hacer predicciones de los valores Q en un algoritmo de aprendizaje por refuerzo. Aquí se explica en detalle cada parte:

Definición del número de acciones:
```
num_actions = 4
```
Aquí se define la variable num_actions con el valor 4. Esto indica el número de acciones posibles que el agente puede tomar en el entorno del juego. En este caso particular, el agente tiene 4 posibles acciones que puede realizar.

- Definición de la función create_q_model():
```
def create_q_model():
    '''Red definida por el artículo de Deepmind'''
    inputs = layers.Input(shape=(84, 84, 4,))
    
    # Circunvoluciones en los marcos de la pantalla.
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    
    layer4 = layers.Flatten()(layer3)
    
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)
    
    return keras.Model(inputs=inputs, outputs=action)
```
En esta función, se define la arquitectura del modelo de red neuronal para hacer predicciones de los valores Q. El modelo utiliza la arquitectura descrita en el artículo de DeepMind.

  -- inputs se crea utilizando la clase Input de Keras y tiene una forma de (84, 84, 4). Esto indica que la entrada del modelo es una imagen de 84x84 píxeles con 4 canales, que corresponden a los 4 marcos apilados del juego.
  
  -- Luego, se aplican tres capas de convolución (Conv2D) con diferentes tamaños de filtro, pasos de desplazamiento y funciones de activación. Cada capa convolucional extrae características de los marcos de la pantalla del juego.
  
  -- Después de las capas de convolución, se utiliza una capa de aplanamiento (Flatten) para convertir la salida en un vector unidimensional.
  
  -- A continuación, se aplica una capa densa (Dense) con 512 unidades y función de activación ReLU para procesar las características extraídas.
  
  -- Finalmente, se utiliza una capa densa de salida (Dense) con num_actions unidades y función de activación lineal para obtener las predicciones de los valores Q para cada acción posible.
 
- Creación del modelo y modelo objetivo:
```
model = create_q_model()
model_target = create_q_model()
```
Aquí se crea una instancia del modelo principal y una instancia del modelo objetivo. El modelo principal (model) se utilizará para hacer predicciones de los valores Q, mientras que el modelo objetivo (model_target) se utilizará para las predicciones de recompensas futuras en el algoritmo de aprendizaje por refuerzo. Ambos modelos tienen la misma arquitectura definida por la función create_q_model().


## Train
Este código implementa un algoritmo de aprendizaje por refuerzo utilizando un modelo de red neuronal para entrenar a un agente en un entorno de juego. Aquí se explica paso a paso el funcionamiento del código:

- Se define el optimizador:
```
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
```

Se utiliza el optimizador Adam con una tasa de aprendizaje de 0.00025 y una norma máxima de clip de 1.0. El optimizador se encargará de ajustar los pesos del modelo durante el entrenamiento.

- Se inicializan varios buffers y variables:
```action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
```
Estos buffers y variables se utilizan para almacenar la información del historial de acciones, estados, recompensas, y otras métricas relacionadas con el entrenamiento del agente.

- Se definen parámetros relacionados con la exploración y la tasa de aprendizaje:
```epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
```
Estos parámetros controlan la tasa de exploración del agente y la tasa a la que se reduce la probabilidad de tomar acciones aleatorias a medida que el entrenamiento avanza. También se define la longitud máxima del búfer de reproducción que almacena las experiencias pasadas del agente.

- Se define el bucle principal del entrenamiento:
``` while True: ```

Este bucle se ejecuta hasta que se cumpla una condición de finalización.

- Se reinicia el entorno y se inicializa la recompensa del episodio:
``` state = np.array(env.reset())
episode_reward = 0 
```

Se reinicia el entorno de juego y se inicializa la variable state con el estado inicial. Además, se inicializa la variable episode_reward para almacenar la recompensa acumulada durante el episodio actual.

- Se ejecuta el bucle principal del episodio: 
```
for timestep in range(1, max_steps_per_episode):
```
Este bucle se ejecuta para cada paso dentro del episodio hasta alcanzar el número máximo de pasos permitidos.

- Se incrementa el contador de fotogramas:
```frame_count += 1
```

Se incrementa el contador de fotogramas para realizar un seguimiento del progreso del entrenamiento.

- Se aplica epsilon-greedy para la exploración del agente:
```
if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
    action = np.random.choice(num_actions)
else:
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    action = tf.argmax(action_probs[0]).numpy()
```
Se utiliza la estrategia epsilon-greedy para decidir si el agente debe realizar una acción aleatoria o utilizar su conocimiento actual para tomar la mejor acción posible. Si el número de fotogramas es menor que epsilon_random_frames o un número aleatorio es mayor que el valor de epsilon, se selecciona una acción aleatoria. De lo contrario, se utiliza el modelo de red neuronal (model) para predecir los valores Q y se selecciona la acción con el valor Q más alto.

- Se actualiza la tasa de exploración epsilon:
``` epsilon -= epsilon_interval / epsilon_greedy_frames
epsilon = max(epsilon, epsilon_min)
```

La tasa de exploración epsilon se reduce ligeramente después de cada paso, lo que disminuye gradualmente la probabilidad de realizar acciones aleatorias a medida que el agente gana más experiencia.

- Se aplica la acción seleccionada en el entorno y se obtiene el siguiente estado, la recompensa y la bandera "done":
``` state_next, reward, done, _ = env.step(action)
state_next = np.array(state_next) 
```

El agente toma la acción seleccionada en el entorno de juego y se obtiene el siguiente estado, la recompensa correspondiente y la bandera "done", que indica si el episodio ha terminado.

- Se actualiza la recompensa acumulada del episodio y se guarda la experiencia en los buffers:
```
episode_reward += reward
action_history.append(action)
state_history.append(state)
state_next_history.append(state_next)
done_history.append(done)
rewards_history.append(reward)
state = state_next
```

La recompensa del episodio se incrementa con la recompensa obtenida en el paso actual. Además, se guarda la experiencia en los buffers correspondientes para su uso posterior en el entrenamiento.

- Se realiza la actualización del modelo:
```if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
    indices = np.random.choice(range(len(done_history)), size=batch_size)
    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_next_history[i] for i in indices])
    rewards_sample = [rewards_history[i] for i in indices]
    action_sample = [action_history[i] for i in indices]
    done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])
    future_rewards = model_target.predict(state_next_sample)
    updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample
    masks = tf.one_hot(action_sample, num_actions)
    with tf.GradientTape() as tape:
        q_values = model(state_sample)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Se realiza la actualización del modelo cada cierto número de acciones (update_after_actions) y una vez que el tamaño del búfer de experiencia alcanza el tamaño del lote (batch_size). En cada actualización, se selecciona una muestra aleatoria de experiencias del búfer de reproducción y se utilizan para actualizar los valores Q del modelo. Luego, se calcula la pérdida entre los valores Q actualizados y los valores Q predichos por el modelo. La pérdida se utiliza para calcular los gradientes y se aplica la optimización al modelo utilizando el optimizador definido anteriormente.

- Se actualiza el modelo objetivo:
``` if frame_count % update_target_network == 0:
    model_target.set_weights(model.get_weights())
```

El modelo objetivo (model_target) se actualiza periódicamente copiando los pesos del modelo principal (model). Esto se hace para estabilizar el entrenamiento y evitar que los objetivos se muevan demasiado rápidamente.

- Se realiza la gestión del tamaño del búfer de experiencia:
```
if len(rewards_history) > max_memory_length:
    del rewards_history[:1]
    del state_history[:1]
    del state_next_history[:1]
    del action_history[:1]
    del done_history[:1]
```
Si el tamaño del búfer de experiencia supera el límite definido (max_memory_length), se eliminan las experiencias más antiguas para mantener el tamaño bajo control y evitar que el búfer se llene demasiado.

- Se comprueba si el episodio ha terminado:
``` if done:
    break
```
Si la bandera "done" indica que el episodio ha terminado, se sale del bucle principal del episodio.

- Se actualiza la recompensa en ejecución y se comprueba si se ha alcanzado la condición de resolución:
``` episode_reward_history.append(episode_reward)
if len(episode_reward_history) > 100:
    del episode_reward_history[:1]
running_reward = np.mean(episode_reward_history)

episode_count += 1

if running_reward > 40:
    print("Solved at episode {}!".format(episode_count))
    break
```
La recompensa del episodio se guarda en el historial y se calcula la recompensa en ejecución tomando el promedio de las últimas 100 recompensas del historial. Si la recompensa en ejecución supera un umbral (40 en este caso), se imprime un mensaje indicando que se ha resuelto el problema y se finaliza el entrenamiento.






















