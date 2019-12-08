import numpy as np

class CM:

  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Numero aleatorio generado con la semilla seed
    seed = 1234
    np_rng = np.random.RandomState( seed )

    self.weights = np.asarray( # Transform list to array
                        np_rng.uniform( # Get samples of a uniform distribution
                            # Limite inferior
			                low = -np.pi/2 ,
                            # Limite superior
                       	    high = np.pi/2 ,
                       	    # Numero de valores a generar ( n_capas_visibles * n_capas_ocultas )
                       	    size = (num_visible, num_hidden)
                       	)
                    )


    # Insertar valores iniciales de error a 0
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)
    # insert values to all index // axis=0:y  axis=1:x
    # np.insert( data, start_from_index, value, axis )

  def train(self, data, max_epochs = 1000, learning_rate = 0.5):

    # Numero de datos de entenamiento
    num_examples = data.shape[0] # Array length (x,y)

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):
      # Matriz de activacion(energia) (datos * pesos)
      pos_hidden_activations = np.dot(data, self.weights)
      # print("\nHidden activations: ")
      # print(pos_hidden_activations)

      # Funcion de activacion ( Cauchy: variacion de energia )
      pos_hidden_probs = self._logistic(pos_hidden_activations) # Matriz de probabilidades de estado
      # Fijar el error a 1
      pos_hidden_probs[:,0] = 1

      # Umbral de activacion
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1) # Matriz de estados resultado

      # Multiplicacion de matriz transpuesta datos * probabilidades
      pos_associations = np.dot(data.T, pos_hidden_probs) # Matriz de estados transpuestos

      ## Reconstruir las unidades de entrada
      # Matriz de activacion inversa (datos * transpuesta_de_pesos)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T) # Matriz de estados
      # Funcion de activacion inversa ( Cauchy )
      neg_visible_probs = self._logistic(neg_visible_activations) # Matriz de probabilidades
      # Despues de realizar los calculos regresar el error a 0
      neg_visible_probs[:,0] = 1

      # Multiplicacion de las matrices probabilidad y peso
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)

      # Funcion de activacion inversa
      neg_hidden_probs = self._logistic(neg_hidden_activations)

      # Multiplicacion de las matrices de probabilidad
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Actualizar pesos
      # Variacion de energia * tasa de aprendizaje
      # Variacion de energia: Energia positiva - Energia negativa
      # Como estamos en un array, debemos dividir la variacion de energia entre el numero de iteraciones
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      # El error es la diferencia entre el valor de entrada, con el valor reconstruido de entrada
      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        print("Entrenamiento %s: el error es %s" % (epoch+1, error))

  def run_visible(self, data):
    # Longitud del array
    num_examples = data.shape[0]

    # Crear matriz bidimensional de y=numero_de_ejemplos ; x=numero_de_capas_ocultas+1
    hidden_states = np.ones((num_examples, self.num_hidden + 1))

    # Agregar columna de errores
    data = np.insert(data, 0, 1, axis = 1)

    # Calcular la matriz de activacion de la capa oculta
    hidden_activations = np.dot(data, self.weights)
    # Calcular la matriz de probabilidades de estado de las neuronas ocultas
    hidden_probs = self._logistic(hidden_activations)
    # Encender las neuronas ocultas de acuerdo a sus probabilidades
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1) # Mayor que un umbral aleatorio
    # Fijar el error a 1
    hidden_states[:,0] = 1

    # Ignorar las neuronas de error
    hidden_states = hidden_states[:,1:]
    return hidden_states

  def run_hidden(self, data):
    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states

  def _logistic(self, x): # Funcion de probabilidad de activacion
    # return 1.0 / (1 + np.exp(-x))
    return .5 + ( np.arctan(x) / np.pi ) # Funcion de Cauchy

if __name__ == '__main__': # Aqui se configuran los parametros
  # Parametros
  capas_visibles = 9
  capas_ocultas = 5
  iteraciones = 30000

  r = CM(num_visible = capas_visibles, num_hidden = capas_ocultas)
  training_data = np.array(
    [
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi
        [1,1,1,0,0,0,0,0,0], # SciFi

        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded
        [0,0,0,1,1,1,0,0,0], # Awarded

        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1], # Anime
        [0,0,0,0,0,0,1,1,1]  # Anime
    ]
  )
  # [      1       ,   2    ,   3   ,     4     ,    5    ,      6       ,     7      ,     8      ,      9        ]
  # [ Harry Potter , Avatar , LOTR3 , Gladiator , Titanic ,  The Origin  , Death Note , Fairy Tail , Akame Ga Kill ]

  print("\n\nDatos de entrenamiento: ")
  print(training_data)

  print("\n\nPesos antes del enternamiento (aleatorios)")
  print( np.around(r.weights,4) )

  r.train(training_data, max_epochs = iteraciones)

  print("\n\nPesos despues de entrenamiento")
  print( np.around(r.weights,4) )

  print("\n")

  ## INTRODUCIR VALORES DE PRUEBA
  test = [ # Arreglo con opiniones de prueba
      [[1,1,1,0,0,1,1,1,1]],
      [[1,1,1,0,0,0,1,1,0]],
      [[0,0,1,1,0,0,0,1,1]]
  ]
  for opinion in test:
    print("\nTest:    ", " ".join( map(str,opinion[0]) ), " # Pertenece a ")
    grupos = r.run_visible( np.asarray(opinion) ) # Obtener los grupos a los cuales pertenece la opinion
    # print(grupos)
    if grupos[0].any() == 0: # Si algun elemento del array es 0
      print("No pertenece a ningun grupo")
    resultados = []
    for i in grupos:
      _i = 0
      for j in i:
        if j==1:
          grupo = [[0,0,0,0]] # Array con de capas_ocultas- elementos
          grupo = np.insert(grupo,_i,1,axis=1)
          resultado = r.run_hidden(grupo)
          valores = []
          error = 0
          _j=0
          for j in resultado[0]:
            valores.append( int(j) )
            if opinion[0][_j] != int(j):
              error+=1
            _j+=1
          print("Grupo",_i+1,":", " ".join( map(str,valores) ), " # error:",error ) # Opinion del grupo
        _i+=1

  print("\n")

# """
  ## IMPRIMIR GRUPOS
  for i in range(capas_ocultas):
    ocultas = [[0,0,0,0]] # Array con de capas_ocultas- elementos
    ocultas = np.insert(ocultas,i,1,axis=1)
    machine = np.array( ocultas ) # Arreglo con grupo de prueba
    # print("Grupo ",i+1,": ",machine) # Dato de grupo
    # print("Valores: ",r.run_hidden(machine)) # Opinion del grupo
    resultado = r.run_hidden(machine)[0]
    valores = []
    for j in resultado:
      valores.append( int(j) )
    print("Grupo",i+1,":", " ".join( map(str,valores) ) ) # Opinion del grupo

  print("\n")
# """




"""

"""
