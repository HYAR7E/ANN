import numpy as np

class RBM:

  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)). One could vary the 
    # standard deviation by multiplying the interval with appropriate value.
    # Here we initialize the weights with mean 0 and standard deviation 0.1. 
    # Reference: Understanding the difficulty of training deep feedforward 
    # neural networks by Xavier Glorot and Yoshua Bengio
    np_rng = np.random.RandomState(1234) # Create random numbers with seed '1234'

    self.weights = np.asarray( # Transform list to array
                        np_rng.uniform( # Get samples of a uniform distribution
			                low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)), # Min value to create >=
                       	    high=0.1 * np.sqrt(6. / (num_hidden + num_visible)), # Max value to create <
                       	    size=(num_visible, num_hidden) # Number of elements to create (a,b) => a*b
                       	)
                    )


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)
    # insert values to all index // axis=0:y  axis=1:x
    # np.insert( data, start_from_index, value, axis )

  def train(self, data, max_epochs = 1000, learning_rate = 0.5):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0] # Array length (x,y)

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs): # Do all the epochs
      # Clamp to the data and sample from the hidden units.
      # (This is the "positive CD phase", aka the reality phase.)

      pos_hidden_activations = np.dot(data, self.weights) # Matrix multiplication
      # print("\nHidden activations: ")
      # print(pos_hidden_activations)

      pos_hidden_probs = self._logistic(pos_hidden_activations) # Gauss or Cauchy function
      # After calcs, return the bias values to 1 again
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      # print("Epoch ",epoch,": ",pos_hidden_probs)

      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1) # Activation umbral
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs) # Matrix multiplication

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T) # Matrix multiplication
      neg_visible_probs = self._logistic(neg_visible_activations) # Gauss or Cauchy function
      # After calcs, return the bias values to 1 again
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.

    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1

    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states

  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.

    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

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

  def _logistic(self, x): # Probability activation function
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__': # CONFIGURE PARAMETERS HERE
  # Parametros
  capas_visibles = 9
  capas_ocultas = 5
  iteraciones = 1000

  r = RBM(num_visible = capas_visibles, num_hidden = capas_ocultas)
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
  print(r.weights)
  r.train(training_data, max_epochs = iteraciones)

  print("\n\nWeights after training")
  print(r.weights)

  print("\n")

  ## INTRODUCIR VALORES DE PRUEBA
  test = [ # Arreglo con opiniones de prueba
      [[1,1,1,0,0,1,1,1,1]],
      [[1,1,1,0,0,0,1,1,0]],
      [[1,1,1,1,1,1,1,1,1]],
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
Iterations = 10 000
Group 1:  [[0. 0. 0. 1. 1. 0. 1. 1. 1.]]
Group 2:  [[1. 1. 1. 0. 1. 1. 0. 0. 0.]]
Group 3:  [[1. 1. 1. 0. 0. 0. 1. 1. 1.]]

Iterations = 20 000
Group 1:  [[0. 0. 0. 1. 1. 1. 0. 1. 1.]]
Group 2:  [[1. 1. 0. 1. 1. 1. 0. 0. 0.]]
Group 3:  [[1. 1. 1. 0. 0. 0. 1. 1. 1.]]

Iterations = 50 000
Group 1:  [[0. 0. 0. 1. 1. 1. 1. 1. 1.]]
Group 2:  [[1. 1. 1. 1. 1. 1. 0. 0. 0.]]
Group 3:  [[1. 1. 1. 0. 0. 0. 1. 1. 1.]]


Seed = 1234
Hidden Layers = 5
Iterations = 50 000
Ref:      [[ 1 2 3 4 5 6 7 8 9 ]]
Group 1:  [[ 0 0 0 0 1 0 1 1 1 ]] // Anime
Group 2:  [[ 1 1 1 1 1 1 0 0 0 ]] // SciFi && Awarded
Group 3:  [[ 0 1 1 0 0 0 1 1 1 ]] // Anime && SciFi
Group 4:  [[ 1 1 1 0 0 0 0 0 0 ]] // SciFi
Group 5:  [[ 0 0 0 1 1 1 0 0 0 ]] // Awarded

Iterations = 50 000

Test 1:   [[ 0 0 0 0 0 0 1 1 1 ]]

Test 2:   [[ 0 0 0 1 1 1 0 0 0 ]]
Group 1:  [[ 0 0 0 0 1 0 1 1 1 ]] // Anime
Group 2:  [[ 1 1 1 1 1 1 0 0 0 ]] // SciFi && Awarded
Group 5:  [[ 0 0 0 1 1 1 0 0 0 ]] // Awarded

Test 3:   [[ 1 1 1 0 0 0 0 0 0 ]]
Group 2:  [[ 1 1 1 1 1 1 0 0 0 ]] // SciFi && Awarded
Group 3:  [[ 0 1 1 0 0 0 1 1 1 ]] // Anime && SciFi
Group 4:  [[ 1 1 1 0 0 0 0 0 0 ]] // SciFi

Test 4:   [[ 1 1 1 0 0 1 1 1 1 ]]
Group 3:  [[ 0 1 1 0 0 0 1 1 1 ]] // Anime && SciFi

-----------------------------------------------------

Seed = 0
Hidden Layers = 5
Iterations = 50 000
Ref:      [[ 1 2 3 4 5 6 7 8 9 ]]
Group 1:  [[ 1 1 1 1 1 1 0 0 0 ]] // SciFi && Awarded
Group 2:  [[ 0 1 0 0 0 0 1 1 1 ]] // Anime
Group 3:  [[ 0 0 0 1 1 1 0 0 0 ]] // Awarded
Group 4:  [[ 1 1 1 0 0 0 0 0 0 ]] // SciFi
Group 5:  [[ 0 0 0 0 1 0 1 1 1 ]] // Anime

Iterations = 50 000

Test 1:   [[ 0 0 0 0 0 0 1 1 1 ]]
Group 2:  [[ 0 1 0 0 0 0 1 1 1 ]] // Anime
Group 5:  [[ 0 0 0 0 1 0 1 1 1 ]] // Anime

Test 2:   [[ 0 0 0 1 1 1 0 0 0 ]]
Group 1:  [[ 1 1 1 1 1 1 0 0 0 ]] // SciFi && Awarded
Group 3:  [[ 0 0 0 1 1 1 0 0 0 ]] // Awarded
Group 5:  [[ 0 0 0 0 1 0 1 1 1 ]] // Anime

Test 3:   [[ 1 1 1 0 0 0 0 0 0 ]]
Group 1:  [[ 1 1 1 1 1 1 0 0 0 ]] // SciFi && Awarded
Group 2:  [[ 0 1 0 0 0 0 1 1 1 ]] // Anime
Group 4:  [[ 1 1 1 0 0 0 0 0 0 ]] // SciFi

Test 4:   [[ 1 1 1 0 0 1 1 1 1 ]]
Group 2:  [[ 0 1 0 0 0 0 1 1 1 ]] // Anime

"""

