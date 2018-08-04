/*
 * artificial neural network framework
 *
 * todo:
 *  - implement multidimensional minimizer
 *  - implement nonlinear least-squares
 *  - implement stochastic process
 */

// GNU scientific library headers
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

// C standard library headers
#include <math.h>
#include <stdio.h>
#include <string.h>

// linux api headers
#include <sys/random.h>

/*
 * soft step (a.k.a sigmoid or logistic) activation function
 *
 * computes elementwise soft step inline
 * f(x) = 1 / (1 + e^-x)
 *
 * type : classifier
 * range : (0, 1)
 *
 * arguments :
 *  x : vector of values to read from and to write to
 */
void ann_softstep(gsl_vector *x)
{
  for (size_t n = 0; n < x->size; n++)
    x->data[n] = 1.0 / (1 + exp(-x->data[n]));
}

/*
 * soft plus activation function
 *
 * computes elementwise soft plus inline
 * f(x) = ln(1 + e^x)
 *
 * type : regressor
 * range : (0, inf)
 *
 * arguments :
 *  x : vector of values to read from and write to
 */
void ann_softplus(gsl_vector *x)
{
  for (size_t n = 0; n < x->size; n++)
    x->data[n] = log(1 + exp(x->data[n]));
}

/*
 * mean squared error loss function
 */
double ann_mse(gsl_vector *outputs, gsl_vector *targets)
{
  double mse = 0.0;

  for (size_t n = 0; n < outputs->size; n++)
    mse += pow(outputs->data[n] - targets->data[n], 2);

  mse = mse / outputs->size;

  return mse;
}

/*
 * artificial neural network layer
 *
 * layers consist of nodes (neurons) that produce outputs (activations), weights
 * that are assigned to connections of inputs (activations of other neurons) to
 * nodes, and biases (thresholds).
 */
typedef struct {
  size_t num_inputs;
  size_t num_outputs;

  gsl_matrix *weights;
  gsl_vector *biases;

  void *next_layer;
  void *prev_layer;
} ann_layer;

/*
 * artificial neural network
 *
 * networks consist of nodes (neurons) in layers. nodes in each layer are
 * connected to nodes of adjacent layers. connections are assigned weights
 * (strengths) and nodes are assigned biases (thresholds).
 */
typedef struct {
  size_t num_layers;

  ann_layer *layers;
} ann_network;

/*
 * allocates memory for an artificial neural network layer
 *
 * arguments :
 *  num_inputs : number of inputs
 *  num_outputs : number of outputs
 *
 * returns a pointer to the allocated artificial neural network layer
 */
ann_layer *ann_layer_alloc(size_t num_inputs, size_t num_outputs)
{
  ann_layer *layer = malloc(sizeof(ann_layer));

  layer->num_inputs = num_inputs;
  layer->num_outputs = num_outputs;

  layer->weights = gsl_matrix_alloc(num_inputs, num_outputs);
  layer->biases = gsl_vector_alloc(num_outputs);

  layer->next_layer = 0;
  layer->prev_layer = 0;

  return layer;
}

/*
 * frees memory of an artificial neural network layer
 *
 * arguments :
 *  layer : artificial neural network layer to free
 */
void ann_layer_free(ann_layer *layer)
{
  gsl_matrix_free(layer->weights);
  gsl_vector_free(layer->biases);

  free(layer);
}

/*
 * initializes an artificial neural network layer
 *
 * intializes weights with standard normally distributed random numbers
 * initializes biases with zero
 *
 * arguments :
 *  layer : artificial neural network layer to initialize
 *  rng : random number generator for initialization
 */
void ann_layer_init(ann_layer *layer, gsl_rng *rng)
{
  // calculate number of weights and biases
  size_t num_weights = layer->num_inputs * layer->num_outputs;
  size_t num_biases = layer->num_outputs;

  // intialize weights with Gaussian random variates with mean of zero and
  // standard deviation of one (standard normal distribution)
  for (size_t n = 0; n < num_weights; n++)
    layer->weights->data[n] = gsl_ran_gaussian(rng, 1);

  // initialize biases with zero
  for (size_t n = 0; n < num_biases; n++)
    layer->biases->data[n] = 0.0;
}

/*
 * forwards inputs to an artificial neural network layer
 *
 * computes matrix-vector product of inputs and weights and sums biases
 *
 * arguments:
 *  layer : artificial neural network layer to forward
 *  inputs : input vector to read
 *  outputs : output vector to write
 * func : activation function to call
 */
void ann_layer_forward(const ann_layer *layer, const gsl_vector *inputs,
  gsl_vector *outputs, void (*func)(gsl_vector *))
{
  // copy biases to outputs
  gsl_vector_memcpy(outputs, layer->biases);

  // compute matrix-vector product and sum of inputs, weights, and biases
  gsl_blas_dgemv(CblasTrans, 1.0, layer->weights, inputs, 1.0, outputs);

  // computes activation inline
  func(outputs);
}

/*
 * allocates memory for an artificial neural network
 *
 * arguments:
 *  num_layers : number of layers
 *  num_nodes : number of nodes in each layer
 *
 * returns a pointer to the allocated artificial neural network
 */
ann_network *ann_network_alloc(size_t num_layers, size_t num_nodes[])
{
  ann_network *network = malloc(sizeof(ann_network));
  network->num_layers = num_layers - 1; // adjust for input layer

  // layer iterators
  ann_layer *next_layer;
  ann_layer *prev_layer;

  // iterate and allocate layers
  for (size_t n = 1; n < num_layers; n++) {
    // layers have number of inputs equal to previous layers number of outputs
    next_layer = ann_layer_alloc(num_nodes[n - 1], num_nodes[n]);

    if (n == 1)
      network->layers = next_layer;
    else {
      next_layer->prev_layer = prev_layer;
      prev_layer->next_layer = next_layer;
    }

    prev_layer = next_layer;
  }

  return network;
}

/*
 * frees memory of an artificial neural network
 *
 * arguments:
 *  network : artificial neural network to free
 */
void ann_network_free(ann_network *network)
{
  // first layer
  ann_layer *layer = network->layers;

  // iterate and free layers
  while (layer != 0) {
    ann_layer *next_layer = layer->next_layer;
    ann_layer_free(layer);
    layer = next_layer;
  }
}

/*
 * initializes an artificial neural network
 *
 * arguments :
 *  network : artificial neural network to initialize
 * rng : random number generator for initialization
 */
void ann_network_init(ann_network *network, gsl_rng *rng)
{
  // first layer
  ann_layer *layer = network->layers;

  // iterate and initialize layers
  while (layer != 0) {
    ann_layer_init(layer, rng);
    layer = layer->next_layer;
  }
}

/*
 * forwards inputs to an artificial neural network layer
 *
 * computes matrix-vector product of inputs and weights and sums biases
 *
 * arguments:
 *  layer : artificial neural network layer to forward
 *  inputs : input vector to read
 *  outputs : output vector to write
 * func : activation function to call
 */
void ann_network_forward(const ann_network *network, const gsl_vector *inputs,
  gsl_vector *outputs, void (*func)(gsl_vector *))
{
  // first layer
  ann_layer *layer = network->layers;

  // allocate computational vectors
  gsl_vector *layer_inputs = gsl_vector_alloc(layer->num_inputs);
  gsl_vector *layer_outputs = gsl_vector_alloc(layer->num_outputs);

  // intialize layer inputs
  gsl_vector_memcpy(layer_inputs, inputs);

  // iterate and forward layers
  while (layer != 0) {
    if (layer != network->layers) { // not first layer
      // free and reallocate computational vectors
      gsl_vector_free(layer_inputs);
      layer_inputs = layer_outputs;
      layer_outputs = gsl_vector_alloc(layer->num_outputs);
    }

    // forward inputs to layer
    ann_layer_forward(layer, layer_inputs, layer_outputs, func);

    // next layer
    layer = layer->next_layer;
  }

  // copy last layers outputs to network outputs
  gsl_vector_memcpy(outputs, layer_outputs);

  // free computational vectors
  gsl_vector_free(layer_inputs);
  gsl_vector_free(layer_outputs);
}

/*
 * train a network to effectively solve the exclusive-or problem
 *
 * given two binary inputs, output one binary output predicting the XOR logic of
 * the given inputs
 *
 * network consists of 1 input layer of 2 nodes, 1 hidden layer of 3
 * nodes, and 1 output layer of 1 node. node outputs are produced using the
 * softstep activation function. network outputs are normalized using a
 * stochastic process. network is trained using a multidimensional minimizer on
 * the mean squared errors of the data set.
 */
int main(int argc, char *argv[])
{
  // generate random number generator seed from linux entropy
  unsigned long int seed;
  getrandom(&seed, sizeof(seed), 0);

  // allocate and initialize Mersenne Twister random number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, seed);

  // neural network structure
  size_t num_layers = 3;
  size_t num_nodes[] = {2, 3, 1};

  // allocate and initialize artificial neural network
  ann_network *network = ann_network_alloc(num_layers, num_nodes);
  ann_network_init(network, rng);

  // training set data
  // {input, input, target}
  double set[4][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
  };

  // mean squared error of training set
  double mse = 0.0;

  // allocate computational vectors
  gsl_vector *inputs = gsl_vector_alloc(2);
  gsl_vector *outputs = gsl_vector_alloc(1);
  gsl_vector *targets = gsl_vector_alloc(1);

  // iterate set data, forward inputs, and calculate loss
  for (size_t n = 0; n < 4; n++) {
    // copy input and target arrays into input and target vector
    memcpy(inputs->data, &set[n][0], sizeof(double) * 2);
    memcpy(targets->data, &set[n][2], sizeof(double));

    // forward inputs to network
    ann_network_forward(network, inputs, outputs, &ann_softstep);

    // calculate loss
    mse += ann_mse(outputs, targets);
  }

  // free computational vectors
  gsl_vector_free(inputs);
  gsl_vector_free(outputs);
  gsl_vector_free(targets);

  // mean training set mean squared errors
  mse = mse / 4.0;
  printf("MSE: %f\n", mse);

  ann_network_free(network);

  return 0;
}
