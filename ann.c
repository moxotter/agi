/* Artificial Neural Network
Input layer: 2 nodes
Hidden layers: 1 layer with 3 nodes
Output layer: 1 node

Target: Classify inputs as exclusive or
*/

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

#include <math.h>
#include <stdio.h>

#include <sys/random.h>

/* softstep activation function
v : apply to vector
*/
void softstep(gsl_vector *v)
{
  for (size_t n = 0; n < v->size; n++)
    v->data[n] = 1.0 / (1.0 + exp(v->data[n]));
}

// artificial neural network layer
typedef struct {
  size_t size_in;
  size_t size_out;
  gsl_matrix *weights;
  gsl_vector *biases;
} ann_layer;

/* allocate artificial neural network layer
size_in : number of inputs
size_out : number of outputs
r : pointer to random number generator
*/
ann_layer *ann_layer_alloc(size_t size_in, size_t size_out, const gsl_rng *r)
{
  // allocate weights and biases
  gsl_matrix *weights = gsl_matrix_alloc(size_in, size_out);
  gsl_vector *biases = gsl_vector_alloc(size_out);

// initialize weights with standard normal distribution random numbers
  for (size_t i = 0; i < size_in * size_out; i++)
    weights->data[i] = gsl_ran_gaussian(r, 1);

// initialize biases with zero
  gsl_vector_set_zero(biases);

  ann_layer *layer = malloc(sizeof(ann_layer));
  layer->size_in = size_in;
  layer->size_out = size_out;
  layer->weights = weights;
  layer->biases = biases;

  return layer;
}

// free allocated artificial neural network layer
void ann_layer_free(ann_layer *layer)
{
  gsl_matrix_free(layer->weights);
  gsl_vector_free(layer->biases);
  free(layer);
}

/* forwards inputs to an artificial neural network layer
l : forward to layer
i : input to forward
o : output of layer
a : activation function
*/
void ann_layer_forward(const ann_layer *l, const gsl_vector *i, gsl_vector *o, const void (*a)(gsl_vector *))
{
  // copy biases to outputs for dgemv
  gsl_vector_memcpy(o, l->biases);

  // product of inputs and weights and sum of biases
  gsl_blas_dgemv(CblasTrans, 1.0, l->weights, i, 1.0, o);

  // apply activation function
  (*a)(o);
}

int main(int argc, char *argv[])
{
  // generate seed
  unsigned long int s;
  getrandom(&s, sizeof(s), 0);

  // allocate and initialize Mersenne Twister random number generator
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, s);

  ann_layer *layer = ann_layer_alloc(2, 3, r);

  printf("Weights:\n");

  for (size_t row = 0; row < 2; row++) {
    for (size_t col = 0; col < 3; col++)
      printf("%f\t", gsl_matrix_get(layer->weights, row, col));

    printf("\n");
  }

  printf("Biases:\n");

  for (size_t n = 0; n < 3; n++)
    printf("%f\t", gsl_vector_get(layer->biases, n));

  printf("\n");

  double arr[2] = {0, 1};
  gsl_vector_view arr_view = gsl_vector_view_array(arr, 2);
  gsl_vector *inputs = &arr_view.vector;

  printf("Inputs:\n");
  for (size_t n = 0; n < inputs->size; n++)
    printf("%f\t", inputs->data[n]);
  printf("\n");

  gsl_vector *outputs = gsl_vector_alloc(3);
  ann_layer_forward(layer, inputs, outputs, &softstep);

  printf("Outputs:\n");
  for (size_t n = 0; n < outputs->size; n++)
    printf("%f\t", outputs->data[n]);
  printf("\n");

  gsl_vector_free(outputs);
  ann_layer_free(layer);

  return 0;
}
