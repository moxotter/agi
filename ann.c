/* Artificial Neural Network
Input layer: 2 nodes
Hidden layers: 1 layer with 3 nodes
Output layer: 1 node

Target: Classify inputs as exclusive or
*/

#include <gsl/gsl_cblas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

#include <stdio.h>

#include <sys/random.h>

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

/* frees allocated artificial neural network layer
*/
void ann_layer_free(ann_layer *layer)
{
  gsl_matrix_free(layer->weights);
  gsl_vector_free(layer->biases);
  free(layer);
}

int main(int argc, char **argv)
{
  // generate seed
  unsigned long int s;
  getrandom(&s, sizeof(s), 0);

  // allocate and initialize Mersenne Twister random number generator
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, s);

  ann_layer *layer = ann_layer_alloc(2, 3, r);

  for (size_t row = 0; row < 2; row++) {
    for (size_t col = 0; col < 3; col++)
      printf("%f\t", gsl_matrix_get(layer->weights, row, col));

    printf("\n");
  }

  ann_layer_free(layer);

  return 0;
}
