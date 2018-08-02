/*
Basic artifical neural network to perform XOR function
2 inputs, 1 output, 3 layers of 3 nodes
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Activation Function
Name:   Soft step (sigmoid)
Type:   Classifier
Range:  (-1, 1)
*/
double softstep(double x)
{
  return 1.0 / (1 + exp(-x));
}

/* Activation Function
Name:   Soft plus
Type:   Regressor
Range:  (0, inf)
*/
double softplus(double x)
{
  return log(1 + exp(x));
}

/*
Generates random values from the standard normal distribution using Box-Muller
transform
*/
double rand_norm()
{
  double u1, u2, z0, z1;

  u1 = rand() / (double)RAND_MAX;
  u2 = rand() / (double)RAND_MAX;

  z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(2 * M_PI * u2);

  return z0;
}

/* Initialize Random Standard Normal Distribution
*/
void init_rand_norm(double *p_array, int size)
{
  for (int i = 0; i < size; i++) {
    p_array[i] = rand_norm();
  }
}

/* Artificial Neural Network Layer
Defines number of inputs and outputs
Number of outputs is equal to number of nodes
Defines biases and weights
Number of biases is equal to number of nodes
Number of weights is equal to number of inputs multiplied by number of outputs
*/
struct t_layer {
  int n_in;
  int n_out;

  double *p_weights;  // size of in * out
  double *p_biases;   // size of out
};

/* Initialize Zero
*/
void init_zero(double *p_array, int size)
{
  for (int i = 0; i < size; i++) {
    p_array[i] = 0;
  }
}

/* Initialize Artificial Neural Network Layer
Initrializes an artificial neural network layer with random standard normal
distribution weights and zero biases.
*/
void init_layer(struct t_layer *p_layer, int n_in, int n_out)
{
  p_layer->n_in = n_in;
  p_layer->n_out = n_out;

  double weights[n_in][n_out];
  double *p_weights = &weights[0][0];
  init_rand_norm(p_weights, n_in * n_out);
  p_layer->p_weights = p_weights;

  double biases[n_out];
  double *p_biases = &biases[0];
  init_zero(p_biases, n_out);
  p_layer->p_biases = p_biases;
}

int main(int argc, char **argv)
{
  int n_in = 2;
  int n_out = 3;

  struct t_layer layer;
  init_layer(&layer, n_in, n_out);

  double array[n_in][n_out];
  double *p_array = &array[0][0];
  srand(time(NULL));
  init_rand_norm(p_array, n_in * n_out);

  for (int row = 0; row < n_in; row++) {
    for (int col = 0; col < n_out; col++) {
      printf("%.3f\t", array[row][col]);
    }
    printf("\n");
  }

  return 0;
}
