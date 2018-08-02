/* Artificial Neural Network
Input layer: 2 nodes
Hidden layers: 1 layer with 3 nodes
Output layer: 1 node

Target: Classify inputs as exclusive or
*/

#include <sys/random.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ann.h"

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

  // replace rand with getrandom to generate uniform distribution between 0 and 1
  u1 = rand() / (double)RAND_MAX;
  u2 = rand() / (double)RAND_MAX;

  z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(2 * M_PI * u2);

  return z0;
}

int main(int argc, char **argv)
{
  return 0;
}
