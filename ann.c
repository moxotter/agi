/* Artificial Neural Network
Input layer: 2 nodes
Hidden layers: 1 layer with 3 nodes
Output layer: 1 node

Target: Classify inputs as exclusive or
*/

#include <sys/random.h>
#include <math.h>
#include <stdio.h>

#include "ann.h"
#include "mt19937-64.h"

/* classifier activation function with (0, 1) interval
Sigmoid/logisitic
derivative of softplus */
double softstep(double x)
{
  return 1.0 / (1 + exp(-x));
}

/* regressor activation function with [0, inf) interval
antiderivative of softstep */
double softplus(double x)
{
  return log(1 + exp(x));
}

/* generate random numbers with standard normal distribution
Box-Muller transform */
double rand_norm()
{
  double u1, u2, z0, z1;

  // Mersenne Twister with [0, 1] interval
  u1 = genrand64_real1();
  u2 = genrand64_real1();

  z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(2 * M_PI * u2);

  return z0;
}

int main(int argc, char **argv)
{
  // generate seed from linux entropy
  unsigned long long seed;
  getrandom(&seed, sizeof(seed), 0);

  // initialize Mersenne Twister
  init_genrand64(seed);

  for (int i = 0; i < 10; i++) {
    printf("%f\n", rand_norm());
  }

  return 0;
}
