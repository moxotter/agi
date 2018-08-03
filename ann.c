/* Artificial Neural Network
Input layer: 2 nodes
Hidden layers: 1 layer with 3 nodes
Output layer: 1 node

Target: Classify inputs as exclusive or
*/

#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <stdio.h>

#include <sys/random.h>

int main(int argc, char **argv)
{
  // generate seed
  unsigned long int s;
  getrandom(&s, sizeof(s), 0);

  // allocate and initialize Mersenne Twister random number generator
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, s);

  // generate random numbers with standard normal distribution
  for (int i = 0; i < 10; i++)
    printf("%f\n", gsl_ran_gaussian(r, 1));

  return 0;
}
