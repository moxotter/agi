/*
Basic artifical neural network to perform XOR function
2 inputs, 1 output, 3 layers
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
Box-Muller transform
Generates random values from the standard normal distribution
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

int main(int argc, char** argv)
{
  // activation function test
  for (int x = -6; x <= 6; x++)
  {
    printf("softstep(%i) = %f\tsoftplus(%i) = %f\n", x, softstep(x), x, softplus(x));
  }

  // random normal function test
  for (int row = 0; row < 3; row++)
  {
    for (int col = 0; col < 3; col++)
    {
      printf("%3f\t", rand_norm());
    }
    printf("\n");
  }

  return 0;
}
