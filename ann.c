/*
Basic artifical neural network to perform XOR function
2 inputs, 1 output, 3 layers of 3 nodes
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
Generates random values from the standard normal distribution using Mox-Muller
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

/*
Initializes an array with random values from the standard normal distribution
*/
void init_rand_norm(double **array, int row, int col)
{
  for (int r = 0; r < row; r++)
  {
    for (int c = 0; c < col; r++)
    {
      array[r][c] = rand_norm();
    }
  }
}

int main(int argc, char **argv)
{
  double array[3][3];
  double *p_array[3];

  for (int i = 0; i < 3; i++){
    *p_array = array[i];
    init_rand_norm(p_array, 3, 3);
  }

  for (int row = 0; row < 3; row++)
  {
    for (int col = 0; col < 3; col++)
    {
      printf("%.3f\t", array[row][col]);
    }
    printf("\n");
  }

  return 0;
}
