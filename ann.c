#include <math.h>
#include <stdio.h>

double sigmoid(double x)
{
  return 1.0 / (1 + exp(-x));
}

int main(int argc, char** argv)
{
  for (int x = -6; x <= 6; x++)
  {
    double y = sigmoid(x);

    printf("%i -> %f\n", x, y);
  }

  return 0;
}
