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

typedef struct {
  gsl_vector *inputs;
  gsl_vector *targets;
  size_t num_layers;
  size_t *num_nodes;
} ann_params;

void ann_fdf(const gsl_vector *x, void *params, double *f, gsl_vector *g)
{
  double *base = x->data;

  ann_params *p = (ann_params *)params;
  gsl_vector *inputs = p->inputs;
  gsl_vector *targets = p->targets;
  gsl_vector *outputs;

  size_t num_inputs;
  size_t num_outputs;

  for (size_t n = 1; n < p->num_layers; n++) {
    num_inputs = p->num_nodes[n - 1];
    num_outputs = p->num_nodes[n];

    gsl_matrix_const_view w = gsl_matrix_const_view_array(base, num_inputs, num_outputs);
    const gsl_matrix *weights = &w.matrix;
    base += num_inputs * num_outputs;

    gsl_vector_const_view b = gsl_vector_const_view_array(base, num_outputs);
    const gsl_vector *biases = &b.vector;
    base += num_outputs;

    outputs = gsl_vector_alloc(num_outputs);
    gsl_vector_memcpy(outputs, biases);
    gsl_blas_dgemv(CblasTrans, 1.0, weights, inputs, 1.0, outputs);

    for (size_t n = 0; n < num_outputs; n++)
      outputs->data[n] = 1.0 / (1.0 + exp(-outputs->data[n]));

    if (n > 1)
      gsl_vector_free(inputs);

    inputs = outputs;
  }

  double loss = 0.0;

  for (size_t n = 0; n < outputs->size; n++)
    loss += pow(outputs->data[n] - targets->data[n], 2.0) / (double)outputs->size;

  *f = loss;
}

int main(int argc, char *argv[])
{
  unsigned long s;
  getrandom(&s, sizeof(s), 0);

  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, s);

  size_t num_layers = 3;
  size_t num_nodes[] = {2, 3, 1};

  ann_params p = {
    .num_layers = num_layers,
    .num_nodes = num_nodes
  };

  size_t num_x = 0;

  for (size_t n = 1; n < p.num_layers; n++)
    num_x += p.num_nodes[n - 1] * p.num_nodes[n] + p.num_nodes[n];

  gsl_vector *x = gsl_vector_alloc(num_x);
  double *base = x->data;

  size_t num_weights;
  size_t num_biases;

  for (size_t n = 1; n < p.num_layers; n++) {
    num_weights = p.num_nodes[n - 1] * p.num_nodes[n];
    num_biases = p.num_nodes[n];

    for (size_t w = 0; w < num_weights; w++, base++)
      *base = gsl_ran_gaussian(r, 1.0);

    for (size_t b = 0; b < num_biases; b++, base++)
      *base = 0.0;
  }

  double data[4][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
  };

  base = &data[0][0];

  gsl_vector_view i = gsl_vector_view_array(base, 2);
  p.inputs = &i.vector;
  base += 2;

  gsl_vector_view t = gsl_vector_view_array(base, 1);
  p.targets = &t.vector;
  base += 1;

  double f;
  gsl_vector *g = gsl_vector_alloc(1);

  ann_fdf(x, &p, &f, g);

  printf("%f\n", f);

  return 0;
}
