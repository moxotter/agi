typedef struct layer {
  unsigned short n_in;
  unsigned short n_out;
  double *weights;  // multidimensional array with dimensions n_in x n_out
  double *biases;   // array with dimensions n_out
} layer_t;

typedef struct network {
  unsigned short n_in;  // number of inputs
  unsigned short n_out; // number of outputs
  unsigned short n_depth; // number of layers (not including input and output layer)
  layer_t *layers; // array with dimensions n_depth
} network_t;
