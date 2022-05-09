#include "defcor_agg.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("defcor_forward", &defcor_forward, "defcor_forward");
  m.def("defcor_backward", &defcor_backward, "defcor_backward");
  m.def("defagg_forward", &defagg_forward, "defagg_forward");
  m.def("defagg_backward", &defagg_backward, "defagg_backward");
}
