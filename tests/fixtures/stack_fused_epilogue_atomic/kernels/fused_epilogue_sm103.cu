#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ar_residual_rmsnorm", []() {});
  m.def("moe_finalize_ar_rmsnorm", []() {});
}
