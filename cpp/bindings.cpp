#include <omp.h>
#include <openfhe.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>

#include <nn.h>
#include <utils.h>
#include <encrypted_value.h>
#include <packing.h>

namespace py = pybind11;

PYBIND11_MODULE(reboot_cpp, m)
{
      // Bind the EncryptedValue class
      py::class_<EncryptedValue>(m, "EncryptedValue")
          // Attributes, getters  and setters
          .def_static("set_context", &EncryptedValue::setContext, py::arg("context"))
          .def_static("get_context", &EncryptedValue::getContext)
          .def(py::init<Ciphertext<DCRTPoly>>(), py::arg("value"))
          .def("get_value", &EncryptedValue::getValue)
          .def("set_value", &EncryptedValue::setValue, py::arg("value"))
          // Operator overloading
          .def(py::self + py::self, py::arg("other"))
          .def(py::self + float(), py::arg("other"))
          .def(py::self + Plaintext(), py::arg("other"))
          .def(float() + py::self, py::arg("other"))
          .def(Plaintext() + py::self, py::arg("other"))
          .def(py::self - py::self, py::arg("other"))
          .def(py::self - float(), py::arg("other"))
          .def(py::self - Plaintext(), py::arg("other"))
          .def(float() - py::self, py::arg("other"))
          .def(Plaintext() - py::self, py::arg("other"))
          .def(py::self * py::self, py::arg("other"))
          .def(py::self * float(), py::arg("other"))
          .def(py::self * Plaintext(), py::arg("other"))
          .def(float() * py::self, py::arg("other"))
          .def(Plaintext() * py::self, py::arg("other"))
          // Other methods
          .def("eval_poly", &EncryptedValue::evalPoly, py::arg("coefficients"))
          .def("bootstrap", &EncryptedValue::bootstrap, py::arg("num_iterations") = 1, py::arg("precision") = 0)
          .def("rotate", &EncryptedValue::rotate, py::arg("index"))
          .def("fast_rotate_precomp", &EncryptedValue::fastRotatePrecomp)
          .def("fast_rotate", &EncryptedValue::fastRotate, py::arg("index"))
          .def("get_level", &EncryptedValue::getLevel)
          .def("sum", &EncryptedValue::sum, py::arg("n_cols"))
          .def("sum_rows", &EncryptedValue::sumRows, py::arg("row_size"))
          .def("sum_cols", &EncryptedValue::sumCols, py::arg("row_size"))
          .def_static("generate_sum_keys", &EncryptedValue::generate_sum_keys, py::arg("kp"), py::arg("row_size"))
          .def("get_memory_size", &EncryptedValue::get_memory_size);

      // Bind the utils
      m.def("get_num_threads", &get_num_threads);
      m.def("set_num_threads", &set_num_threads, py::arg("num_threads"));
      m.def("encode_array", [](vector<vector<double>> array, const CryptoContext<DCRTPoly> &cc, int level)
            {
            py::array out = py::cast(encode_array(array, cc, level));
            return out; }, py::arg("array"), py::arg("cc"), py::arg("level") = 0);
      m.def("encrypt_array", [](vector<vector<double>> array, const CryptoContext<DCRTPoly> &cc, const KeyPair<DCRTPoly> &key_pair, int level)
            {
            py::array out = py::cast(encrypt_array(array, cc, key_pair, level));
            return out; }, py::arg("array"), py::arg("cc"), py::arg("key_pair"), py::arg("level") = 0);
      m.def("decrypt_array", [](vector<EncryptedValue> array, const CryptoContext<DCRTPoly> &cc, const KeyPair<DCRTPoly> &key_pair, int n_cols)
            {
            py::array out = py::cast(decrypt_array(array, cc, key_pair, n_cols));
            return out; }, py::arg("array"), py::arg("cc"), py::arg("key_pair"), py::arg("n_cols"));
      m.def("bootstrap_array", [](vector<EncryptedValue> array, int num_iterations, int precision)
            {
            py::array out = py::cast(bootstrap_array(array, num_iterations, precision));
            return out; }, py::arg("array"), py::arg("num_iterations"), py::arg("precision"));

      // Bind the neural network functions
      m.def("enc_l2_loss", [](vector<EncryptedValue> y_true, vector<EncryptedValue> y_pred, int n_cols)
            {
            py::array out = py::cast(enc_l2_loss(y_true, y_pred, n_cols));
            return out; }, py::arg("y_true"), py::arg("y_pred"), py::arg("n_cols"));
      m.def("enc_l2_loss_grad", [](vector<EncryptedValue> y_true, vector<EncryptedValue> y_pred)
            {
            py::array out = py::cast(enc_l2_loss_grad(y_true, y_pred));
            return out; }, py::arg("y_true"), py::arg("y_pred"));
      m.def("square_forward", [](vector<EncryptedValue> x)
            {
            py::array out = py::cast(square_forward(x));
            return out; }, py::arg("x"));
      m.def("square_backward", [](vector<EncryptedValue> delta, vector<EncryptedValue> last_input)
            {
            py::array out = py::cast(square_backward(delta, last_input));
            return out; }, py::arg("delta"), py::arg("last_input"));
      m.def("poly_relu_forward", [](vector<EncryptedValue> x)
            {
            py::array out = py::cast(poly_relu_forward(x));
            return out; }, py::arg("x"));
      m.def("poly_relu_backward", [](vector<EncryptedValue> delta, vector<EncryptedValue> last_input)
            {
            py::array out = py::cast(poly_relu_backward(delta, last_input));
            return out; }, py::arg("delta"), py::arg("last_input"));
      m.def("init_nesterov", [](EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr)
            {
            auto out = init_nesterov(weight_gradient, momentum, weight_decay, layer_weights, lr);
            return py::make_tuple(get<0>(out), get<1>(out)); }, py::arg("weight_gradient"), py::arg("momentum"), py::arg("weight_decay"), py::arg("layer_weights"), py::arg("lr"));
      m.def("compute_nesterov", [](EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr, EncryptedValue velocity)
            {
            auto out = compute_nesterov(weight_gradient, momentum, weight_decay, layer_weights, lr, velocity);
            return py::make_tuple(get<0>(out), get<1>(out)); }, py::arg("weight_gradient"), py::arg("momentum"), py::arg("weight_decay"), py::arg("layer_weights"), py::arg("lr"), py::arg("velocity"));

      // Bind packing algorithms
      m.def("row_packing_forward", [](vector<EncryptedValue> X, EncryptedValue W, int row_size)
            {
            py::array out = py::cast(row_packing_forward(X, W, row_size));
            return out; }, py::arg("X"), py::arg("W"), py::arg("row_size"));
      m.def("row_packing_backward", [](vector<EncryptedValue> delta, EncryptedValue W, int row_size)
            {
            py::array out = py::cast(row_packing_backward(delta, W, row_size));
            return out; }, py::arg("delta"), py::arg("W"), py::arg("row_size"));
      m.def("col_packing_forward", [](vector<EncryptedValue> X, EncryptedValue W, int row_size)
            {
            py::array out = py::cast(col_packing_forward(X, W, row_size));
            return out; }, py::arg("X"), py::arg("W"), py::arg("row_size"));
      m.def("col_packing_backward", [](vector<EncryptedValue> delta, EncryptedValue W, int row_size)
            {
            py::array out = py::cast(col_packing_backward(delta, W, row_size));
            return out; }, py::arg("delta"), py::arg("W"), py::arg("row_size"));
      m.def("packing_weight_update", [](vector<EncryptedValue> X, vector<EncryptedValue> delta)
            { return packing_weight_update(X, delta); }, py::arg("X"), py::arg("delta"));
}