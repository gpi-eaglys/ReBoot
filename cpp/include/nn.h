#pragma once

#include <vector>

#include <encrypted_value.h>

using namespace std;

vector<EncryptedValue> enc_l2_loss(vector<EncryptedValue> y_true, vector<EncryptedValue> y_pred, int n_cols);
vector<EncryptedValue> enc_l2_loss_grad(vector<EncryptedValue> y_true, vector<EncryptedValue> y_pred);

vector<EncryptedValue> square_forward(vector<EncryptedValue> x);
vector<EncryptedValue> square_backward(vector<EncryptedValue> delta, vector<EncryptedValue> last_input);
vector<EncryptedValue> poly_relu_forward(vector<EncryptedValue> x);
vector<EncryptedValue> poly_relu_backward(vector<EncryptedValue> delta, vector<EncryptedValue> last_input);

tuple<EncryptedValue, EncryptedValue> init_nesterov(EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr);
tuple<EncryptedValue, EncryptedValue> compute_nesterov(EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr, EncryptedValue velocity);
