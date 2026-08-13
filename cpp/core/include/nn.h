#pragma once

#include <vector>

#include "encrypted_value.h"


std::vector<EncryptedValue> enc_l2_loss(std::vector<EncryptedValue> y_true, std::vector<EncryptedValue> y_pred, int n_cols);
std::vector<EncryptedValue> enc_l2_loss_grad(std::vector<EncryptedValue> y_true, std::vector<EncryptedValue> y_pred);

std::vector<EncryptedValue> square_forward(std::vector<EncryptedValue> x);
std::vector<EncryptedValue> square_backward(std::vector<EncryptedValue> delta, std::vector<EncryptedValue> last_input);
std::vector<EncryptedValue> poly_relu_forward(std::vector<EncryptedValue> x);
std::vector<EncryptedValue> poly_relu_backward(std::vector<EncryptedValue> delta, std::vector<EncryptedValue> last_input);

std::tuple<EncryptedValue, EncryptedValue> init_nesterov(EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr);
std::tuple<EncryptedValue, EncryptedValue> compute_nesterov(EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr, EncryptedValue velocity);
