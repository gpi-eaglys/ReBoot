#pragma once

#include <omp.h>
#include <vector>
#include <map>
#include <algorithm>

#include <openfhe.h>
#include "encrypted_value.h"

std::vector<EncryptedValue> row_packing_forward(std::vector<EncryptedValue> X, EncryptedValue W, int row_size);
std::vector<EncryptedValue> row_packing_backward(std::vector<EncryptedValue> delta, EncryptedValue W, int row_size);
std::vector<EncryptedValue> col_packing_forward(std::vector<EncryptedValue> X, EncryptedValue W, int row_size);
std::vector<EncryptedValue> col_packing_backward(std::vector<EncryptedValue> delta, EncryptedValue W, int row_size);
EncryptedValue packing_weight_update(std::vector<EncryptedValue> X, std::vector<EncryptedValue> delta);