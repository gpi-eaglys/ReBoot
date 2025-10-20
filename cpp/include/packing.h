#pragma once

#include <omp.h>
#include <vector>
#include <map>
#include <algorithm>

#include <openfhe.h>
#include <encrypted_value.h>

using namespace lbcrypto;
using namespace std;

vector<EncryptedValue> row_packing_forward(vector<EncryptedValue> X, EncryptedValue W, int row_size);
vector<EncryptedValue> row_packing_backward(vector<EncryptedValue> delta, EncryptedValue W, int row_size);
vector<EncryptedValue> col_packing_forward(vector<EncryptedValue> X, EncryptedValue W, int row_size);
vector<EncryptedValue> col_packing_backward(vector<EncryptedValue> delta, EncryptedValue W, int row_size);
EncryptedValue packing_weight_update(vector<EncryptedValue> X, vector<EncryptedValue> delta);