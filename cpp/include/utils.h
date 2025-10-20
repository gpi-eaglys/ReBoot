#pragma once

#include <vector>

#include <encrypted_value.h>
#include <openfhe.h>

using namespace std;
using namespace lbcrypto;

int get_num_threads();
void set_num_threads(int num_threads);
vector<Plaintext> encode_array(vector<vector<double>> array, const CryptoContext<DCRTPoly> &cc, int level);
vector<EncryptedValue> encrypt_array(vector<vector<double>> array, const CryptoContext<DCRTPoly> &cc, const KeyPair<DCRTPoly> &key_pair, int level);
vector<vector<double>> decrypt_array(vector<EncryptedValue> array, const CryptoContext<DCRTPoly> &cc, const KeyPair<DCRTPoly> &key_pair, int n_cols);
vector<EncryptedValue> bootstrap_array(vector<EncryptedValue> array, int num_iterations, int precision);
