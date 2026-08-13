#pragma once

#include <vector>

#include "encrypted_value.h"
#include <openfhe.h>

int get_num_threads();

void set_num_threads(int num_threads);

std::vector<lbcrypto::Plaintext> encode_array(std::vector<std::vector<double>> array, const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, int level);

std::vector<EncryptedValue> encrypt_array(std::vector<std::vector<double>> array, const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &key_pair, int level);

std::vector<std::vector<double>> decrypt_array(std::vector<EncryptedValue> array, const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &key_pair, int n_cols);

std::vector<EncryptedValue> bootstrap_array(std::vector<EncryptedValue> array, int num_iterations, int precision);
