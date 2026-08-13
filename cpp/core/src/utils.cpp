#include <omp.h>
#include <vector>

#include <encrypted_value.h>
#include <openfhe.h>

int get_num_threads()
{
    return omp_get_max_threads();
}

void set_num_threads(int num_threads)
{
    omp_set_num_threads(num_threads);
}

std::vector<lbcrypto::Plaintext> encode_array(std::vector<std::vector<double>> array, const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, int level)
{
    std::vector<lbcrypto::Plaintext> encoded_array(array.size());
#pragma omp parallel for
    for (size_t i = 0; i < array.size(); i++)
    {
        lbcrypto::Plaintext plain_value = cc->MakeCKKSPackedPlaintext(array[i], 1, level);
        encoded_array[i] = plain_value;
    }
    return encoded_array;
}

std::vector<EncryptedValue> encrypt_array(std::vector<std::vector<double>> array, const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &key_pair, int level)
{
    std::vector<EncryptedValue> encrypted_array(array.size());
#pragma omp parallel for
    for (size_t i = 0; i < array.size(); i++)
    {
        lbcrypto::Plaintext plain_value = cc->MakeCKKSPackedPlaintext(array[i], 1, level);
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypted_value = cc->Encrypt(key_pair.publicKey, plain_value);
        encrypted_array[i] = EncryptedValue(encrypted_value);
    }
    return encrypted_array;
}

std::vector<std::vector<double>> decrypt_array(std::vector<EncryptedValue> array, const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &key_pair, int n_cols)
{
    std::vector<std::vector<double>> decrypted_array(array.size());
#pragma omp parallel for
    for (size_t i = 0; i < array.size(); i++)
    {
        lbcrypto::Plaintext plain_value;
        cc->Decrypt(key_pair.secretKey, array[i].getValue(), &plain_value);
        plain_value->SetLength(n_cols);
        decrypted_array[i] = plain_value->GetRealPackedValue();
    }
    return decrypted_array;
}

std::vector<EncryptedValue> bootstrap_array(std::vector<EncryptedValue> array, int num_iterations, int precision)
{
    std::vector<EncryptedValue> bootstrapped_array(array.size());

#pragma omp parallel for
    for (size_t i = 0; i < array.size(); i++)
    {
        bootstrapped_array[i] = array[i].bootstrap(num_iterations, precision);
    }

    return bootstrapped_array;
}