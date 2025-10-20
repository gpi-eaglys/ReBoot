#include <omp.h>
#include <vector>

#include <encrypted_value.h>
#include <openfhe.h>

using namespace std;
using namespace lbcrypto;

int get_num_threads()
{
    return omp_get_max_threads();
}

void set_num_threads(int num_threads)
{
    omp_set_num_threads(num_threads);
}

vector<Plaintext> encode_array(vector<vector<double>> array, const CryptoContext<DCRTPoly> &cc, int level)
{
    vector<Plaintext> encoded_array(array.size());
#pragma omp parallel for
    for (int i = 0; i < array.size(); i++)
    {
        Plaintext plain_value = cc->MakeCKKSPackedPlaintext(array[i], 1, level);
        encoded_array[i] = plain_value;
    }
    return encoded_array;
}

vector<EncryptedValue> encrypt_array(vector<vector<double>> array, const CryptoContext<DCRTPoly> &cc, const KeyPair<DCRTPoly> &key_pair, int level)
{
    vector<EncryptedValue> encrypted_array(array.size());
#pragma omp parallel for
    for (int i = 0; i < array.size(); i++)
    {
        Plaintext plain_value = cc->MakeCKKSPackedPlaintext(array[i], 1, level);
        Ciphertext<DCRTPoly> encrypted_value = cc->Encrypt(key_pair.publicKey, plain_value);
        encrypted_array[i] = EncryptedValue(encrypted_value);
    }
    return encrypted_array;
}

vector<vector<double>> decrypt_array(vector<EncryptedValue> array, const CryptoContext<DCRTPoly> &cc, const KeyPair<DCRTPoly> &key_pair, int n_cols)
{
    vector<vector<double>> decrypted_array(array.size());
#pragma omp parallel for
    for (int i = 0; i < array.size(); i++)
    {
        Plaintext plain_value;
        cc->Decrypt(key_pair.secretKey, array[i].getValue(), &plain_value);
        plain_value->SetLength(n_cols);
        decrypted_array[i] = plain_value->GetRealPackedValue();
    }
    return decrypted_array;
}

vector<EncryptedValue> bootstrap_array(vector<EncryptedValue> array, int num_iterations, int precision)
{
    vector<EncryptedValue> bootstrapped_array(array.size());

#pragma omp parallel for
    for (int i = 0; i < array.size(); i++)
    {
        bootstrapped_array[i] = array[i].bootstrap(num_iterations, precision);
    }

    return bootstrapped_array;
}