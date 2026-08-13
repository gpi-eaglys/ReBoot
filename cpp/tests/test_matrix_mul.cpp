#include <gtest/gtest.h>

#include <openfhe.h>
#include <encrypted_value.h>
#include <utils.h>
#include <packing.h>

TEST(MatrixMul, RowPackedMatrixMultiplication)
{
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(1);
    parameters.SetScalingModSize(50);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetRingDim(1 << 12);

    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    EncryptedValue::setContext(cc);

    lbcrypto::KeyPair<lbcrypto::DCRTPoly> kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);
    cc->EvalSumKeyGen(kp.secretKey);

    // Row-packed matrix multiplication (see packing.cpp): each row X[i] is
    // encrypted as its features "expanded" (each feature repeated col_size
    // times), W is encrypted as the row-major flattened row_size x col_size
    // weight matrix, and row_packing_forward computes y[i] = X[i] @ W.
    const int col_size = 2;
    EncryptedValue::generate_sum_keys(kp, col_size);

    // W = [[1, 2], [3, 4]] (row_size=2 input features x col_size=2 output features).
    std::vector<double> w_flat = {1, 2, 3, 4};
    EncryptedValue W = encrypt_array({w_flat}, cc, kp, 0)[0];

    // x1 = [5, 6] -> expanded [5, 5, 6, 6]; x2 = [7, 8] -> expanded [7, 7, 8, 8]
    std::vector<std::vector<double>> expanded_x = {
        {5, 5, 6, 6},
        {7, 7, 8, 8},
    };
    std::vector<EncryptedValue> X = encrypt_array(expanded_x, cc, kp, 0);

    std::vector<EncryptedValue> enc_out = row_packing_forward(X, W, col_size);

    std::vector<std::vector<double>> result = decrypt_array(enc_out, cc, kp, col_size);

    // x1 @ W = [5*1 + 6*3, 5*2 + 6*4] = [23, 34]
    // x2 @ W = [7*1 + 8*3, 7*2 + 8*4] = [31, 46]
    std::vector<std::vector<double>> expected = {
        {23, 34},
        {31, 46},
    };

    for (size_t i = 0; i < expected.size(); i++)
    {
        for (size_t j = 0; j < expected[i].size(); j++)
        {
            EXPECT_NEAR(result[i][j], expected[i][j], 1e-6);
        }
    }
}
