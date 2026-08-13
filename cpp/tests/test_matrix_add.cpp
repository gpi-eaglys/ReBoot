#include <gtest/gtest.h>

#include <openfhe.h>
#include <encrypted_value.h>
#include <utils.h>

TEST(MatrixAdd, ElementwiseCkksAddition)
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
    EncryptedValue::setContext(cc);

    lbcrypto::KeyPair<lbcrypto::DCRTPoly> kp = cc->KeyGen();

    std::vector<std::vector<double>> a = {{1, 2}, {3, 4}};
    std::vector<std::vector<double>> b = {{5, 6}, {7, 8}};
    std::vector<std::vector<double>> expected = {{6, 8}, {10, 12}};

    std::vector<EncryptedValue> enc_a = encrypt_array(a, cc, kp, 0);
    std::vector<EncryptedValue> enc_b = encrypt_array(b, cc, kp, 0);

    std::vector<EncryptedValue> enc_sum(enc_a.size());
    for (size_t i = 0; i < enc_a.size(); i++)
    {
        enc_sum[i] = enc_a[i] + enc_b[i];
    }

    std::vector<std::vector<double>> result = decrypt_array(enc_sum, cc, kp, 2);

    for (size_t i = 0; i < expected.size(); i++)
    {
        for (size_t j = 0; j < expected[i].size(); j++)
        {
            EXPECT_NEAR(result[i][j], expected[i][j], 1e-6);
        }
    }
}
