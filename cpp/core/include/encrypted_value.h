#pragma once

#include <vector>
#include <memory>
#include <map>
#include <openfhe.h>


class EncryptedValue
{
public:
    /**
     * A class used to represent an encrypted CKKS value.
     * It is a wrapper for lbcrypto::Ciphertext used to override the arithmetic operators.
     *
     * Attributes:
     *     value (lbcrypto::Ciphertext): The encrypted value represented by the class.
     *     cc (lbcrypto::CryptoContext): The cryptocontext object from the OpenFHE library. Required to perform operations.
     */
    // Constructors
    EncryptedValue();
    EncryptedValue(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> val);
    static EncryptedValue zero();

    // Getters
    static lbcrypto::CryptoContext<lbcrypto::DCRTPoly> getContext();
    static std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> getSumRowsKeys();
    static std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> getSumColsKeys();
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> getValue() const;

    // Setters
    static void setContext(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context);
    static void setSumRowsKeys(const std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> &keys);
    static void setSumColsKeys(const std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> &keys);
    void setValue(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> val);

    static void generate_sum_keys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &kp, int row_size);

    EncryptedValue operator+(const EncryptedValue &other) const;
    EncryptedValue operator+(float other) const;
    EncryptedValue operator+(lbcrypto::Plaintext other) const;
    friend EncryptedValue operator+(float other, const EncryptedValue &e);
    friend EncryptedValue operator+(lbcrypto::Plaintext other, const EncryptedValue &e);

    EncryptedValue operator-(const EncryptedValue &other) const;
    EncryptedValue operator-(float other) const;
    EncryptedValue operator-(lbcrypto::Plaintext other) const;
    friend EncryptedValue operator-(float other, const EncryptedValue &e);
    friend EncryptedValue operator-(lbcrypto::Plaintext other, const EncryptedValue &e);

    EncryptedValue operator*(const EncryptedValue &other) const;
    EncryptedValue operator*(float other) const;
    EncryptedValue operator*(lbcrypto::Plaintext other) const;
    friend EncryptedValue operator*(float other, const EncryptedValue &e);
    friend EncryptedValue operator*(lbcrypto::Plaintext other, const EncryptedValue &e);

    EncryptedValue square() const;
    EncryptedValue evalPoly(const std::vector<double> coefficients) const;
    EncryptedValue bootstrap(int num_iterations = 1, int precision = 0) const;
    EncryptedValue rotate(int index) const;

    void fastRotatePrecomp();
    EncryptedValue fastRotate(int index) const;
    int getLevel() const;
    EncryptedValue sum(int n_cols) const;

    EncryptedValue sumRows(int row_size) const;
    EncryptedValue sumCols(int row_size) const;
    size_t get_memory_size();

private:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> value;
    
    // digit decomposition of the ciphertext, produced by cc->EvalFastRotationPrecompute(value)
    std::shared_ptr<std::vector<lbcrypto::DCRTPoly>> precomp;
    
    static lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
    static std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> sumRowsKeys;
    static std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> sumColsKeys;
    static lbcrypto::KeyPair<lbcrypto::DCRTPoly> kp;
};

using CipherTexts = std::vector<EncryptedValue>;
