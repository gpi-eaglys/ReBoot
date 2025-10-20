#pragma once

#include <vector>
#include <memory>
#include <map>
#include <openfhe.h>

using namespace lbcrypto;
using namespace std;

class EncryptedValue
{
public:
    /**
     * A class used to represent an encrypted CKKS value.
     * It is a wrapper for openfhe::Ciphertext used to override the arithmetic operators.
     *
     * Attributes:
     *     value (openfhe::Ciphertext): The encrypted value represented by the class.
     *     cc (openfhe::CryptoContext): The cryptocontext object from the OpenFHE library. Required to perform operations.
     */
    // Constructors
    EncryptedValue();
    EncryptedValue(Ciphertext<DCRTPoly> val);
    static EncryptedValue zero();

    // Getters
    static CryptoContext<DCRTPoly> getContext();
    static map<usint, EvalKey<DCRTPoly>> getSumRowsKeys();
    static map<usint, EvalKey<DCRTPoly>> getSumColsKeys();
    Ciphertext<DCRTPoly> getValue() const;

    // Setters
    static void setContext(CryptoContext<DCRTPoly> context);
    static void setSumRowsKeys(const map<usint, EvalKey<DCRTPoly>> &keys);
    static void setSumColsKeys(const map<usint, EvalKey<DCRTPoly>> &keys);
    void setValue(Ciphertext<DCRTPoly> val);

    static void generate_sum_keys(const KeyPair<DCRTPoly> &kp, int row_size);

    EncryptedValue operator+(const EncryptedValue &other) const;
    EncryptedValue operator+(float other) const;
    EncryptedValue operator+(Plaintext other) const;
    friend EncryptedValue operator+(float other, const EncryptedValue &e);
    friend EncryptedValue operator+(Plaintext other, const EncryptedValue &e);

    EncryptedValue operator-(const EncryptedValue &other) const;
    EncryptedValue operator-(float other) const;
    EncryptedValue operator-(Plaintext other) const;
    friend EncryptedValue operator-(float other, const EncryptedValue &e);
    friend EncryptedValue operator-(Plaintext other, const EncryptedValue &e);

    EncryptedValue operator*(const EncryptedValue &other) const;
    EncryptedValue operator*(float other) const;
    EncryptedValue operator*(Plaintext other) const;
    friend EncryptedValue operator*(float other, const EncryptedValue &e);
    friend EncryptedValue operator*(Plaintext other, const EncryptedValue &e);

    EncryptedValue square() const;
    EncryptedValue evalPoly(const vector<double> coefficients) const;
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
    Ciphertext<DCRTPoly> value;
    shared_ptr<vector<DCRTPoly>> precomp;
    static CryptoContext<DCRTPoly> cc;
    static map<usint, EvalKey<DCRTPoly>> sumRowsKeys;
    static map<usint, EvalKey<DCRTPoly>> sumColsKeys;
    static KeyPair<DCRTPoly> kp;
};
