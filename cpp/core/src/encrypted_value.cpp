#include <vector>
#include <memory>
#include <map>
#include <openfhe.h>
#include <encrypted_value.h>

// Static members initialization
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> EncryptedValue::cc = nullptr;
std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> EncryptedValue::sumRowsKeys;
std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> EncryptedValue::sumColsKeys;
lbcrypto::KeyPair<lbcrypto::DCRTPoly> EncryptedValue::kp;

// Constructors
EncryptedValue::EncryptedValue() {}
EncryptedValue::EncryptedValue(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> val) : value(val) {}

EncryptedValue EncryptedValue::zero()
{
    std::vector<std::complex<double>> zeros = {0.0};
    lbcrypto::Plaintext pt = cc->MakeCKKSPackedPlaintext(zeros);
    return EncryptedValue(cc->Encrypt(kp.publicKey, pt));
}

// Getters
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> EncryptedValue::getContext()
{
    return cc;
}

std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> EncryptedValue::getSumRowsKeys()
{
    return sumRowsKeys;
}

std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> EncryptedValue::getSumColsKeys()
{
    return sumColsKeys;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EncryptedValue::getValue() const
{
    return this->value;
}

// Setters
void EncryptedValue::setContext(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context)
{
    cc = context;
}

void EncryptedValue::setSumRowsKeys(const std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> &keys)
{
    sumRowsKeys = keys;
}

void EncryptedValue::setSumColsKeys(const std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> &keys)
{
    sumColsKeys = keys;
}

void EncryptedValue::setValue(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> val)
{
    this->value = val;
}

void EncryptedValue::generate_sum_keys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &kp, int row_size)
{
    EncryptedValue::sumRowsKeys = *cc->EvalSumRowsKeyGen(kp.secretKey, nullptr, row_size);
    EncryptedValue::sumColsKeys = *cc->EvalSumColsKeyGen(kp.secretKey);
    EncryptedValue::kp = kp;
}

// Operator +
EncryptedValue EncryptedValue::operator+(const EncryptedValue &other) const
{
    return EncryptedValue(cc->EvalAdd(this->value, other.value));
}

EncryptedValue EncryptedValue::operator+(float other) const
{
    return EncryptedValue(cc->EvalAdd(this->value, other));
}

EncryptedValue EncryptedValue::operator+(lbcrypto::Plaintext other) const
{
    return EncryptedValue(cc->EvalAdd(this->value, other));
}

EncryptedValue operator+(float other, const EncryptedValue &e)
{
    return EncryptedValue(e.cc->EvalAdd(other, e.value));
}

EncryptedValue operator+(lbcrypto::Plaintext other, const EncryptedValue &e)
{
    return EncryptedValue(e.cc->EvalAdd(other, e.value));
}

// Operator -
EncryptedValue EncryptedValue::operator-(const EncryptedValue &other) const
{
    return EncryptedValue(cc->EvalSub(this->value, other.value));
}

EncryptedValue EncryptedValue::operator-(float other) const
{
    return EncryptedValue(cc->EvalSub(this->value, other));
}

EncryptedValue EncryptedValue::operator-(lbcrypto::Plaintext other) const
{
    return EncryptedValue(cc->EvalSub(this->value, other));
}

EncryptedValue operator-(float other, const EncryptedValue &e)
{
    return EncryptedValue(e.cc->EvalSub(other, e.value));
}

EncryptedValue operator-(lbcrypto::Plaintext other, const EncryptedValue &e)
{
    return EncryptedValue(e.cc->EvalSub(other, e.value));
}

// Operator *
EncryptedValue EncryptedValue::operator*(const EncryptedValue &other) const
{
    return EncryptedValue(cc->EvalMult(this->value, other.value));
}

EncryptedValue EncryptedValue::operator*(float other) const
{
    return EncryptedValue(cc->EvalMult(this->value, other));
}

EncryptedValue EncryptedValue::operator*(lbcrypto::Plaintext other) const
{
    return EncryptedValue(cc->EvalMult(this->value, other));
}

EncryptedValue operator*(float other, const EncryptedValue &e)
{
    return EncryptedValue(e.cc->EvalMult(other, e.value));
}

EncryptedValue operator*(lbcrypto::Plaintext other, const EncryptedValue &e)
{
    return EncryptedValue(e.cc->EvalMult(other, e.value));
}

// Other methods
EncryptedValue EncryptedValue::square() const
{
    return EncryptedValue(cc->EvalSquare(this->value));
}

EncryptedValue EncryptedValue::evalPoly(const std::vector<double> coefficients) const
{
    return EncryptedValue(cc->EvalPoly(this->value, coefficients));
}

EncryptedValue EncryptedValue::bootstrap(int num_iterations, int precision) const
{
    return EncryptedValue(cc->EvalBootstrap(this->value, num_iterations, precision));
}

EncryptedValue EncryptedValue::rotate(int index) const
{
    return EncryptedValue(cc->EvalRotate(this->value, index));
}

void EncryptedValue::fastRotatePrecomp()
{
    this->precomp = cc->EvalFastRotationPrecompute(this->value);
}

EncryptedValue EncryptedValue::fastRotate(int index) const
{
    return EncryptedValue(cc->EvalFastRotation(this->value, index, cc->GetCyclotomicOrder(), this->precomp));
}

int EncryptedValue::getLevel() const
{
    return this->value->GetLevel();
}

EncryptedValue EncryptedValue::sum(int n_cols) const
{
    return EncryptedValue(cc->EvalSum(this->value, n_cols));
}

EncryptedValue EncryptedValue::sumRows(int row_size) const
{
    return EncryptedValue(cc->EvalSumRows(this->value, row_size, this->sumRowsKeys));
}

EncryptedValue EncryptedValue::sumCols(int row_size) const
{
    return EncryptedValue(cc->EvalSumCols(this->value, row_size, this->sumColsKeys));
}

size_t EncryptedValue::get_memory_size()
{
    size_t size = 0;
    for (auto &element : this->value->GetElements())
    {
        for (auto &subelements : element.GetAllElements())
        {
            size += subelements.GetLength() * sizeof(subelements[0]);
        }
    }
    return size;
}