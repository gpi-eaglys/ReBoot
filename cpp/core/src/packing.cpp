#include <omp.h>
#include <vector>
#include <map>
#include <algorithm>

#include <openfhe.h>
#include <encrypted_value.h>

#pragma omp declare reduction(                                         \
        EncryptedValueAdd:EncryptedValue : omp_out = omp_out + omp_in) \
    initializer(omp_priv = EncryptedValue::zero())

std::vector<EncryptedValue> row_packing_forward(std::vector<EncryptedValue> X, EncryptedValue W, int row_size)
{
    std::vector<EncryptedValue> out(X.size());

#pragma omp parallel for
    for (size_t i = 0; i < X.size(); i++)
    {
        out[i] = X[i] * W;
        out[i] = out[i].sumRows(row_size);
    }

    return out;
}

std::vector<EncryptedValue> row_packing_backward(std::vector<EncryptedValue> delta, EncryptedValue W, int row_size)
{
    std::vector<EncryptedValue> new_delta(delta.size());

#pragma omp parallel for
    for (size_t i = 0; i < delta.size(); i++)
    {
        new_delta[i] = delta[i] * W;
        new_delta[i] = new_delta[i].sumCols(row_size);
    }

    return new_delta;
}

std::vector<EncryptedValue> col_packing_forward(std::vector<EncryptedValue> X, EncryptedValue W, int row_size)
{
    std::vector<EncryptedValue> out(X.size());

#pragma omp parallel for
    for (size_t i = 0; i < X.size(); i++)
    {
        out[i] = X[i] * W;
        out[i] = out[i].sumCols(row_size);
    }

    return out;
}

std::vector<EncryptedValue> col_packing_backward(std::vector<EncryptedValue> delta, EncryptedValue W, int row_size)
{
    std::vector<EncryptedValue> new_delta(delta.size());

#pragma omp parallel for
    for (size_t i = 0; i < delta.size(); i++)
    {
        new_delta[i] = delta[i] * W;
        new_delta[i] = new_delta[i].sumRows(row_size);
    }

    return new_delta;
}

EncryptedValue packing_weight_update(std::vector<EncryptedValue> X, std::vector<EncryptedValue> delta)
{
    EncryptedValue out = EncryptedValue::zero();

#pragma omp parallel for reduction(EncryptedValueAdd : out)
    for (size_t i = 0; i < X.size(); i++)
    {
        out = out + X[i] * delta[i];
    }

    return out;
}
