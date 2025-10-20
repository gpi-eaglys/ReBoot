#include <omp.h>
#include <vector>
#include <map>
#include <algorithm>

#include <openfhe.h>
#include <encrypted_value.h>

using namespace lbcrypto;
using namespace std;

#pragma omp declare reduction(                                         \
        EncryptedValueAdd:EncryptedValue : omp_out = omp_out + omp_in) \
    initializer(omp_priv = EncryptedValue::zero())

vector<EncryptedValue> row_packing_forward(vector<EncryptedValue> X, EncryptedValue W, int row_size)
{
    vector<EncryptedValue> out(X.size());

#pragma omp parallel for
    for (int i = 0; i < X.size(); i++)
    {
        out[i] = X[i] * W;
        out[i] = out[i].sumRows(row_size);
    }

    return out;
}

vector<EncryptedValue> row_packing_backward(vector<EncryptedValue> delta, EncryptedValue W, int row_size)
{
    vector<EncryptedValue> new_delta(delta.size());

#pragma omp parallel for
    for (int i = 0; i < delta.size(); i++)
    {
        new_delta[i] = delta[i] * W;
        new_delta[i] = new_delta[i].sumCols(row_size);
    }

    return new_delta;
}

vector<EncryptedValue> col_packing_forward(vector<EncryptedValue> X, EncryptedValue W, int row_size)
{
    vector<EncryptedValue> out(X.size());

#pragma omp parallel for
    for (int i = 0; i < X.size(); i++)
    {
        out[i] = X[i] * W;
        out[i] = out[i].sumCols(row_size);
    }

    return out;
}

vector<EncryptedValue> col_packing_backward(vector<EncryptedValue> delta, EncryptedValue W, int row_size)
{
    vector<EncryptedValue> new_delta(delta.size());

#pragma omp parallel for
    for (int i = 0; i < delta.size(); i++)
    {
        new_delta[i] = delta[i] * W;
        new_delta[i] = new_delta[i].sumRows(row_size);
    }

    return new_delta;
}

EncryptedValue packing_weight_update(vector<EncryptedValue> X, vector<EncryptedValue> delta)
{
    EncryptedValue out = EncryptedValue::zero();

#pragma omp parallel for reduction(EncryptedValueAdd : out)
    for (int i = 0; i < X.size(); i++)
    {
        out = out + X[i] * delta[i];
    }

    return out;
}
