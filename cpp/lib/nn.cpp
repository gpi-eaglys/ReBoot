#include <omp.h>
#include <vector>

#include <openfhe.h>
#include <encrypted_value.h>

using namespace std;

vector<EncryptedValue> enc_l2_loss(vector<EncryptedValue> y_true, vector<EncryptedValue> y_pred, int n_cols)
{
    vector<EncryptedValue> loss(y_true.size());
#pragma omp parallel for
    for (int i = 0; i < y_true.size(); i++)
    {
        EncryptedValue temp = (y_pred[i] - y_true[i]).square();
        loss[i] = temp.sum(n_cols);
    }
    return loss;
}

vector<EncryptedValue> enc_l2_loss_grad(vector<EncryptedValue> y_true, vector<EncryptedValue> y_pred)
{
    vector<EncryptedValue> loss(y_true.size());
#pragma omp parallel for
    for (int i = 0; i < y_true.size(); i++)
    {
        loss[i] = y_pred[i] - y_true[i];
    }
    return loss;
}

vector<EncryptedValue> square_forward(vector<EncryptedValue> x)
{
    vector<EncryptedValue> out(x.size());
#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        out[i] = x[i].square();
    }
    return out;
}

vector<EncryptedValue> square_backward(vector<EncryptedValue> delta, vector<EncryptedValue> last_input)
{
    vector<EncryptedValue> new_delta(delta.size());
#pragma omp parallel for
    for (int i = 0; i < delta.size(); i++)
    {
        new_delta[i] = delta[i] * (last_input[i] * 2);
    }
    return new_delta;
}

vector<EncryptedValue> poly_relu_forward(vector<EncryptedValue> x)
{
    vector<EncryptedValue> out(x.size());
#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        out[i] = x[i].square() + x[i];
    }
    return out;
}

vector<EncryptedValue> poly_relu_backward(vector<EncryptedValue> delta, vector<EncryptedValue> last_input)
{
    vector<EncryptedValue> new_delta(delta.size());
#pragma omp parallel for
    for (int i = 0; i < delta.size(); i++)
    {
        new_delta[i] = delta[i] * (last_input[i] * 2 + 1);
    }
    return new_delta;
}

tuple<EncryptedValue, EncryptedValue> init_nesterov(EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr)
{
    EncryptedValue weight_update, new_velocity;

    if (weight_decay != 0.0)
    {
        // Add weight decay to the original gradient
        weight_gradient = weight_gradient + weight_decay * layer_weights;
    }

    // Initialize the velocity
    new_velocity = weight_gradient;

    // Compute the weight update with momentum
    weight_update = lr * new_velocity + (momentum * lr) * weight_gradient;

    return make_tuple(weight_update, new_velocity);
}

tuple<EncryptedValue, EncryptedValue> compute_nesterov(EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr, EncryptedValue velocity)
{
    EncryptedValue weight_update, new_velocity;

    if (weight_decay != 0)
    {
        // Add weight decay to the original gradient
        weight_gradient = weight_gradient + weight_decay * layer_weights;
    }

    // Compute the weight update with momentum
    weight_update = lr * weight_gradient + (momentum * lr) * weight_gradient + (momentum * momentum * lr) * velocity;

    // Update the velocity
    new_velocity = momentum * velocity + weight_gradient;

    return make_tuple(weight_update, new_velocity);
}