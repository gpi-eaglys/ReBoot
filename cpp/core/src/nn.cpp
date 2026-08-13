#include <omp.h>
#include <vector>

#include <openfhe.h>
#include <encrypted_value.h>

std::vector<EncryptedValue> enc_l2_loss(std::vector<EncryptedValue> y_true, std::vector<EncryptedValue> y_pred, int n_cols)
{
    std::vector<EncryptedValue> loss(y_true.size());
#pragma omp parallel for
    for (size_t i = 0; i < y_true.size(); i++)
    {
        EncryptedValue temp = (y_pred[i] - y_true[i]).square();
        loss[i] = temp.sum(n_cols);
    }
    return loss;
}

std::vector<EncryptedValue> enc_l2_loss_grad(std::vector<EncryptedValue> y_true, std::vector<EncryptedValue> y_pred)
{
    std::vector<EncryptedValue> loss(y_true.size());
#pragma omp parallel for
    for (size_t i = 0; i < y_true.size(); i++)
    {
        loss[i] = y_pred[i] - y_true[i];
    }
    return loss;
}

std::vector<EncryptedValue> square_forward(std::vector<EncryptedValue> x)
{
    std::vector<EncryptedValue> out(x.size());
#pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++)
    {
        out[i] = x[i].square();
    }
    return out;
}

std::vector<EncryptedValue> square_backward(std::vector<EncryptedValue> delta, std::vector<EncryptedValue> last_input)
{
    std::vector<EncryptedValue> new_delta(delta.size());
#pragma omp parallel for
    for (size_t i = 0; i < delta.size(); i++)
    {
        new_delta[i] = delta[i] * (last_input[i] * 2);
    }
    return new_delta;
}

std::vector<EncryptedValue> poly_relu_forward(std::vector<EncryptedValue> x)
{
    std::vector<EncryptedValue> out(x.size());
#pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++)
    {
        out[i] = x[i].square() + x[i];
    }
    return out;
}

std::vector<EncryptedValue> poly_relu_backward(std::vector<EncryptedValue> delta, std::vector<EncryptedValue> last_input)
{
    std::vector<EncryptedValue> new_delta(delta.size());
#pragma omp parallel for
    for (size_t i = 0; i < delta.size(); i++)
    {
        new_delta[i] = delta[i] * (last_input[i] * 2 + 1);
    }
    return new_delta;
}

std::tuple<EncryptedValue, EncryptedValue> init_nesterov(EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr)
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

    return std::make_tuple(weight_update, new_velocity);
}

std::tuple<EncryptedValue, EncryptedValue> compute_nesterov(EncryptedValue weight_gradient, float momentum, float weight_decay, EncryptedValue layer_weights, float lr, EncryptedValue velocity)
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

    return std::make_tuple(weight_update, new_velocity);
}