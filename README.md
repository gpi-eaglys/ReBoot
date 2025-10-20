# ReBoot

## Requirements

ReBoot was developed and tested with:

- **Python:** 3.10.12
- **OpenFHE:** 1.2.1
- **OpenFHE-Python:** 0.8.9

Use the provided `.devcontainer` files to spin up a VSCode DevContainer with the library correctly installed.

It will install the OpenFHE and OpenFHE-Python libraries, along with the necessary dependencies to run ReBoot.

## Multiplicative depth

This table summarizes the multiplicative depth required to perform a training step with different MLP architectures.
The worst-case multiplicative depth is represented.

| Architecture     | Forward | Backward | Weights |  Additional depth per step |
|------------------|:-------:|:--------:|:-------:|:--------------------------:|
| No hidden layers | 1       | 1        | 3       | 3                          |
| 1 hidden layer   | 3       | 5        | 7       | 7                          |
| 2 hidden layers  | 5       | 7        | 9       | 7                          |
| 3 hidden layers  | 7       | 9        | 11      | 7                          |

**Remark:** the use of **weight decay** or **momentum** in the optimizer does not increase the depth.
