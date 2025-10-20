from __future__ import annotations

import numpy as np
import openfhe as fhe
import yaml
from numpy.typing import NDArray

import reboot_cpp

PT = fhe.Plaintext
CT = fhe.Ciphertext
CC = fhe.CryptoContext
KP = fhe.KeyPair


class CryptoContext:
    """
    Class that manages the CKKS homomorphic encryption scheme.
    It provides methods to encode, encrypt, and decrypt NumPy arrays containing EncryptedValue objects.
    Both c2p and c2c operations are supported.
    Matrices are encoded diagonally.

    Attributes:
        cc (ofhe.CryptoContext): Cryptocontext object from the OpenFHE library. It holds encryption parameters.
        kp (ofhe.KeyPair): Keypair object from the OpenFHE library. It holds the public and secret keys.
        is_initialized (bool): A flag to check whether the context has been initialized.
        bs_levels (int): The number of levels required for bootstrapping.
        bs_num_iterations (int): The number of iterations for bootstrapping.
        bs_precision (int): The precision for iterative bootstrapping.
        num_slots (int): The number of slots in the ciphertexts.
    """

    def __init__(
        self, config_path: str, model: "Model" | None = None, additional_depth: int = 0
    ) -> None:
        """
        Initialize the CryptoContext object with the parameters from the configuration file.
        Will raise an exception if the configuration is unsupported or incorrect.
        If a Model is provided, the multiplicative depth and the number of slots will be set based on the model's architecture.

        Args:
            config_path (str): The path to the configuration file.
            model(Model): The model to recommend the parameters for. Defaults to None.
            additional_depth (int): Additional depth to use. Defaults to 0.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        params = config["params"]
        bootstrap_params = config["bootstrap_params"]
        features = config["features"]

        if model is not None:
            depth, slots, row_size, col_size = self.get_recommended_parameters(
                model, additional_depth
            )
            params["mult_depth"] = depth
            params["num_slots"] = slots

        if fhe.get_native_int() == 128:
            print("Using high-precision 128 bit computations.")
            scaling_mod_size = params["128_scaling_mod_size"]
            first_mod = params["128_first_mod"]
            rescale_method = getattr(fhe, params["128_rescale_method"])
        else:
            scaling_mod_size = params["64_scaling_mod_size"]
            first_mod = params["64_first_mod"]
            rescale_method = getattr(fhe, params["64_rescale_method"])

        parameters = fhe.CCParamsCKKSRNS()
        mult_depth = params["mult_depth"]

        # Setup bootstrapping parameters if required
        if bootstrap_params["run_bootstrap"]:
            sk_dist = getattr(fhe, bootstrap_params["secret_key_dist"])
            level_budget = bootstrap_params["level_budget"]
            num_iterations = bootstrap_params["num_iterations"]

            max_budget = max([1, np.ceil(np.log2(params["num_slots"]))])
            if level_budget[0] > max_budget or level_budget[1] > max_budget:
                raise ValueError("Level budget too high for the number of slots.")

            self.bs_levels = fhe.FHECKKSRNS.GetBootstrapDepth(level_budget, sk_dist)
            mult_depth = mult_depth + self.bs_levels + num_iterations
            parameters.SetSecretKeyDist(sk_dist)
            self.bs_iterations = num_iterations
            self.bs_precision = bootstrap_params["precision"]

        # Setup the scheme parameters
        parameters.SetMultiplicativeDepth(mult_depth)
        parameters.SetFirstModSize(first_mod)
        parameters.SetScalingModSize(scaling_mod_size)
        parameters.SetScalingTechnique(rescale_method)
        # Batch size is not set as full-packing is required

        # Set the security level
        if params["ring_dim"] is not None:
            parameters.SetRingDim(params["ring_dim"])
        match params["security_level"]:
            case "HEStd_NotSet":
                parameters.SetSecurityLevel(fhe.SecurityLevel.HEStd_NotSet)
            case "HEStd_128_classic":
                parameters.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
            case "HEStd_192_classic":
                parameters.SetSecurityLevel(fhe.SecurityLevel.HEStd_192_classic)
            case "HEStd_256_classic":
                parameters.SetSecurityLevel(fhe.SecurityLevel.HEStd_256_classic)
            case _:
                raise ValueError("Invalid security level")

        # Enable the required features
        cc: CC = fhe.GenCryptoContext(parameters)
        if features["pke"]:
            cc.Enable(fhe.PKESchemeFeature.PKE)
        if features["keyswitch"]:
            cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
        if features["leveledshe"]:
            cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
        if features["advancedshe"]:
            cc.Enable(fhe.PKESchemeFeature.ADVANCEDSHE)
        if features["fhe"]:
            cc.Enable(fhe.PKESchemeFeature.FHE)

        # Generate the keys
        key_pair: KP = cc.KeyGen()
        cc.EvalMultKeyGen(key_pair.secretKey)
        cc.EvalSumKeyGen(key_pair.secretKey)

        # Pre-computations for bootstrapping
        if bootstrap_params["run_bootstrap"]:
            slots = cc.GetRingDimension() // 2
            cc.EvalBootstrapSetup(level_budget, slots=slots)
            cc.EvalBootstrapKeyGen(key_pair.secretKey, slots)

        # Log the parameters
        print("CKKS parameters:")
        print(f"  - Security level: {parameters.GetSecurityLevel()}")
        print(f"  - Ring dimension: {cc.GetRingDimension()}")
        print(f"  - Multiplicative depth: {parameters.GetMultiplicativeDepth()}")
        print(f"  - Rescale method: {parameters.GetScalingTechnique()}")
        print(f"  - Number of slots: {cc.GetRingDimension() // 2}")
        print(f"  - First modulus size: {parameters.GetFirstModSize()} bits")
        print(f"  - Scaling modulus size: {parameters.GetScalingModSize()} bits")

        if bootstrap_params["run_bootstrap"]:
            print("\nBootstrapping parameters:")
            print(f"  - Bootstrapping enabled: {bootstrap_params['run_bootstrap']}")
            print(f"  - Bootstrapping cost: {self.bs_levels}")
            print(f"  - Level budget: {level_budget}")
            print(f"  - Number of iterations: {num_iterations}")
            print(f"  - Secret key distribution: {sk_dist}")

        # Set internal attributes and initialize the EncryptedValue class
        self.cc = cc
        self.kp = key_pair
        self.is_initialized = True
        self.num_slots = cc.GetRingDimension() // 2
        self.cyclotomic_order = cc.GetCyclotomicOrder()
        
        self.row_size = row_size
        self.col_size = col_size
        
        # Adjust row_size according to the ring dimension
        ratio =  self.num_slots // (self.row_size * self.col_size) 
        self.row_size = self.row_size * ratio
        self.valid_slots = self.row_size * self.col_size

        if model is not None:
            print("\nNetwork parameters:")
            print(f"  - Multiplicative depth: {depth}")
            print(f"  - Valid slots: {self.valid_slots}")
            print(f"  - Row size: {self.row_size}")
            print(f"  - Col size: {self.col_size}\n")

        reboot_cpp.EncryptedValue.set_context(cc)
        reboot_cpp.EncryptedValue.generate_sum_keys(self.kp, self.col_size)

    def get_recommended_parameters(
        self, model: "Model", additional_depth: int = 0
    ) -> tuple[int, int, int, int]:
        """
        Given a non-encrypted model, print the recommended multiplicative depth and number of slots for the CKKS scheme in order to train the model.
        Additionally, compute the required `row_size` and `col_size` for full matrix packing.

        Args:
            model (Model): The model to recommend the parameters for.
            additional_depth (int): Additional depth to use.

        Returns:
            tuple[int, int]: The recommended multiplicative depth and number of slots.
        """
        # The multiplicative depth is given by the number of layers in the network
        network_depths = [3, 8, 11, 13]
        depth = network_depths[model.num_layers - 1] + additional_depth

        # The number of slots is given by row_size x col_size
        layers = model.get_layers_with_parameters()
        rows_layers_nodes, cols_layers_nodes = [], []
        row_packing = True
        offset = 0
        for _ in range(0, len(model.blocks)):
            # Blocks
            if row_packing:
                rows_layers_nodes.append(layers[offset].weights.shape[1])
                cols_layers_nodes.append(layers[offset + 1].weights.shape[1])
            else:
                cols_layers_nodes.append(layers[offset].weights.shape[1])
                rows_layers_nodes.append(layers[offset + 1].weights.shape[1])
            offset = offset + 2
            row_packing = not row_packing
        if model.num_layers > 1:
            # Second to last layer
            if row_packing:
                rows_layers_nodes.append(layers[-2].weights.shape[1])
            else:
                cols_layers_nodes.append(layers[-2].weights.shape[1])
        # Last layer
        if row_packing:
            rows_layers_nodes.append(layers[-1].weights.shape[1])
        else:
            cols_layers_nodes.append(layers[-1].weights.shape[1])

        row_size = self.next_power_of_two(
            max(rows_layers_nodes + [layers[0].weights.shape[0]])
        )
        col_size = self.next_power_of_two(
            max(cols_layers_nodes + [layers[0].weights.shape[1]])
        )
        num_slots = row_size * col_size

        return depth, num_slots, row_size, col_size

    def encode_array(self, array: NDArray[np.float32], level: int = 0) -> NDArray[PT]:
        """
        Encode a NumPy array of floats into an array of Plaintext objects from the OpenFHE library.
        It behaves as self.encrypt_array(), but does not encrypt the data.

        Args:
            array (NDArray[float]): The input array to encode.
            level (int): The level at which encode the ciphertext. Defaults to 0.

        Returns:
            NDArray[PT]: The encoded array.
        """
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        if array.ndim == 1:
            array = array.reshape(1, -1)

        return reboot_cpp.encode_array(array, self.cc, level)

    def encrypt_array(
        self, array: NDArray[np.float32], level: int = 0
    ) -> NDArray[reboot_cpp.EncryptedValue]:
        """
        Encrypt a NumPy array of floats into an array of EncryptedValue objects.

        Args:
            array (NDArray[float]): The input array to encrypt.
            level (int): The level at which encode the ciphertext. Defaults to 0.

        Returns:
            NDArray[EncryptedValue]: The encrypted array.
        """
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        if array.ndim == 1:
            array = array.reshape(1, -1)

        return reboot_cpp.encrypt_array(array, self.cc, self.kp, level)

    def decrypt_array(
        self,
        array: NDArray[reboot_cpp.EncryptedValue] | reboot_cpp.EncryptedValue,
        num_slots: int,
    ) -> NDArray[np.float32]:
        """
        Decrypt a NumPy array of EncryptedValue objects into an array of floats.

        Args:
            array (NDArray[EncryptedValue] | EncryptedValue): The input array to decrypt.
            num_slots (int): The number of slots in the ciphertext.

        Returns:
            NDArray[float32]: The decrypted array.
        """
        if isinstance(array, reboot_cpp.EncryptedValue):
            array = np.array([array])
        return reboot_cpp.decrypt_array(array, self.cc, self.kp, num_slots).astype(
            np.float32
        )

    def get_precision(self, array: NDArray[reboot_cpp.EncryptedValue]) -> str:
        """
        Return an estimate of the precision of the encrypted data based on the infinity norm of the plaintext data. This involves decryption as precision cannot be computed on ciphertext data. Warning: precision estimates are not always accurate.

        Args:
            array (NDArray[EncryptedValue]): The input array of encrypted values.

        Returns:
            str: A string with the mean, max, and min precision of the array.
        """

        def __get_precision(value: reboot_cpp.EncryptedValue) -> float:
            plain_value: PT = self.cc.Decrypt(self.kp.secretKey, value.get_value())
            return plain_value.GetLogPrecision()

        get_precision_f = np.frompyfunc(__get_precision, nin=1, nout=1)
        precisions = get_precision_f(array)
        return f"({precisions.mean():5.2f}|{precisions.max():5.2f}|{precisions.min():5.2f})"

    def get_level(self, array: NDArray[reboot_cpp.EncryptedValue]) -> int:
        """
        Return the level of the ciphertexts in the array.

        Args:
            NDArray[EncryptedValue]: The input array of encrypted values.

        Returns:
            int: The level of the ciphertext.
        """
        return array.item(0).get_level()

    def is_encrypted(self, array: NDArray[reboot_cpp.EncryptedValue]) -> bool:
        """
        Check if the array contains encrypted data.

        Args:
            array (NDArray[EncryptedValue]): The input array of encrypted values.

        Returns:
            bool: True if the array contains encrypted data, False otherwise.
        """
        return isinstance(array.item(0), reboot_cpp.EncryptedValue)

    def repeat_and_encrypt(self, x, debug: bool = False):
        x = np.pad(x, ((0, 0), (0, self.col_size - x.shape[1])))
        repeated_x = np.tile(x, self.row_size)
        if debug:
            print(f"Repeated length: ({len(repeated_x)})")
        encrypted_x = reboot_cpp.encrypt_array(repeated_x, self.cc, self.kp)
        return encrypted_x

    def expand_and_encrypt(self, x, debug: bool = False):
        x = np.pad(x, ((0, 0), (0, self.row_size - x.shape[1])))
        expanded_x = np.repeat(x, self.col_size, axis=1)
        if debug:
            print(f"Expanded length: ({len(expanded_x)})")
        encrypted_x = reboot_cpp.encrypt_array(expanded_x, self.cc, self.kp)
        return encrypted_x

    def next_power_of_two(self, x):
        return 1 << (x - 1).bit_length()
