import numpy as np
from numpy.typing import NDArray

import reboot_py

Array = NDArray[np.float32 | reboot_py.EncryptedValue]
Parameter = NDArray[np.float32 | reboot_py.EncryptedValue] | None
