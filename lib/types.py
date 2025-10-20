import numpy as np
from numpy.typing import NDArray

import reboot_cpp

Array = NDArray[np.float32 | reboot_cpp.EncryptedValue]
Parameter = NDArray[np.float32 | reboot_cpp.EncryptedValue] | None
