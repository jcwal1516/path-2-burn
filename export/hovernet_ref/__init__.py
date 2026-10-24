"""HoVer-Net's network, vendored from vqdang/hover_net (models/hovernet, MIT License).

net_desc.py is unchanged. net_utils.py drops its unused `config` import, and utils.py keeps
only the two crop helpers the network uses (the losses need matplotlib and are training-only).
See LICENSE.
"""

from .net_desc import HoVerNet

__all__ = ["HoVerNet"]
