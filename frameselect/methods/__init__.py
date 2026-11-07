# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Frame selection methods — import all to trigger registration.

6 selectors ship registered: two static (``uniform``, ``medr``) and four
representation-based (``kmeans``, ``efs``, ``agfs``, ``qav``), each
implemented here without an upstream codebase to draw on (to the best of
our knowledge). Add a new selector by creating a module here and listing
it below.
"""

from . import uniform  # noqa: F401
from . import kmeans  # noqa: F401
from . import efs  # noqa: F401
from . import medr  # noqa: F401
from . import agfs  # noqa: F401
from . import qav  # noqa: F401
