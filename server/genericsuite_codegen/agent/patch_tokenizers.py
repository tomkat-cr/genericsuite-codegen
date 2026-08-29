"""
Patch to avoid this warning:
huggingface/tokenizers: The current process just got forked, after parallelism
has already been used. Disabling parallelism to avoid deadlocks...
"""

import os


def patch_tokenizers():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
