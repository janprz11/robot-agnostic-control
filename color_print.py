# -*- coding: ascii -*-

from typing import Any

# ansi color codes
BLUE = "\033[94m"
CYAN = "\033[96m"
GRAY = "\033[90m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
YELLOW = "\033[93m"


# uses a single print call for thread safety
def cprint(*args: Any, color: str = RESET, **kwargs: Any) -> None:
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    message = sep.join(str(a) for a in args)
    print(f"{color}{message}{RESET}", end=end, **kwargs)
