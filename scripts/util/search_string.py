import re
from scripts.retro_diffusion import rd

def searchString(string, *args):
    out = []

    # Iterate over the range of arguments, excluding the last one
    for x in range(len(args) - 1):
        # Perform a regex search in the string using the current and next argument as lookaround patterns
        # Append the matched substring to the output list
        try:
            out.append(
                re.search(f"(?<={{{args[x]}}}).*(?={{{args[x+1]}}})", string).group()
            )
        except:
            if args[x] not in string:
                rd.logger(f"\n[#ab333d]Could not find: {args[x]}")

    return out