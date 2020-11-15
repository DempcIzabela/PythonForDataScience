def tagged(func):
    def wrapper(args):
        return ("<title>"+ func(args) + "</title>")
    return wrapper

@tagged
def from_input(inp):
    string = inp.strip()
    return string