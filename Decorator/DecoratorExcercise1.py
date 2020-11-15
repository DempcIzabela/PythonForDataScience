#converts output to a string, and adds the pound sign to this string to represent the price of a product.
def price_string(func):
    def wrapper(arg):
        return "£" + str(func(arg))

    return wrapper

@price_string
def new_price(arg):
    return 0.9*arg
