# put your python code here
a = int(input())
b = int(input())
c = int(input())

def funkcja(input_value):
    x = int(input_value/2)
    if input_value%2!=0 :
        x = x+1
        return x
    else:
        return x

print(funkcja(a)+funkcja(b)+funkcja(c))