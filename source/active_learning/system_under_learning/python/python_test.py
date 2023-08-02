x = 0

print("Printing from within the script")

def test_no_arg():
    print("Inside test no arg")
    return 1

def test_with_global():
    global x

    x += 3
    print("Inside test no arg")
    return x

def test_with_global_and_arg(i: int):
    global x

    x += i
    print("Inside test with arg")
    return x

if __name__ == "__main__":
    print(test_no_arg())
    print(test_with_global())
    print(test_with_global())
    print(test_with_global_and_arg(4))
    print(test_with_global_and_arg(5))
