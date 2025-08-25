num = int(input("Enter a number:"))

while True:
    s = sum(int(d) for d in str(num))
    if(s<10):
        print("Single Digit Num:",s)
        break
    else:
        num = score

