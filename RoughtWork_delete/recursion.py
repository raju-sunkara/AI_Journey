def recFun(number):
    #number = 0
    if number == 0:
        return 1
    else:
        print (number)
        return float(number * recFun(number - 1))

def main():
    
    number = 6 #float(input("Enter a number for factorial : "))
    print (recFun(number))

main()