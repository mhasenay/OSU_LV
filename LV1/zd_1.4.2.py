try:
    grade = float(input("Unesite ocjenu u intervalu [0.0, 0.1]: "))
    
    if grade < 0.0 or grade > 1.0:
        print("Uneseni broj nije u ispravnom intervalu!")
    elif grade >= 0.9: print("A")
    elif grade >= 0.8: print("B")
    elif grade >= 0.7: print("C")
    elif grade >= 0.6: print("D")
    else: print("F")

except:
    print("Pogreška pri unosu")