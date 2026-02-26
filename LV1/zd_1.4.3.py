def calculate_average(numbers):
    if not numbers:
        return 0
    
    sum = 0
    for number in numbers:
        sum += number

    return float(sum/len(numbers))


numbers = []
while True:
    print("Ako želite završiti napišite: Done")
    value = input("Unesite broj: ")
    if value.lower() == "done":
        break

    try:
        number = float(value)
        numbers.append(number)
    except:
        print("Pogrešan unos")

if numbers:
    print(f"Broj unosa: {len(numbers)}")
    print(f"Srednja vrijednost: {calculate_average(numbers):.2f}")
    print(f"Min: {min(numbers)}")
    print(f"Max: {max(numbers)}")
    numbers.sort()
    print(f"Sortirana lista: {numbers}")
else:    
    print("Nema unesenih brojeva")