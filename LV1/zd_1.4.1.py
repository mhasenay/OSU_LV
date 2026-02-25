def total_euro(hours, rate):
    return hours * rate

hours = int(input("Radni sati: "))
rate = float(input("eura/h: "))
total = total_euro(hours, rate)
print(f"Ukupno: {total:.2f} eura")