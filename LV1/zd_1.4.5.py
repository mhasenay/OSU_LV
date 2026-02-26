def calculate_average(word_count, message_number):
    return float(word_count/message_number)
    

ham_words = 0
ham_counter = 0
spam_words = 0
spam_counter = 0
exclamation_counter = 0

file = open("LV1/SMSSpamCollection.txt")

for line in file:
    line = line.strip()
    if line.startswith("ham"):
        ham_counter += 1
        ham_words += len(line.split())-1
    elif line.startswith("spam"):
        spam_counter += 1
        spam_words += len(line.split())-1
        if line.endswith("!"):
            exclamation_counter += 1

file.close()

print(f"Prosječni broj ham: {calculate_average(ham_words, ham_counter):.2f}")
print(f"Prosječni broj spam: {calculate_average(spam_words, spam_counter):.2f}")
print(f"Broj spam poruka koje zavrsavaju s !: {exclamation_counter}")