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
        #isto kao za ham, popravi:
        spam_counter =0
        if line.endswith("!"):
            #ovo popravi
            exclamation_counter = 0

file.close()

#Napisi print
#Napisi funkciju za racunanje averagea