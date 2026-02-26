word_count = {}

fhand = open("LV1/song.txt")

for line in fhand:
    words = line.split()
    for word in words:
        word = word.rstrip(',.!?').lower()
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

fhand.close()

counter = 0
for word, count in word_count.items():
    if count == 1:
        print(word)
        counter += 1
print(f"Broj riječi koje se ponavljaju jednom: {counter}")