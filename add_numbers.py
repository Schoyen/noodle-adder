
with open("eight_bit_numbers.csv", "w") as f:
    for i in range(int(2 ** 8)):
        for j in range(int(2 ** 8)):

            res = (i + j) & 0xff
            a = bin(i)[2:].zfill(8)
            b = bin(j)[2:].zfill(8)
            res = bin(res)[2:].zfill(8)

            f.write("{0}, {1}, {2}\n".format(a, b, res))
