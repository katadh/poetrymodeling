import os

for root, dirs, files in os.walk("../data/ohhlaraw"):
    for file in files:
        with open("../data/ohhlaraw/"+file) as f:
            lines = f.read().split("\n")

            included_lines = []
            for line in lines:
                if not line:
                    continue
                if line[0] == "[":
                    continue
                if len(line.split(":")[0]) < 15:
                    continue

                newline = "".join([char for char in line if char in "qwertyuiopasdfghjklzxcvbnm1234567890QWERTYUIOPASDFGHJKLZXCVBNM,.!?&-':;'\"() \n"])
                included_lines.append(newline)

            finalstring = "\n".join(included_lines)

            with open("../data/ohhla/"+file+".clean.txt", "w") as f:
                f.write(finalstring)
