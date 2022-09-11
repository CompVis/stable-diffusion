# with open("nsfw-ids.txt", 'r', encoding="utf8") as nsfwfile:
#     nsfw_list = list(nsfwfile)
import tqdm
# ##Read line
# current_saved_file_count = 0
# current_line_count = 0
# for line in nsfw_list:
#     print(line)
#     last3_line_raw = line[-4:]
#     last3_line = last3_line_raw.zfill(4)
#     print(last3_line_raw)
#     print(last3_line)

def file_len(filename):
    with open(filename) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def writetofile(input):
    f = open("files2download.txt", "a")
    f.write(input + "\n")
    f.close()

#converts nsfw-ids.txt entries to rsync readable file

with open("nsfw-ids.txt", 'r', encoding="utf8") as nsfwfile:
    nsfw_list = list(nsfwfile)
count = 0
linescount = file_len("nsfw-ids.txt")

##Read line
for line in nsfw_list:
    line = line.strip()
    # print(line)
    linefilled1 = line.zfill(4)
    linelast3 = linefilled1[-3:]
    linedirectory = linelast3.zfill(4)
    # print("line: " + ">>"+ line + "<<")
    # print("Linefilled1: " + linefilled1)
    # print("linelast3: " + linelast3)
    # print("linedirectory: " + linedirectory)
    directory = "original/" + linedirectory + "/" + line + ".jpg"
    # print(directory)
    # print(directory2)
    writetofile(directory)
    count = count + 1
    print(str(count) + "/" + str(linescount))
