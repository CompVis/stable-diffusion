import json
import os
import shutil
json_file_path = "metadata/posts000000000000.json" ##Name of the JSON file to use

os.mkdir("labeled_data")

def writefile(filename, text):
    f = open(filename, "w")
    f.write(text)
    print('Saved the following: ' + text)
    f.close()
#Converts tags to T2I-like prompts (blue_dress, 1girl -> A blue dress, one girl)
def convert_tags_to_humantxt(input):
    tars = input
    tars = tars.replace(' ', ', ')
    tars = tars.replace('_', ' ')
    tars = tars.replace('1girl', 'one girl')
    tars = tars.replace('2girls', 'two girls')
    tars = tars.replace('3girls', 'three girls')
    tars = tars.replace('4girls', 'four girls')
    tars = tars.replace('5girls', 'five girls')
    ##Implying it will ever be able to differentiate so many entities
    tars = tars.replace('6girls', 'six girls')

    #Almost forgot about boys tags... I wonder if theres also for other entities?
    tars = tars.replace('1boy', 'one girl')
    tars = tars.replace('2boys', 'two boys')
    tars = tars.replace('3boys', 'three boys')
    tars = tars.replace('4boys', 'four boys')
    tars = tars.replace('5boys', 'five boys')
    tars = tars.replace('6boys', 'six boys')
    print("FINAL TARS: " + tars)
    return tars

#Converts ratings to X content
def convert_rating_to_humanrating(input):
    if input == "e":
        return "explicit content"
    if input == "g":
        return "general content"
    if input == "q":
        return "questionable content"
    if input == "s":
        return "sensitive content"
        ##This will be the start of everything unethical

##Open the file
with open(json_file_path, 'r', encoding="utf8") as json_file:
    json_list = list(json_file)

##Read line
current_saved_file_count = 0
current_line_count = 0
for json_str in json_list:
    current_line_count = current_line_count + 1
    ##415627 last line of 00.json, ignore
    print("Current Line:" + str(current_line_count) + '/415627 | Current saved files count: ' + str(current_saved_file_count) )
    #here, result = line
    result = json.loads(json_str)

    try:
        img_id = str(result['id'])
    except Exception:
        img_id = "nan"
        print("failed to get img_id")
        continue

    try:
        tmp_img_id = img_id[-3:]
        img_id_last3 = tmp_img_id.zfill(3)
    except Exception:
        img_id_last3 = "nan"
        print("failed to get img_id_last3")
        continue
    
    try:
        img_tags = result['tag_string']
    except Exception:
        img_tags = "none"
        print("failed to get img_tags")
        continue

    try:
        img_ext = result['file_ext']
    except Exception:
        print("failed to get img_ext")
        continue

    try:
        img_rating = result['rating']
    except Exception:
        print("failed to get img_rating")
        continue

    file_path = "512px/0" + img_id_last3 + "/" + img_id + "." + img_ext
    if os.path.exists(file_path):
        shutil.copyfile(file_path, 'labeled_data/' + img_id + "." + img_ext)
        humanoid_tags = convert_tags_to_humantxt(img_tags)
        humanoid_rating = convert_rating_to_humanrating(img_rating)
        to_write = humanoid_tags + ', ' + humanoid_rating + ', uploaded on Danbooru'
        txt_name = "labeled_data/" + img_id + '.txt'
        writefile(txt_name, to_write)
        current_saved_file_count = current_saved_file_count + 1
        

