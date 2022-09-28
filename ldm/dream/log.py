"""
Functions for better format logging
    write_log -- logs the name of the output image, prompt, and prompt args to the terminal and different types of file
        1 write_log_message -- Writes a message to the console
        2 write_log_files -- Writes a message to files
        2.1 write_log_default -- File in plain text
        2.2 write_log_txt -- File in txt format
        2.3 write_log_markdown -- File in markdown format
"""

import os


def write_log(results, log_path, file_types, output_cntr):
    """
    logs the name of the output image, prompt, and prompt args to the terminal and files
    """
    output_cntr = write_log_message(results, output_cntr)
    write_log_files(results, log_path, file_types)
    return output_cntr


def write_log_message(results, output_cntr):
    """logs to the terminal"""
    if len(results) == 0:
        return output_cntr
    log_lines = [f"{path}: {prompt}\n" for path, prompt in results]
    if len(log_lines)>1:
        subcntr = 1
        for l in log_lines:
           print(f"[{output_cntr}.{subcntr}] {l}", end="")
           subcntr += 1
    else:
           print(f"[{output_cntr}] {log_lines[0]}", end="")
    return output_cntr+1

def write_log_files(results, log_path, file_types):
    for file_type in file_types:
        if file_type == "txt":
            write_log_txt(log_path, results)
        elif file_type == "md" or file_type == "markdown":
            write_log_markdown(log_path, results)
        else:
            print(f"'{file_type}' format is not supported, so write in plain text")
            write_log_default(log_path, results, file_type)


def write_log_default(log_path, results, file_type):
    plain_txt_lines = [f"{path}: {prompt}\n" for path, prompt in results]
    with open(log_path + "." + file_type, "a", encoding="utf-8") as file:
        file.writelines(plain_txt_lines)


def write_log_txt(log_path, results):
    txt_lines = [f"{path}: {prompt}\n" for path, prompt in results]
    with open(log_path + ".txt", "a", encoding="utf-8") as file:
        file.writelines(txt_lines)


def write_log_markdown(log_path, results):
    md_lines = []
    for path, prompt in results:
        file_name = os.path.basename(path)
        md_lines.append(f"## {file_name}\n![]({file_name})\n\n{prompt}\n")
    with open(log_path + ".md", "a", encoding="utf-8") as file:
        file.writelines(md_lines)
