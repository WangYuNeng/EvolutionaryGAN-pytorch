import subprocess
import os

dir_name = "data/"


def get_file_names(start_percent, end_percent):
    subprocess.call(["wget", "https://dumps.wikimedia.org/enwiki/latest/", "-O", dir_name + "index.html"])
    target_string = "<a href=\"enwiki-latest-pages-articles"
    target_len = len(target_string)
    filenames = list()
    with open(dir_name + "index.html", "r") as f:
        for line in f:
            # capture lines with <a href="enwiki-latest-pages-articles1~27
            if line.startswith(target_string) and line[target_len].isdigit() and line.find("-rss") == -1:
                start_idx = len("<a href=\"")
                end_idx = line.find("\">")
                filenames.append(line[start_idx:end_idx])

        num_files = len(filenames)
        return filenames[int(num_files*start_percent): int(num_files*end_percent)]


def download_and_parse(filenames, start_percent, end_percent):
    output_f = dir_name + str(start_percent) + "to" + str(end_percent) + ".txt"
    if os.path.exists(output_f):
        print("WARN:", output_f, "has already existed")
        return
    output_f = open(output_f, "w")
    for i, f in enumerate(filenames):
        script = "wikifil.pl"
        url = "https://dumps.wikimedia.org/enwiki/latest/" + f
        f = dir_name + f
        extracted_f = f[:-4]
        if not os.path.exists(f):
            print("wget", url, "-O", f)
            subprocess.call(["wget", url, "-O", f])
        else:
            print("WARN:", f, "has already existed")
        if not os.path.exists(extracted_f):
            print("bzip2", f, "-dk")
            subprocess.call(["bzip2", f, "-dk"])
        else:
            print("WARN:", extracted_f, "has already existed")
        print("perl", script, extracted_f)
        subprocess.call(["perl", script, extracted_f], stdout=output_f)

        # remove extracted file to save space
        print("rm", extracted_f)
        subprocess.call(["rm", extracted_f])
    output_f.close()


if __name__ == '__main__':
    (start_percent, end_percent) = (0, 0.1)
    fnames = get_file_names(start_percent, end_percent)
    download_and_parse(fnames, start_percent, end_percent)
