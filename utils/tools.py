# tools
import os 


# file saver tool
file = os.path.join("data", "file.txt")

def save_file(text):
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass  

    with open(file, "a") as f:
        f.writelines([text + "\n"])

        return "file saved"

