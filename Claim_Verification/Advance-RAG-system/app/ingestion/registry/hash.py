import hashlib 

def generate_file_hash(file_path:str)-> str:
    sha256_hash = hashlib.sha256()
    with open(file_path,"rb") as file:
        while chunk := file.read(8192):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
