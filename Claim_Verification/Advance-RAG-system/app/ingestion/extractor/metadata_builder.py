import os


class MetadataBuilder:

    @staticmethod
    def build(file_path: str):

        stats = os.stat(file_path)

        return {
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": stats.st_size
        }