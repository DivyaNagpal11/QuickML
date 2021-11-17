from zipfile import ZipFile
import os


def unzip(file, project_id):
    directory = r"mlp/static/Resources/" + str(project_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with ZipFile(file, 'r') as zip:
        zip.extractall(directory)
