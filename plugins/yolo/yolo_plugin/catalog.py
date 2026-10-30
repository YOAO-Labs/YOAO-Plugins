"""Read this installed distribution's immutable pretrained descriptor bytes."""
from hashlib import sha256
from importlib.resources import files


def catalog_resource():
    return files("yolo_plugin").joinpath("pretrained-assets.json")


def catalog_bytes():
    return catalog_resource().read_bytes()


def catalog_sha256():
    return sha256(catalog_bytes()).hexdigest()
