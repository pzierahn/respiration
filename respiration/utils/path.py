import os


def project_root() -> str:
    """Returns project root folder"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def file_path(*path) -> str:
    """Returns path from project root to file"""
    return os.path.join(project_root(), *path)


def dir_path(*dirs, mkdir: bool = True) -> str:
    """Returns path from project root to directory"""

    path = os.path.join(project_root(), *dirs)

    if mkdir:
        os.makedirs(path, exist_ok=True)

    return path
