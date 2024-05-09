import os


def project_root() -> str:
    """Returns project root folder"""
    return os.path.abspath(os.path.join('..', '..'))


def file_path(*paths: str) -> str:
    """Returns path from project root to file"""
    return os.path.join(project_root(), *paths)


def dir_path(*dirs: str, mkdir: bool = False) -> str:
    """Returns path from project root to directory"""

    path = os.path.join(project_root(), *dirs)

    if mkdir:
        os.makedirs(path, exist_ok=True)

    return path


def join_paths(*paths: str) -> str:
    """Joins paths"""
    return str(os.path.join(*paths))
