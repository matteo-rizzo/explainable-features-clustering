import glob
import os
from pathlib import Path
from typing import re


def intersect_dicts(dict_a: dict, dict_b: dict, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using dict_a values
    return {k: v for k, v in dict_a.items() if k in dict_b
            and not any(x in k for x in exclude) and v.shape == dict_b[k].shape}


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def increment_path(path: str, exist_ok: bool = True, sep: str = ''):
    """
    Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.

    :param path: path to check
    :param exist_ok: ok to overwrite
    :param sep: separator character
    :return: either same path (if it doesn't exist) or an incremental one
    """
    path = Path(path)
    # If the path is ok
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    # Otherwise give incremental name
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def check_file_exists(file: str) -> str:
    """
    Check if a file exists in the specified path or in the current directory and all its subdirectories
    :param file: file to check
    :return: the file (str) if found, will raise an assertion error otherwise
    """
    # Search for file if not found
    if Path(file).is_file() or file == '':
        return file
    else:
        # Searches in all subdirectories
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File Not Found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def get_latest_run(search_dir: str = '.'):
    """
    Find the file matching the pattern "last*.pt" in a directory and its subdirectories

    :param search_dir: directory to search in
    :return: most recently modified file matching the pattern
    """
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''
