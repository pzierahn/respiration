import json


def write_json(filename: str, data: any):
    with open(filename, 'w') as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
            default=str)
