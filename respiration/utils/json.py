import json


def prettify_json(data: any) -> str:
    """
    Prettify JSON data
    :param data:
    :return:
    """
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def pretty_print(data: any):
    """
    Prettify JSON data and print it
    :param data:
    :return:
    """
    print(prettify_json(data))


def write_json(filename: str, data: any):
    """
    Write JSON data to file
    :param filename:
    :param data:
    :return:
    """

    with open(filename, 'w') as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
            default=str)
