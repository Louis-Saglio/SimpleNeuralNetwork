import json


def is_iterable(obj):
    try:
        for _ in obj:
            return True
    except TypeError:
        return False


def obj_to_dict(data):
    return (
        {k: obj_to_dict(v) for (k, v) in dict(data.__dict__).items() if isinstance(k, str) and not k.startswith("__")}
        if getattr(data, "__dict__", False)
        else (
            [obj_to_dict(datum) for datum in data]
            if (is_iterable(data) and not isinstance(data, (str, bytes)))
            else repr(data)
        )
    )


def obj_to_json(data):
    return json.dumps(obj_to_dict(data), indent=2)


def rpprint(data):
    print(obj_to_json(data))


def batch(iterable, n=1):
    # credits :
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks#answer-8290508
    size = len(iterable)
    for ndx in range(0, size, n):
        yield iterable[ndx : min(ndx + n, size)]
