from itertools import count

_counter = count()

def fresh_name():
    return f"id_{next(_counter)}"

def reset_name_provider():
    global _counter
    _counter = count()
