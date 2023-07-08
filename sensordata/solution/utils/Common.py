import operator
from typing import Any, Union


def getByKey(_dict: dict, _key: Union[list, str]) -> Any:
    if isinstance(_key, list):
        return operator.itemgetter(*_key)(_dict)
    else:
        return operator.itemgetter(_key)(_dict)
