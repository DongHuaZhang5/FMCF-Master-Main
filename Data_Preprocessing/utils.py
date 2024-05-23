
import re
import numpy as np
import pickle as pkl


def re_0002(i):
    # split camel case & snake case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0].lower(), tmp[1].lower())
    else:
        return ' '.format(tmp)
# first regex for removing special characters, the second for camelCase and snake_case
re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

re_opt = re.compile(r'\+|-|\*|/|\*\*|\+\+|--|%|<<|>>|&&|\|\||&|\|\^|<|>|<=|>=|==|!=|=|\|=|^=|&=|<<=|>>=|\+=|-=|\*=|/=|%=|:=|~|=:|_')



