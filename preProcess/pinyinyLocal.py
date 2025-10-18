#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Translate chinese hanzi to pinyin by python
Created by Eric Lo on 2010-05-20.
Copyright (c) 2010 __lxneng@gmail.com__. http://lxneng.com All rights reserved.
"""

"""
Forked by skydarkchen <skydark2 at gmail>
"""

import os.path



class PinyinyLocal:

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                             'Mandarin.dat')

    def __init__(self):
        self.dict = {}
        self.revdict = {}
        for line in open(self.data_path):
            k, v = line.strip().split('\t')
            v = v.lower().split(' ')
            hz = chr(int('0x%s' % k, 16))
            self.dict[hz] = v
            for vkey in v:
                self.revdict.setdefault(vkey, [])
                self.revdict[vkey].append(hz)

    def py2hz(self, pinyin):
        if pinyin == '':
            return []
        pinyin = pinyin.lower()
        if pinyin[-1].isdigit():
            return self.revdict.get(pinyin, [])
        ret = []
        for i in range(1, 6):
            key = '%s%s' % (pinyin, i)
            ret += self.revdict.get(key, [])
        return ret

    def get_pinyin(self, chars='', splitter='', tone=False):
        result = []
        for char in chars:
            v = self.dict.get(char, None)
            if v:
                v = v[0]
                if not tone and v[-1].isdigit():
                    v = v[:-1]
            else:
                v = char
            result.append(v)
        return splitter.join(result)

    def get_initials(self, char=''):
        if char == '':
            return ''
        return self.dict.get(char, [char])[0][0].upper()


if __name__ == '__main__':
    p = PinyinyLocal()
    print(p.get_pinyin("上海"))

