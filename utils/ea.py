import hashlib
import json
import pickle
import traceback
from collections import defaultdict
import numpy as np
import numpy
from progressbar import *
import base64
import os
import time

np.set_printoptions(linewidth=500, precision=4, threshold=32, edgeitems=3)


class Ea(defaultdict):
    def __init__(self):
        super(Ea, self).__init__()

    def __getattr__(self, name):
        if name in self:
            if self[name] is None:
                return ""
            return self[name]
        n = Ea()
        super(Ea, self).__setitem__(name, n)
        return n

    def __setattr__(self, name, value):
        super(Ea, self).__setitem__(name, value)

    def __getitem__(self, name):
        if name not in self:
            if type(name) is int:
                super(Ea, self).__setitem__(name, Ea())
            else:
                return None
        return super(Ea, self).__getitem__(name)

    @staticmethod
    def __to_str(ff, v):
        rv_str = ""
        if isinstance(v, float):
            rv_str += "%s: %.2f \n" % (ff, v)
        elif isinstance(v, int):
            rv_str += "%s: %d \n" % (ff, v)
        elif isinstance(v, bytes):
            rv_str += "%s: Bytes,%s \n" % (ff, len(v))
        elif isinstance(v, str):
            if v == "...":
                rv_str += "%s: %s \n" % (ff, v)
            else:
                rv_str += "%s: '%s' \n" % (ff, v)
        elif isinstance(v, np.ndarray) or isinstance(v, numpy.ndarray):
            rv_str += "%s: np.ndarray %s \n" % (ff, v.shape)
            # if v.shape == ():
            #     o = str(v)
            # else:
            #     i = ff.find('☞')
            #     oo = ff[0:i].ljust(len(ff))
            #     oo = oo.replace('├', '│')
            #     oo = oo.replace('─', ' ')
            #     oo = oo.replace('└', ' ')
            #     s = "\n%s" % oo.ljust(len(ff) + 1)
            #     # np.set_printoptions(linewidth=500, precision=2,threshold=3, edgeitems=3)
            #     if len(v) > 4:
            #         o = s + s.join(map(str, v[:3]))
            #         o = o + s + "    ...   "
            #         o = o + s + str(v[-1])
            #     else:
            #         o = s + s.join(map(str, v))
            #
            # rv_str += "%s: np.ndarray:%s %s\n" % (ff, v.shape, o)

        elif isinstance(v, tuple):
            rv_str += "%s: (%s) \n" % (ff, u",".join(map(str, v)))
        elif callable(v):
            rv_str += "%s: %s() \n" % (ff, v.__name__)
        elif v is None:
            rv_str += "%s: None \n" % ff
        else:
            rv_str += "%s%s \n" % (ff, type(v))

        return rv_str

    @staticmethod
    def _list_to_str(rv_str, ff, fs, limit, arr):
        if Ea._list_is_in_types(arr, [int, float, str]):
            rv_str = rv_str.rstrip("\n")
            rv_str += " %s \n" % (arr.__str__())
            return rv_str
        total = min(len(arr), limit + 1)
        for i in range(0, total):
            if isinstance(arr[i], Ea):
                if i < total - 1:
                    fff = "├─"
                    ffs = "│ "
                    rv_str += "%s%s□[%d]\n" % (fs, fff, i)
                    rv_str += arr[i].to_str("", limit, "%s%s" % (fs, ffs))
                else:
                    fff = "└─"
                    ffs = "  "
                    if len(arr) > limit:
                        rv_str += "%s%s□[.] ...\n" % (fs, fff)
                    else:
                        rv_str += "%s%s□[%d]\n" % (fs, fff, i)
                        rv_str += arr[i].to_str("", limit, "%s%s" % (fs, ffs))
            elif isinstance(arr[i], list):
                if i < total - 1:
                    fff = "├─"
                    ffs = "│ "
                    rv_str += "%s%s□[%d]: list(%d)\n" % (fs, fff, i, len(arr[i]))
                    rv_str = Ea._list_to_str(rv_str, ff, fs + ffs, limit, arr[i])
                else:
                    fff = "└─"
                    ffs = "  "
                    if len(arr) > limit:
                        rv_str += "%s%s□[.] ...\n" % (fs, fff)
                    else:
                        rv_str += "%s%s□[%d]\n" % (fs, fff, i)
                        rv_str = Ea._list_to_str(rv_str, ff, fs + ffs, limit, arr[i])
            elif isinstance(arr[i], set):
                if i < total - 1:
                    fff = "├─"
                    ffs = "│ "
                    rv_str += "%s%s□[%d]: set(%d)\n" % (fs, fff, i, len(arr[i]))
                    rv_str = Ea._set_to_str(rv_str, fs + ffs, limit, arr[i])
                else:
                    fff = "└─"
                    ffs = "  "
                    if len(arr) > limit:
                        rv_str += "%s%s□[.] ...\n" % (fs, fff)
                    else:
                        rv_str += "%s%s□[%d] set(%d)\n" % (fs, fff, i, len(arr[i]))
                        rv_str = Ea._set_to_str(rv_str, fs + ffs, limit, arr[i])
            else:
                if i < total - 1:
                    fff = "├─"
                    rv_str += Ea.__to_str("%s%s ☞[%d]" % (fs, fff, i), arr[i])
                else:
                    fff = "└─"
                    if len(arr) > limit:
                        rv_str += Ea.__to_str("%s%s ☞[.]" % (fs, fff), '...')
                    else:
                        rv_str += Ea.__to_str("%s%s ☞[%d]" % (fs, fff, i), arr[i])

        return rv_str

    @staticmethod
    def _set_to_str(rv_str, fs, limit, ss):
        arr = list(ss)
        total = min(len(arr), limit + 1)
        for i in range(0, total):
            # $ set 不可能包含 EA3,list,set 对象
            if i < total - 1:
                fff = "├─"
                rv_str += Ea.__to_str("%s%s ☞[%d]" % (fs, fff, i), arr[i])
            else:
                fff = "└─"
                if len(arr) > limit:
                    rv_str += Ea.__to_str("%s%s ☞[.]" % (fs, fff), '...')
                else:
                    rv_str += Ea.__to_str("%s%s ☞[%d]" % (fs, fff, i), arr[i])

        return rv_str

    def to_str(self, root="root", limit=1, f=""):
        rv_str = ""
        space = ""
        if limit <= 0:
            limit = -1

        if root:
            rv_str = "\n □[%s]\n" % root
            space = " "

        for k in sorted(self.keys()):
            # if k[0] == '_':
            #     continue

            if k is sorted(self.keys())[len(list(self.keys())) - 1]:
                ff = f + space + "└─"
                fs = f + space + "  "
            else:
                ff = f + space + "├─"
                fs = f + space + "│ "

            if isinstance(self[k], Ea):
                rv_str += "%s□[%s]\n" % (ff, k)
                rv_str += self[k].to_str("", limit, fs)
            elif isinstance(self[k], set):
                rv_str += "%s□[%s]: set(%d)\n" % (ff, k, len(self[k]))
                rv_str = self._set_to_str(rv_str, fs, limit, self[k])
            elif isinstance(self[k], list):
                rv_str += "%s□[%s]: list(%d)\n" % (ff, k, len(self[k]))
                rv_str = self._list_to_str(rv_str, ff, fs, limit, self[k])
            else:
                rv_str += self.__to_str("%s ☞[%s]" % (ff, k), self[k])

        return rv_str

    @staticmethod
    def dumps(obj):
        return pickle.dumps(obj)

    @staticmethod
    def loads(bytes_obj):
        return pickle.loads(bytes_obj)

    @staticmethod
    def jsons(obj, indent=None):
        return json.dumps(obj, ensure_ascii=False, indent=indent, separators=(',', ':'))

    @staticmethod
    def load_jsons(s):
        j = json.loads(s)
        if j:
            rv = Ea.clone(j)
            return rv
        return None

    @staticmethod
    def dump(obj, name, path="."):
        if not os.path.exists(path):
            os.makedirs(path)

        if '.pkl' not in name:
            name += '.pkl'

        full_path = "%s/%s" % (path, name)
        full_path = full_path.replace("//", "/")
        print(full_path)
        pickle.dump(obj, open(full_path, "wb"))

    @staticmethod
    def save_json(obj, full_path, indent=None, encoding='utf-8'):
        json.dump(obj, open(full_path, "wt", encoding=encoding), ensure_ascii=False, indent=indent,
                  separators=(',', ':'))

    @staticmethod
    def dump_json(obj, name, path=".", indent=None, encoding='utf-8'):
        if not os.path.exists(path):
            os.makedirs(path)

        if '.json' not in name:
            name += '.json'

        full_path = "%s/%s" % (path, name)
        full_path = full_path.replace("//", "/")
        # print(full_path)
        json.dump(obj, open(full_path, "wt", encoding=encoding), ensure_ascii=False, indent=indent,
                  separators=(',', ':'))

    @staticmethod
    def dump_base64(obj):
        pkl_bytes = pickle.dumps(obj)
        return bytes.decode(base64.b64encode(pkl_bytes), encoding="utf8")

    @staticmethod
    def load_base64(base64_str):
        b64_bytes = base64.b64decode(base64_str)
        return Ea.loads(b64_bytes)

    @staticmethod
    def load(f_path):
        if not os.path.exists(f_path):
            return None

        if f_path[-min(5, len(f_path)):] == '.json':
            fp = open(f_path, encoding='utf-8')
            j = json.load(fp)
            if j:
                rv = Ea.clone(j)
                return rv
            return None
        elif f_path[-min(4, len(f_path)):] == '.pkl':
            return pickle.load(open(f_path, "rb"))
        else:
            return None

    # @staticmethod
    # def log(obj, logger=None, limit=1, root=""):
    #     if logger is None:
    #         logger = lg(name="", level=logging.INFO)
    #
    #     if root:
    #         r = root
    #     else:
    #         r = traceback.extract_stack(limit=2)[0][3][8:][:-1]
    #
    #     if not obj:
    #         # r = traceback.extract_stack(limit=2)[0][3][8:][:-1]
    #         logger.info(r)
    #         return
    #
    #     if isinstance(obj, Ea):
    #         # r = traceback.extract_stack(limit=2)[0][3][8:][:-1]
    #         ss = obj.to_str(root=r, limit=limit)
    #         logger.info(ss)
    #     elif isinstance(obj, list):
    #         # r = traceback.extract_stack(limit=2)[0][3][8:][:-1]
    #         rv_str = " □[%s]: list(%d)\n" % (r, len(obj))
    #         logger.info(Ea._list_to_str(rv_str, "", " ", limit=limit, arr=obj))
    #     elif isinstance(obj, np.ndarray):
    #         print("aaa")
    #     else:
    #         # r = traceback.extract_stack(limit=2)[0][3][8:][:-1]
    #         logger.info(Ea.__to_str(r, obj))

    @staticmethod
    def _list_is_in_types(arr: list, types: list):
        rv = True
        for obj in arr:
            if isinstance(obj, list):
                rv = rv & Ea._list_is_in_types(obj, types)
            elif type(obj) not in types:
                return False
        return rv

    @staticmethod
    def show(obj, limit=1, root=""):
        if root:
            r = root
        else:
            r = traceback.extract_stack(limit=2)[0][3][8:][:-1]
        if not obj:
            print(r, None)
            return
        if isinstance(obj, Ea):
            print(obj.to_str(root=r, limit=limit))
        elif isinstance(obj, list):
            rv_str = " □[%s]: list(%d)\n" % (r, len(obj))
            print(Ea._list_to_str(rv_str, "", " ", limit=limit, arr=obj))
        elif isinstance(obj, np.ndarray):
            print(obj)
        elif isinstance(obj, dict):
            _obj = Ea.clone(obj)
            print(_obj.to_str(root=r, limit=limit))
        else:
            print(Ea.__to_str(r, obj))

    def _clone(self, dic):
        for k in dic:
            if isinstance(dic[k], dict):
                self[k] = Ea.clone(dic[k])
            elif isinstance(dic[k], list):
                self[k] = []
                for item in dic[k]:
                    if isinstance(item, dict):
                        self[k].append(Ea.clone(item))
                    else:
                        self[k].append(item)
            else:
                self[k] = dic[k]
        return self

    @staticmethod
    def clone(dic):
        ea = Ea()
        ea._clone(dic)
        return ea

    def has_key(self, key):
        return key in self

    @staticmethod
    def gget(obj, chain_arr):
        if isinstance(chain_arr, str):
            chain_arr = chain_arr.split(".")
        key_count = len(chain_arr)
        if key_count == 0:
            return obj

        key = chain_arr[0]
        if isinstance(obj, list):
            i = int(key)
            if i < len(obj):
                chain_arr.pop(0)
                return Ea.gget(obj[i], chain_arr)
        elif isinstance(obj, Ea):
            chain_arr.pop(0)
            return Ea.gget(obj[key], chain_arr)
        return None

    @staticmethod
    def keychain_items(obj, keychain=None):
        rv = []
        if not keychain:
            keychain = traceback.extract_stack(limit=2)[0][3][23:][:-1]
        if isinstance(obj, Ea) or isinstance(obj, dict):
            for k in obj:
                rv += Ea.keychain_items(obj[k], keychain + "." + k)
            return rv
        elif isinstance(obj, list):
            for i in range(0, len(obj)):
                rv += Ea.keychain_items(obj[i], keychain + ".%d" % i)
            return rv
        else:
            rv.append((keychain, obj))
            return rv

    @staticmethod
    def series(s):
        rv = Ea()
        for key in list(s.index):
            rv[key] = s[key]
        return rv

    @staticmethod
    def df(df, key="index"):
        rv = Ea()

        if key == "index":
            for index, row in df.iterrows():
                rv[index] = Ea.series(row)
        else:
            for index, row in df.iterrows():
                rv[row[key]] = Ea.series(row)
        return rv

    @staticmethod
    def md5_key(ea):
        m2 = hashlib.md5()
        s = ea.to_json()
        m2.update(s.encode())
        key = (m2.hexdigest()).upper()
        return key

    class TestUtils:
        @staticmethod
        def base64Test():
            a = {"name": "王小波", "age": 18, "flag": True, "address": "aaaaaaaaaa街道"}
            base64_str = Ea.dump_base64(a)
            print(base64_str)
            b = Ea.load_base64(base64_str)
            print(b, type(b))

        @staticmethod
        def demo_ea():
            A = Ea()
            A.a = 'a'
            A.b = 'b'
            A.c.cc = 'cc'
            A.d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
            Ea.show(A)

def demo():
    Ea.TestUtils.base64Test()
    Ea.TestUtils.demo_ea()
    pass

if __name__ == '__main__':
    demo()
    pass
