from configparser import RawConfigParser
import numpy as np
import json

class Config:
    @staticmethod
    def get_conf(file_path, section, key, default=None):
        cfg = Config.get_conf_dic(file_path)
        if cfg is not None:
            if section in cfg:
                if key in cfg[section]:
                    value = cfg[section][key]
                    return value
        return default

    @staticmethod
    def get_conf_dic(file_path):
        config = RawConfigParser()
        config.read(file_path)
        conf_dic = {}
        for key in config.sections():
            conf_dic[key] = {}
            for _k,_v in config.items(key):
                conf_dic[key][_k] = _v
        return conf_dic

    @staticmethod
    def rm_option(file_path, section, key):
        print("remove [%s] %s from %s." % (section, key, file_path))
        config = RawConfigParser()
        config.read(file_path)
        rv = config.remove_option(section, key)
        if rv:
            cfgfile = open(file_path, 'w')
            config.write(cfgfile, space_around_delimiters=True)  # use flag in case case you need to avoid white space.
            cfgfile.close()

    @staticmethod
    def rm_section(file_path, section):
        print("remove [%s] from %s." % (section, file_path))
        config = RawConfigParser()
        config.read(file_path)
        rv = config.remove_section(section)
        if rv:
            cfgfile = open(file_path, 'w')
            config.write(cfgfile, space_around_delimiters=True)  # use flag in case case you need to avoid white space.
            cfgfile.close()


    @staticmethod
    def set_conf(file_path, section, key, value):
        config = RawConfigParser()
        config.read(file_path)
        if not config.has_section(section):
            config.add_section(section)
        if type(value) is np.mat:
            config.set(section, key, value.tolist())
        elif type(value) is np.ndarray:
            config.set(section, key, value.tolist())
        else:
            config.set(section, key, value)
        cfgfile = open(file_path, 'w')
        config.write(cfgfile, space_around_delimiters=True)  # use flag in case case you need to avoid white space.
        cfgfile.close()
        # print("----- Config Save:  [%s] '%s=%s' to %s." % (section, key, config.get(section, key), file_path))


def demo():
    img_w = Config.get_conf("../../map2d/config.ini", "Map2D", "img_w")
    print("img_w =",img_w)

    cfg_dic = Config.get_conf_dic("../../map2d/config.ini")
    print(cfg_dic)
    pass

if __name__ == '__main__':
    demo()
    pass
