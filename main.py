import configparser

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("settings.ini")
    print(config.sections())
