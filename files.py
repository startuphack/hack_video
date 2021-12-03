# -*- coding: utf-8 -*-
import gzip
import io
import pickle


def pickle_dump(
    object,
    filename,
    gzip_file=True,
    protocol=pickle.HIGHEST_PROTOCOL
):
    """
    Сохранить объект в файл
    :param object: сохраняемый объект
    :param filename: имя файла
    :param gzip_file: сжимать ли сериализацию объекта с gzip
    :param protocol: протокол pickle
    :param use_cloudpickle: если True, используется cloudpickle, иначе - pickle.
    У cloudpickle есть проблема с упаковкой itemgetter объектов со списком slice.
    """
    o_method = gzip.open if gzip_file else open

    with io.BufferedWriter(o_method(filename, 'w')) as output:
        pickle.dump(object, output, protocol=protocol)


def pickle_load(filename, gzip_file=True):
    """
    Загрузить объект из файла filename
    :param filename: имя файла
    :param gzip_file: является ли файл сжатым с gzip
    :return: объект, загруженный из файла
    """
    i_method = gzip.open if gzip_file else open
    with i_method(filename) as input:
        return pickle.load(input)
