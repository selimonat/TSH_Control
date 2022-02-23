import configparser
from deta import Deta
import pickle
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                    )
logger = logging.getLogger(__name__)

parser = configparser.ConfigParser()
parser.read('config.txt')


def ensure_iterable(obj):
    """
    Ensures OBJ is iterables by wrapping it inside a list when needed.
    :param obj:
    :return:
    """
    if type(obj) is not list():
        obj = [obj]
    return obj


def deta_drive(drive_name='physio'):
    """
    Deta connector to drive containing physiological datasets.
    Returns: Drive instance.

    """
    deta = Deta(parser.get("deta", 'projectKey'))
    drive = deta.Drive(drive_name)
    return drive


def download(filenames):
    """
    Downloads the pickle files from the deta drive. The folder structure is hard coded to the code so you can only
    give the filename.
    Args:
        filenames:
            A list containing get_complaints, get_dosage, get_mood, get_tsh or any combinations thereof.
            For example:
            filenames = ['get_mood', 'get_tsh', 'get_dosage', 'get_complaints']

    Returns:
        A dict export of the pandas dataframe.
    """
    # filenames = ensure_iterable(filenames)

    data = dict()
    drive = deta_drive()
    for filename in filenames:
        logger.info(f"Downloading {filename}.")
        filename = f'data/clean/{filename}.pkl'
        data.update(pickle.loads(drive.get(filename).read()))
    return data

