import configparser
from deta import Deta
import pickle

parser = configparser.ConfigParser()
parser.read('config.txt')


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

    data = dict()
    drive = deta_drive()
    for filename in filenames:
        filename = f'data/clean/{filename}.pkl'
        data.update(pickle.loads(drive.get(filename).read()))
    return data
