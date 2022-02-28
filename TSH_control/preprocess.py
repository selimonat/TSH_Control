import pandas as pd
import datetime
import xml.etree.ElementTree
from joblib import Memory
import logging
import pickle
import os.path
from deta_utils import deta_drive
from utils import iter_records

location = '../cachedir'
memory = Memory(location, verbose=0)

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                    )
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

FILE_LAB = "../data/raw/BloodTests.csv"  # the file that is updated with each lab result
FILE_RATINGS = "../data/raw/Daylio_export.csv"
FILE_APPLE_HEALTH = "../data/raw/apple_health_export/export.xml"
FILE_APPLE_HEALTH_CACHE = "../data/raw/apple_health_export/export_cached.csv"
DIR_WRITE = "../data/clean/"

RESOLUTION = 'M'

"""
    Preprocess lab results, mood ratings and health complaints data, resampled at a weekly resolution.
"""


def get_dosage():
    """
        Get medicament dosage history, units µg. Source: manual txt file.

        :return: df with the following columns
        ['medication_change_start', 'dosage']
    """
    logger.info('Getting dosage data.')
    cols = ['test_date', 'TSH', 'T3', 'T4', 'medication_change_start', 'medication_change_stop', 'dosage']
    df_dosage = pd.read_csv(FILE_LAB, delimiter=',', header=None, names=cols)
    df_dosage["medication_change_start"] = pd.to_datetime(df_dosage["medication_change_start"], format='%d-%m-%Y')
    df_dosage["medication_change_stop"] = pd.to_datetime(df_dosage["medication_change_stop"], format='%d-%m-%Y')

    # make data
    df_dosage = pd.concat([
        pd.DataFrame({'startDate': df_dosage["medication_change_start"], 'dosage': df_dosage["dosage"]}),
        pd.DataFrame({'startDate': df_dosage["medication_change_stop"], 'dosage': df_dosage[
            "dosage"]})]).sort_values(by='startDate')

    df_dosage = df_dosage[['startDate', 'dosage']]

    df_dosage.set_index('startDate', inplace=True)

    df_dosage2 = df_dosage.resample(RESOLUTION).mean().interpolate(method='nearest')
    return df_dosage2


def get_tsh():
    """
        Get TSH time-series resampled weekly, aligned to Mondays.
        source: manual txt file.

        :return: df with the following columns
        ['test_date', 'TSH']
    """
    logger.info('Getting tsh data.')
    cols = ['test_date', 'TSH', 'T3', 'T4', 'medication_change_start', 'medication_change_stop', 'dosage']
    df_tsh = pd.read_csv(FILE_LAB, delimiter=',', header=None, names=cols)
    df_tsh["test_date"] = pd.to_datetime(df_tsh["test_date"], format='%d-%m-%Y')
    df_tsh = df_tsh[['test_date', 'TSH']]
    df_tsh.rename(columns={'TSH': 'tsh'}, inplace=True)
    df_tsh = df_tsh.loc[~df_tsh['test_date'].isna()]
    df_tsh.rename(columns={'test_date': 'startDate'}, inplace=True)
    df_tsh.set_index('startDate', inplace=True)
    # this protects the step-wise nature of the dosage regime.
    df_tsh2 = df_tsh.resample('D').mean().pad().resample(RESOLUTION).max()
    return df_tsh2


def get_mood():
    """
    Reads mood ratings from file and returns a time-series resampled weekly, aligned to Mondays.

    Source: Daylio app export.
    :return: df
    """
    # read mood data
    logger.info('Getting mood data.')
    df_mood = pd.read_csv(FILE_RATINGS, usecols=[0, 4])
    df_mood.index = pd.to_datetime(df_mood["full_date"])
    df_mood.sort_index(inplace=True)
    df_mood.index.rename('startDate', inplace=True)

    df_mood = df_mood.resample(RESOLUTION, convention='start').mean()

    return df_mood


def get_complaints():
    """
    Returns health complaints. Source: Daylio app export.
    :return: a df with one binary dummy variable for each health complaint data.
    """
    # read mood data
    logger.info('Getting complaints data.')
    df_mood = pd.read_csv(FILE_RATINGS, usecols=[0, 4, 5])
    df_mood.index = pd.to_datetime(df_mood["full_date"])
    df_mood.sort_index(inplace=True)
    # expand multi activities
    act = df_mood["activities"].str.split("|", expand=True)
    # remove spaces
    for col in act:
        act[col] = act[col].str.strip()
    act = pd.get_dummies(act.stack().droplevel(1))
    act = act.resample(RESOLUTION, convention='start').sum()
    act.index.rename('startDate', inplace=True)
    # remove columns which are not complaints
    cols = ['arrhythmia', 'bad_sleep', 'eyes hurte', 'fatigue', 'mood_angry', 'pain_back', 'pain_bone', 'pain_chest',
            'pain_foot', 'pain_hand', 'pain_hip', 'pain_knee', 'pain_neck', 'pain_stomach', 'restless']
    act = act[cols]
    return act


def get_cumulative_complaints():
    """
    Generates a cumulative view of complaints by aggregating across all complaints for a given time point.

    Returns:
    df, where each time interval consists of the sum
    """
    df = get_complaints()
    df = pd.DataFrame({'complaints': df.sum(axis=1)})

    return df


def return_apple_health_as_df(force=False):
    """
    Caches raw apple health export as a pandas dataframe. Use force to re-cache it.
    Returns:
        A pandas dataframe.
    """
    if not os.path.isfile(FILE_APPLE_HEALTH_CACHE):
        logger.info("Parsing Health.app XML.")
        data = xml.etree.ElementTree.parse(FILE_APPLE_HEALTH).getroot()
        logger.info("Converting XML to DF.")
        df_apple = pd.DataFrame.from_dict(iter_records(data))
        df_apple = df_apple[['startDate', 'value', 'sourceName', 'type']]
        df_apple['startDate'] = pd.to_datetime(df_apple['startDate'])
        df_apple.to_csv(FILE_APPLE_HEALTH_CACHE)
        logger.info("File cached.")
    else:
        logger.info("Loading Apple Health data from disk.")
        df_apple = pd.read_csv(FILE_APPLE_HEALTH_CACHE, parse_dates=['startDate'], index_col=[0])

    mask = df_apple['startDate'] >= '2018-09'
    df_apple = df_apple[mask]
    df_apple['week_number'] = df_apple.startDate.dt.isocalendar().week
    df_apple['year'] = df_apple.startDate.dt.isocalendar().year
    return df_apple


def get_apple_health_weight():
    """
        Read iOS Health.app data export
        :return: df
    """
    act = 'HKQuantityTypeIdentifierBodyMass'
    logger.info(f'Getting {act} data.')
    df_apple = return_apple_health_as_df()
    df_apple = df_apple.loc[df_apple.type == act]
    df_apple['value'] = df_apple['value'].astype(float)

    df1 = df_apple.groupby(['week_number', 'year', 'sourceName']).mean().reset_index()
    dates = df1.year*100+df1.week_number
    df1['date'] = pd.to_datetime(dates.astype(str) + '1', format='%Y%W%w')

    df1 = df1.set_index('date')
    df1 = df1.drop(columns=['week_number', 'year'])
    df1 = df1.sort_index()
    return pd.DataFrame({"weight": df1.resample(RESOLUTION)['value'].mean()}).interpolate(method='linear')


def get_apple_health_distance():
    """
        Read iOS Health.app data export
        :return: df
    """
    act = 'HKQuantityTypeIdentifierDistanceWalkingRunning'
    logger.info(f'Getting {act} data.')
    df_apple = return_apple_health_as_df()
    df_apple = df_apple.loc[df_apple.type == act]
    df_apple['value'] = df_apple.loc[:, 'value'].astype(float)

    df = df_apple.groupby(['week_number', 'year', 'sourceName']).sum().reset_index()
    dates = df.year*100+df.week_number
    df['date'] = pd.to_datetime(dates.astype(str) + '1', format='%Y%W%w')

    df = pd.DataFrame(df.reset_index().pivot_table(index='date', columns='sourceName', values='value',
                                              aggfunc='mean').mean(axis=1), columns=['value'])

    df = df.sort_index()
    return pd.DataFrame({"distance": df.resample(RESOLUTION)['value'].mean()})


def get_funs():
    """
        Returns all the available functions returning physiological datasets.
        Use this to loop over.
    """
    return [get_tsh, get_dosage, get_mood, get_complaints, get_cumulative_complaints, get_apple_health_weight,
            get_apple_health_distance, get_correlation]


def data_to(where='local'):
    """
    Pickles physiological datasets (results of get_X() exported as dicts) either locally or remotely (to deta cloud).
    Local files are saved in clean dir under data dir.
    Remote files are saved to deta server as per configuration keys.

    Args:
        where: 'local' or 'remote'.

    Returns:
        True.
    """
    drive = deta_drive(drive_name='physio')
    for fun in get_funs():
        df = fun()
        df.index = df.index.astype(str)
        filename = f'{DIR_WRITE}{fun.__name__}.pkl'
        if where is 'local':
            logger.info(f"Pickling {filename} to local disk.")
            with open(filename, 'wb') as handle:
                dummy = df.to_dict()
                # we have to add this top level key otherwise as the correlation dict is loaded
                # previously loaded keys get overwritten.
                if fun is get_correlation:
                    dummy = {'correlation': dummy}
                pickle.dump(dummy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif where is 'remote':  # for remote to work local must be first called.
            if os.path.isfile(filename):
                logger.info(f"Sending {filename} to deta drive.")
                drive.put(filename, path=filename)
            else:
                raise ValueError('Local files are not yet saved.')
        else:
            raise ValueError('where argument must be either \'local\' or \'remote\'.')
    return True


def get_correlation():
    """

    Returns:
        df
    """
    logger.info('Computing correlation matrix, will need to call all funs')
    dfl = list()
    for f in [get_tsh, get_dosage, get_mood, get_cumulative_complaints, get_apple_health_weight,
              get_apple_health_distance]:
        dfl.append(f())
    df = pd.concat(dfl, axis=1)
    # only take interesting columns
    df = df[['tsh', 'dosage', 'mood', 'complaints', 'weight', 'distance']].corr().pow(2)
    # zero the diagonal
    df.values[[list(range(df.shape[0]))]*2] = 0
    return df


if __name__ == "__main__":
    # df = get_apple_health_distance()
    # dfl = list()
    # for f in get_funs():
    #     dfl.append(f())
    data_to('local')
    data_to('remote')
