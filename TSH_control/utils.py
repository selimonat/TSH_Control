def iter_records(health_data) -> dict or None:
    """
        Utility to parse Apple Health.app data export.
    """
    health_data_attr = health_data.attrib
    for rec in health_data.iterfind('.//Record'):
        rec_dict = health_data_attr.copy()
        rec_dict.update(health_data.attrib)
        for k, v in rec.attrib.items():
            # if 'date' in k.lower():
            #     res = datetime.datetime.strptime(v, '%Y-%m-%d %H:%M:%S %z').date()
            # print(f"{k},{v} ==> {res}")
            # rec_dict[k] = v
            # else:
            rec_dict[k] = v
        yield rec_dict