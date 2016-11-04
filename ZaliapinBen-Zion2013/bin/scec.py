#!/usr/bin/python


import sys
import datetime


def main(argv):
    # do not use events without magnitude information (magnitude = 0.0; see http://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_DD/hs_1981_2011_catalog_v01.format)
    events = sorted([e for e in load_scec(sys.stdin) if e[3] != 0], key=lambda e: e[0])
    for t, m, x, y in events:
        print(t, '\t', m, '\t', y, '\t', x)


def load_scec(fh):
    """
    - http://scedc.caltech.edu/research-tools/alt-2011-dd-hauksson-yang-shearer.html
    - http://service.scedc.caltech.edu/ftp/catalogs/hauksson/Socal_DD/hs_1981_2011_catalog_v01.format
    """
    for l in fh:
        # 1981 01 01 04 13 55.710   3301565 33.25524 -115.96763   5.664  2.26  45  17   0.21   1   4   260   460      76   0.300   0.800   0.003   0.003  le ct Poly5
        year, month, day, hour, minute, second, scecid, lat, lon, depth, magnitude, _ = l.split(maxsplit=11)
        fsecond = float(second)
        isecond = int(fsecond)
        delta_second = 0
        if isecond > 59:
            delta_second = isecond - 59
            fsecond -= delta_second
            isecond = 59
        delta_minute = 0
        minute = int(minute)
        if minute > 59:
            delta_minute = minute - 59
            minute = 59
        elif minute < 0:
            delta_minute = -minute
            minute = 0
        t = datetime.datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            minute,
            isecond,
            round(1000000*(fsecond - isecond)),
        ) + datetime.timedelta(minutes=delta_minute, seconds=delta_second)
        yield (
            t.timestamp(),
            float(magnitude),
            float(lon),
            float(lat),
        )


if __name__ == '__main__':
    main(sys.argv)
