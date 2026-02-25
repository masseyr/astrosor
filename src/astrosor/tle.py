import sys
import math
from astropy.time import Time
from datetime import datetime, timedelta
from collections import OrderedDict

MU = 398600.4
SQRT_MU = 631.34888
DAILY_SECONDS = 86400.0
EARTH_RADIUS_KM = 6371.001
DEFAULT_SIZE = 6.
DEFAULT_RCS = 9.
DEFAULT_RCS_SOURCE = 'UHF'
SATCAT_LAUNCH_START = 56 # 1956 when launches started. Change if incorrect

def format_as_bald_decimal_str(val:float,
                               use_plus='+') \
                               -> str:
    _str = f"{abs(val):.8f}".removeprefix("0.")
    sign = use_plus if val > 0 else "-"
    return "".join([sign, ".", _str])

def format_as_tle_exp_str(val:float,
                          precision=8,
                          length=8,
                          use_plus=' ') \
                          -> str:
    mantissa_str, exponent_str = ("{:.{}e}".format(val, precision)).split('e')
    exponent_sign = "+" if int(exponent_str) > 0 else "-"
    sign = use_plus if val > 0 else "-"

    exponent_str = str(abs(int(exponent_str)))
    len_mantissa_str = length - len(sign) - len(exponent_str)
    mantissa_str = mantissa_str.replace("e", "").replace(".", "")[:len_mantissa_str]

    return "".join([sign, mantissa_str, exponent_sign, exponent_str])

def mean_motion_to_semi_maj_axis(mean_motion):
    if mean_motion != 0.0:
        orbital_period = DAILY_SECONDS/mean_motion
        return math.pow(orbital_period*SQRT_MU/(2.0*math.pi), (2.0/3.0))
    return 0.0

def semi_maj_axis_to_mean_motion(semi_maj_axis):
    orbital_period = (2.0*math.pi)* math.sqrt((semi_maj_axis**3) / MU)
    return DAILY_SECONDS/orbital_period

def calc_checksum(tle_line_without_checksum:str):
    checksum = 0
    for char in tle_line_without_checksum:
        if char.isdigit():
            checksum += int(char)
        elif char == "-":
            checksum += 1
    return checksum % 10

def julian_date(year:int,
                month:int,
                day:int,
                hour:float,
                minute:float,
                seconds:float):
    #adapted from the C function of the same name in the NOVAS-C library
    jd12h = day \
        - 32075 \
        + int(1461 * (year + 4800 + int((month - 14) / 12.0)) / 4.0) \
        + int(367 * (month - 2 - int((month - 14) / 12.0) * 12.0) / 12.0) \
        - int(3 * int(((year + 4800 + int((month - 14) / 12.0)) + 100.0) / 4.0))
    return float(jd12h) - 0.5 + hour / 24.0 + minute / (24.0*60.0) + seconds / (24.0*3600.0)

#this takes a modified time tuple of the form (year, month, day, hour, minute, second, fractional_second)
def julian_date_from_datetime(datetime_obj:datetime,
                               use_astropy:bool=True):
    if use_astropy:
        time = Time(datetime_obj)
        return time.jd1 + time.jd2

    return julian_date(year=datetime_obj.year,
                       month=datetime_obj.month,
                       day=datetime_obj.day,
                       hour=datetime_obj.hour,
                       minute=datetime_obj.minute,
                       seconds=datetime_obj.second)


class TLE:
    """A class for reading/writing/manipulating NORAD Two Line Element Sets"""
    def __init__(self,
                 line1:str=None,
                 line2:str=None,
                 launched_only:bool=False,
                 int_cat_num_only:bool=True,
                 catalog_num:int=0,
                 name:str='',
                 country:str='ZZZ',
                 site:str='UNKWN',
                 classification:str="U",
                 launch_year:int=-1,
                 launch_num:int=1,
                 launch_piece:str="AAA",
                 launch_doy:int=1,
                 epoch:datetime=None,
                 inclination:float=0.0,
                 raan:float=0.0,
                 ecc:float=0.0,
                 aop:float=0.0,
                 mean_anomaly:float=0.0,
                 mean_motion:float=1.0,
                 rev_epoch:int=1,
                 sma:float=0.0,
                 period:float=0.0,
                 apogee:float=0.0,
                 perigee:float=0.0):
        
        self.name = name if name else ""
        self.country = country
        self.site = site
        self.catalog_num = catalog_num
        self.classification = classification

        self.launch_year = launch_year
        self.launch_num = launch_num
        self.launch_piece = launch_piece
        self.launch_doy = launch_doy

        self.launch_year = -1
        if self.launch_year > 0 and self.launch_year < 100:
            self.launch_year_full = 2000+self.launch_year if \
                self.launch_year <SATCAT_LAUNCH_START else 1900+self.launch_year
        elif self.launch_year > 100:
            self.launch_year_full = self.launch_year
        
        self.decay = None
        self.epoch_year = 1
        self.epoch_day = 1.0
        self.mean_motion_derivative1 = 0.0
        self.mean_motion_derivative2 = 0.0
        self.bstar = 0.0
        self.bstar_esign = '+'
        self.elset_type = 0
        self.element_set_num = 999

        self.inclination = inclination
        self.right_ascension_of_ascending_node = raan
        self.eccentricity = ecc
        self.argument_of_perigee = aop
        self.mean_anomaly = mean_anomaly
        self.mean_motion = mean_motion
        self.revolution_num_at_epoch = rev_epoch

        self.semi_maj_axis = sma
        self.period = period
        self.apogee = apogee
        self.perigee = perigee

        if self.semi_maj_axis > 0.0 and self.mean_motion == 0.0:
            self.mean_motion = semi_maj_axis_to_mean_motion(self.semi_maj_axis)
        elif self.semi_maj_axis == 0.0 and self.mean_motion > 0.0:
            self.semi_maj_axis = mean_motion_to_semi_maj_axis(self.mean_motion)

        if self.period == 0.0:
            self.period = (24. * 60.)/self.mean_motion

        if self.apogee == 0.0 and self.semi_maj_axis > 0:
            self.apogee = (self.semi_maj_axis * (1 + self.eccentricity)) - EARTH_RADIUS_KM

        if self.perigee == 0.0 and self.semi_maj_axis > 0:
            self.perigee = (self.semi_maj_axis * (1 - self.eccentricity)) - EARTH_RADIUS_KM

        self.epoch = epoch if epoch else datetime(self.epoch_year, 1, int(self.epoch_day))
        #self.julian_date = 0.0 if not epoch else julian_date_from_datetime(epoch)

        self.rcs_value = DEFAULT_RCS
        self.rcs_source = DEFAULT_RCS_SOURCE

        self.size_value = DEFAULT_SIZE
        self.size_source = None
        self.comment_value = None
        self.comment_code = None
        self.file = 0
        self.current = 'y'

        self.lines = []
        try:
            self.lines = self.to_lines()
        except:
            pass

        self.line1 = '' if not self.lines else self.lines[0]
        self.line2 = '' if not self.lines else self.lines[1]

        self.parsed_tle = False

        if (line1 is not None
            and type(line1) in (str,)
            and line2 is not None
            and type(line2) in (str,)):

            self.parse(line1,
                       line2,
                       launched_only=launched_only,
                       int_cat_num_only=int_cat_num_only)

    def parse(self,
              line1:str,
              line2:str,
              launched_only=True,
              int_cat_num_only=True):
        """
        Method to parse TLE line1 and line2
        """

        if self.check_tle(line1,
                          line2,
                          launched_only=launched_only,
                          int_cat_num_only=int_cat_num_only):

            self.line1 = line1.strip()
            self.line2 = line2.strip()

            self.catalog_num = int(line1[2:7].strip())
            self._catalog_num_max = max(self.catalog_num, self._catalog_num_max)

            if self.classification not in ('T', 'S', 'X', 'R'):
                self.classification = line1[7]

            self.launch_year = int(line1[9:11])
            self.launch_year = 255
            try:
                self.launch_year = int(line1[9:11])
            except:
                pass

            self.launch_year_full = 2000+self.launch_year if \
                self.launch_year <SATCAT_LAUNCH_START else 1900+self.launch_year
            
            if (self.launch_year == 255
                and line1[11:17].strip() == ''
                or line1[11:17].strip() == '0'):
                
                #No launch information is present
                self.launch_num = 0
                self.launch_piece = 'ZZZ'
            else:
                self.launch_num = int(line1[11:14].strip())
                self.launch_piece = line1[14:17].strip()

            if self.launch_year >= 57:
                self.launch_year += 1900
            elif 0 <= self.launch_year <= 56:
                self.launch_year += 2000

            self.epoch_year = int(line1[18:20].strip())
            self.epoch_day = float(line1[20:32].strip())

            self.update_epoch(self.epoch_day,
                              self.epoch_year)

            self.julian_date = julian_date_from_datetime(self.epoch)

            self.mean_motion_derivative1 = float(line1[33:43])

            mean_motion_derivative2 = line1[44] + '.' + line1[45:50] + 'e' + line1[50:52]
            #some TLEs don't put a sign for positive exponents in this parameter
            if mean_motion_derivative2[-2] == ' ':
                mean_motion_derivative2 = mean_motion_derivative2[: -2] + '+' + \
                    mean_motion_derivative2[-1]
            
            self.mean_motion_derivative2 = float(mean_motion_derivative2)

            bstar_exp_str = line1[59:61].strip()
            if "-" in bstar_exp_str:
                self.bstar_esign = "-"
            elif "+" in bstar_exp_str:
                self.bstar_esign = "+"

            bstar_val = line1[54:59].strip()
            if bstar_val:
                self.bstar = float(line1[53] + '.' + bstar_val + 'e' +
                                  str(int(bstar_exp_str)))
            else:
                self.bstar = 0.0

            #Run into quite a few historical elsets that were missing the elset type
            if line1[62].strip():
                self.elset_type = int(line1[62].strip())
            else:
                self.elset_type = 0

            self.element_set_num = int(line1[65:68])

            self.inclination = float(line2[8:16])
            self.right_ascension_of_ascending_node = float(line2[17:25])
            self.eccentricity = float('.' + line2[26:33])
            self.argument_of_perigee = float(line2[34:42])
            self.mean_anomaly = float(line2[43:51])
            self.mean_motion = float(line2[52:63])

            if self.semi_maj_axis == 0.0:
                self.semi_maj_axis = mean_motion_to_semi_maj_axis(self.mean_motion)
            elif self.mean_motion == 0.0:
                self.mean_motion = semi_maj_axis_to_mean_motion(self.semi_maj_axis)

            if self.period == 0.0:
                self.period = (24. * 60.)/self.mean_motion

            self.apogee = (self.semi_maj_axis * (1 + self.eccentricity)) - EARTH_RADIUS_KM
            
            self.perigee = (self.semi_maj_axis * (1 - self.eccentricity)) - EARTH_RADIUS_KM

            try:
                self.revolution_num_at_epoch = float(line2[63:68])
            except Exception as e:
                sys.stdout.write(f"{{str(e)}}: Considering as 0\n")
                self.revolution_num_at_epoch = 0.0
            
            self.parsed_tle = True

    @property
    def julian_date(self):
        return julian_date_from_datetime(self.epoch)

    def to_dict(self):
        """Convert TLE class to dict"""
        line1, line2 = self.to_lines()
        
        return OrderedDict({
            "catalog_num": self.catalog_num,
            "classification": self.classification,
            "launch_year": self.launch_year,
            "launch_num": self.launch_num,
            "launch_piece": self.launch_piece,
            "jdate": self.julian_date,
            "epoch": self.epoch.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch_frac": self.epoch.microsecond,
            "dmm": self.mean_motion_derivative1,
            "dmmm": self.mean_motion_derivative2,
            "bstar": self.bstar,
            "bstar_esign": self.bstar_esign,
            "type": self.elset_type,
            "elem_num": self.element_set_num,
            "i": self.inclination,
            "raan": self.right_ascension_of_ascending_node,
            "e": self.eccentricity,
            "arg_per": self.argument_of_perigee,
            "m_an": self.mean_anomaly,
            "mean_motion": self.mean_motion,
            "rev": self.revolution_num_at_epoch,
            "line1": line1,
            "line2": line2
        })

    def to_satcat(self):
        
        return OrderedDict({
            "IntDes": f"{self.launch_year_full:4d}-{self.launch_num:03d}{self.launch_piece}",
            "CatNum": self.catalog_num,
            "SatName": self.name.upper() if len(self.name) <= 25 else self.name[:25].upper(),
            "Country": self.country,
            "Launch": (datetime(self.launch_year_full,1,1)+timedelta(days=self.launch_doy-1)).strftime("%Y-%m-%d"),
            "Site": self.site.upper(),
            "Decay": self.decay,
            "Period": self.period,
            "Inclination": self.inclination,
            "Apogee": int(round(self.apogee)),
            "Perigee": int(round(self.perigee)),
            "Comment": self.comment_value,
            "CommentCode": self.comment_code,
            "RCSValue": self.rcs_value,
            "RCSSource": self.rcs_source,
            "File": self.file,
            "launch_year": self.launch_year_full,
            "launch_num": self.launch_num,
            "launch_piece": self.launch_piece,
            "current": self.current,
            "size_est": self.size_value
        })

    def to_lines(self,
                 name_line=False):
        """
        Returns tuple of (line1, line2)
        """
        
        # Int designator
        international_designator = f"{self.launch_year % 100:02}{self.launch_num:0>3d}{self.launch_piece:3s}"

        # epoch
        year = self.epoch.year % 100
        day_of_year = self.epoch.timetuple().tm_yday
        fractional_day = (self.epoch.hour * self.epoch.minute/60. + self.epoch.second/3600.
                            + self.epoch.microsecond/1e6) / 24
        epoch_str = f"{year:02d}{day_of_year:03d}.{int(fractional_day*1e8):08d}"[0:14]

        # mean motion terms
        mean_motion_derivative1_str = format_as_bald_decimal_str(self.mean_motion_derivative1)
        mean_motion_derivative2_str = format_as_tle_exp_str(self.mean_motion_derivative2)
        bstar_str = format_as_tle_exp_str(self.bstar)

        line1 = f"1 {self.catalog_num:05d}{self.classification:1s}"+" "+\
            f"{international_designator:<8}"+" "+\
            f"{epoch_str}"+" "+\
            f"{mean_motion_derivative1_str:10s}"+" "+\
            f"{mean_motion_derivative2_str:8s}"+" "+\
            f"{bstar_str:8s}"+" "+\
            f"{self.elset_type:1d}"+" "+\
            f"{self.element_set_num%1000:>4}"
        
        line1_checksum = calc_checksum(line1)
        line1 += f"{line1_checksum:1d}"

        # line 2
        line2 = f"2 {self.catalog_num:05d}"+" "+\
            f"{self.inclination:08.4f}"[:8]+" "+\
            f"{self.right_ascension_of_ascending_node:08.4f}"[:8]+" "+\
            f"{int(self.eccentricity*1e7):07d}"+" "+\
            f"{self.argument_of_perigee:08.4f}"+" "+\
            f"{self.mean_anomaly:08.4f}"+" "+\
            f"{self.mean_motion:11.8f}"+\
            f"{int(self.revolution_num_at_epoch) % 100000:05d}"

        line2_checksum = calc_checksum(line2)
        line2 += f"{line2_checksum:1d}"

        if (name_line
            and self.name
            and type(self.name) in (str,)):
            line0 = f"{self.name.upper()}"
            return [line0, line1, line2]

        return [line1, line2]

    def __repr__(self):
        return f"TLE({self.catalog_num}:{self.orbit_type()}){self.name}"

    def orbit_type(self) -> str:
        """
        Returns obit type string: LEO, HEO, MEO or GEO
        """
        if self.period < 225:
            return "LEO"
        elif self.eccentricity >= 0.3:
            return "HEO"
        elif self.period < 800:
            return "MEO"
        else:
            return "GEO"
        
    def as_text(self):
        return "\n".join(self.to_lines())

    def check_tle(self,
                line1:str,
                line2:str,
                launched_only=True,
                int_cat_num_only=True):
        
        # basic tests
        if line1[0:2] != '1 ' or line2[0:2] != '2 ':
            sys.stdout.write(f"TLE Line Numbers Incorrect: ({line1[0:1]}), ({line2[0:1]})\n")
            raise RuntimeError("TLE Line Numbers incorrect")
            # return False

        if len(line1) < 68 or len(line1) > 70 or len(line2) < 68 or len(line2) > 70:
            sys.stdout.write(f"TLE Length Error: {str(len(line1))}, {str(len(line2))}\n")
            return False

        id1 = line1[2:7]
        id2 = line2[2:7]

        if id1 != id2:
            sys.stdout.write(f"NORAD ID does not match on lines of TLE: {line1[2:7]}, {line2[2:7]}\n")
            return False

        # try:
        #     checksum = float(line2[63:68])
        # except:
        #     sys.stdout.write(f"Failed parsing TLE Rev Number from (line2[63:68])\n")
        #     return False

        # conditional tests
        if int_cat_num_only and any(char.isalpha() for char in id1):
            sys.stdout.write(f"Unrecognized NORAD ID: (line1[2:7])\n")
            return False

        if launched_only:
            try:
                int(line1[11:14].strip())
            except:
                sys.stdout.write(f"Failed parsing launch number for {id1}\n")
                return False

        return True

    def update_epoch(self,
                    julian_day:float=1.0,
                    year:int=-1):
        
        if 0 < year < 100:
            if SATCAT_LAUNCH_START < year < 99:
                year += 1900
            elif 0 <= year < SATCAT_LAUNCH_START:
                year += 2000
        
        day = int(julian_day)
        frac_hours = (julian_day - day)*24.0
        hours = int(frac_hours)
        frac_minutes = (frac_hours - hours)*60.0
        minutes = int(frac_minutes)
        frac_seconds = (frac_minutes - minutes)*60.0
        seconds = int(frac_seconds)
        microseconds = (frac_seconds - seconds)*1e6

        self.epoch = datetime(year=year,
                            month=1,
                            day=1,
                            hour=0,
                            minute=0,
                            second=0,
                            microsecond=0) + \
                    timedelta(days=day-1,
                            hours=hours,
                            minutes=minutes,
                            seconds=seconds,
                            microseconds=microseconds)

if __name__ == "__main__":
    now = datetime.now()
    
    print(julian_date_from_datetime(now, use_astropy=False))
    print(julian_date_from_datetime(now, use_astropy=True))

    a = TLE()
    a.update_epoch(82.544,24)
    print(a.epoch)
    print(julian_date_from_datetime(a.epoch))
    print(semi_maj_axis_to_mean_motion(42166.3))
    print(a.as_text())