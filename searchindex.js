Search.setIndex({"docnames": ["articles/alert_detection", "articles/anomaly_detection", "articles/intro", "authors", "codeofconduct", "contributing", "disclaimer", "history", "index", "installation", "modules", "pynssp", "pynssp.core", "pynssp.detectors", "readme", "usage"], "filenames": ["articles/alert_detection.md", "articles/anomaly_detection.md", "articles/intro.md", "authors.rst", "codeofconduct.rst", "contributing.rst", "disclaimer.rst", "history.rst", "index.rst", "installation.rst", "modules.rst", "pynssp.rst", "pynssp.core.rst", "pynssp.detectors.rst", "readme.md", "usage.rst"], "titles": ["Anomaly Detection", "Anomaly Detection", "Getting started", "Credits", "Creating a Culture of Innovation", "Contributing", "Disclaimer", "Release Notes", "Welcome to pynssp\u2019s documentation!", "Installing pynssp", "pynssp", "pynssp package", "pynssp.core package", "pynssp.detectors package", "About pynssp", "Getting started"], "terms": {"In": [0, 1, 2, 5, 6, 13, 15], "thi": [0, 1, 2, 4, 5, 6, 8, 9, 13, 15], "tutori": [0, 1, 2, 14, 15], "we": [0, 1, 2, 4, 5, 15], "describ": [0, 1, 2, 15], "how": [0, 1, 2, 4, 15], "perform": [0, 1, 6], "trend": [0, 1, 10, 11], "classif": [0, 1], "analysi": [0, 1, 13], "us": [0, 1, 2, 4, 6, 11, 13, 14, 15], "time": [0, 1, 4, 6, 8, 11, 13, 14], "seri": [0, 1, 8, 11, 13, 14], "vignett": [0, 1, 2, 15], "sourc": [0, 1, 4, 5, 8, 11, 12, 13, 14], "cli": [0, 1, 2, 15], "cc": [0, 1], "dd": [0, 1, 2, 15], "coronaviru": [0, 1], "v2": [0, 1, 14], "definit": [0, 1, 13], "limit": [0, 1, 6, 13, 14], "ed": [0, 1], "visit": [0, 1, 13, 14], "ha": [0, 1, 2, 4, 6, 12, 15], "been": [0, 1, 2, 6, 15], "emerg": [0, 1, 13], "ye": [0, 1], "start": [0, 1, 8, 11, 13, 14], "load": [0, 1, 2, 15], "pynssp": [0, 1, 2, 15], "packag": [0, 1, 2, 8, 10, 14, 15], "relat": [0, 1, 4, 5, 14], "depend": [0, 1], "import": [0, 1, 2, 11, 13, 14, 15], "panda": [0, 1, 2, 12, 13, 14, 15], "pd": [0, 1, 2, 13, 14, 15], "addit": [0, 1, 2, 11, 12, 15], "visual": [0, 1], "purpos": [0, 1, 6, 14], "seaborn": [0, 1], "sn": [0, 1], "matplotlib": [0, 1], "pyplot": [0, 1], "plt": [0, 1], "date": [0, 1, 2, 11, 13, 14, 15], "mdate": [0, 1], "next": [0, 1, 2, 15], "creat": [0, 1, 8, 11, 14], "an": [0, 1, 8, 11, 12, 13, 14], "user": [0, 1, 4, 5, 6, 8, 11, 14], "profil": [0, 1, 8, 11, 14], "object": [0, 1, 2, 11, 12, 13, 14, 15], "class": [0, 1, 2, 4, 11, 12, 13, 14, 15], "credenti": [0, 1, 2, 10, 11, 15], "myprofil": [0, 1, 2, 11, 14, 15], "create_profil": [0, 1, 2, 10, 11, 14, 15], "save": [0, 1, 2, 6, 12, 15], "file": [0, 1, 2, 4, 5, 11, 12, 14, 15], "futur": [0, 1, 2, 15], "pickl": [0, 1, 2, 11, 12, 15], "With": [0, 1], "authent": [0, 1, 11], "tabl": [0, 1, 14], "api": [0, 1, 2, 11, 12, 14, 15], "url": [0, 1, 2, 11, 12, 14, 15], "http": [0, 1, 2, 9, 11, 14, 15], "essence2": [0, 1, 8, 11], "syndromicsurveil": [0, 1, 2, 11, 14, 15], "org": [0, 1, 2, 11, 14, 15], "nssp_essenc": [0, 1, 2, 11, 14, 15], "timeseri": [0, 1, 2, 11, 15], "enddat": [0, 1, 2, 11, 14, 15], "20nov20": [0, 1], "ccddcategori": [0, 1, 2, 11, 15], "20cc": [0, 1, 2, 15], "20with": [0, 1, 2, 15], "20cli": [0, 1, 2, 15], "20dd": [0, 1, 2, 15], "20and": [0, 1, 2, 15], "20coronaviru": [0, 1, 2, 15], "20v2": [0, 1, 2, 15], "percentparam": [0, 1, 2, 11, 15], "geographysystem": [0, 1, 2, 11, 15], "hospitaldhhsregion": [0, 1, 2, 15], "datasourc": [0, 1, 2, 11, 15], "va_hospdreg": [0, 1, 2, 15], "probrepswitch": [0, 1, 2, 11, 15], "startdat": [0, 1, 2, 11, 14, 15], "22aug20": [0, 1], "timeresolut": [0, 1, 2, 11, 15], "daili": [0, 1, 2, 11, 13, 15], "hasbeen": [0, 1, 2, 15], "1": [0, 1, 2, 5, 8, 11, 13, 14, 15], "medicalgroupingsystem": [0, 1, 2, 11, 15], "essencesyndrom": [0, 1, 2, 11, 15], "userid": [0, 1, 2, 11, 15], "2362": [0, 1, 2, 15], "aqttarget": [0, 1, 2, 11, 15], "stratval": [0, 1, 2, 15], "multistratv": [0, 1, 2, 15], "geographi": [0, 1, 2, 11, 15], "graphonli": [0, 1, 2, 15], "true": [0, 1, 2, 12, 13, 15], "numseri": [0, 1, 2, 15], "0": [0, 1, 2, 5, 8, 11, 13, 14, 15], "graphopt": [0, 1, 2, 15], "multiplesmal": [0, 1, 2, 15], "seriesperyear": [0, 1, 2, 15], "fals": [0, 1, 2, 11, 12, 15], "nonzerocomposit": [0, 1, 2, 15], "removezeroseri": [0, 1, 2, 15], "startmonth": [0, 1, 2, 15], "januari": [0, 1, 2, 15], "api_data": [0, 1, 11, 14], "get_essence_data": [0, 1, 2, 10, 11, 15], "inspect": [0, 1, 14], "frame": [0, 1, 13], "info": [0, 1, 11], "befor": [0, 1, 4, 5], "appli": [0, 8, 13], "function": [0, 1, 2, 4, 11, 13, 15], "let": [0, 1], "": [0, 1, 4, 5, 6, 11, 13, 14], "first": [0, 1, 3, 7, 13], "group": [0, 1, 13], "hh": [0, 1, 4], "region": [0, 1, 13], "df_hh": [0, 1], "groupbi": [0, 1], "hospitaldhhsregion_displai": [0, 1], "The": [0, 1, 2, 4, 6, 11, 12, 13, 14, 15], "compar": [0, 1, 13], "most": [0, 1, 13], "recent": [0, 1, 13], "count": [0, 1, 2, 11, 13, 15], "baselin": [0, 1, 13], "expect": [0, 1, 4, 5, 13], "For": [0, 1, 2, 4, 13, 14, 15], "test": [0, 1, 13], "give": [0, 1, 5, 12, 13], "influenc": [0, 1, 13], "observ": [0, 1, 13], "algorithm": [0, 1, 13], "i": [0, 1, 2, 4, 5, 6, 12, 13, 14, 15], "appropri": [0, 1, 4, 13], "do": [0, 1, 4, 13], "have": [0, 1, 2, 4, 5, 9, 13, 14, 15], "characterist": [0, 1, 13], "featur": [0, 1, 13], "model": [0, 1, 13], "It": [0, 1, 13, 14], "more": [0, 1, 4, 13, 14], "applic": [0, 1, 13, 14], "depart": [0, 1, 13], "certain": [0, 1, 13], "hospit": [0, 1, 13], "small": [0, 1, 13], "below": [0, 1, 2, 13, 15], "10": [0, 1, 2, 13, 15], "becaus": [0, 1, 13], "case": [0, 1, 2, 11, 13, 15], "chosen": [0, 1, 13], "geograph": [0, 1, 13], "can": [0, 1, 2, 4, 5, 9, 12, 14, 15], "alert_ewma": [0, 1, 11, 13], "run": [0, 1, 2, 9, 15], "help": [0, 1, 4, 8], "your": [0, 1, 2, 4, 9, 14, 15], "python": [0, 1, 2, 9, 14, 15], "consol": [0, 1, 2, 15], "jupyt": [0, 1, 2, 15], "notebook": [0, 1, 2, 15], "jupyterlab": [0, 1], "df_ewma": [0, 1, 13], "t": [0, 1, 4, 5, 9, 13], "y": [0, 1, 13], "datacount": [0, 1], "subset": [0, 1, 11], "datafram": [0, 1, 2, 11, 12, 13, 15], "4": [0, 1, 11, 13], "df_ewma_region": [0, 1], "now": [0, 1], "set": [0, 1, 4], "plot": [0, 1, 2, 11, 12, 15], "grid": [0, 1], "set_styl": [0, 1], "whitegrid": [0, 1], "white": [0, 1], "ax": [0, 1], "size": [0, 1, 4, 13], "axi": [0, 1], "figur": [0, 1], "figsiz": [0, 1], "12": [0, 1, 13], "6": [0, 1, 11], "gca": [0, 1], "set_ylim": [0, 1], "5000": [0, 1], "lineplot": [0, 1], "x": [0, 1], "color": [0, 1, 4], "blue": [0, 1, 13], "linewidth": [0, 1], "scatterplot": [0, 1], "hue": [0, 1], "alert": [0, 1, 8, 13, 14], "palett": [0, 1], "black": [0, 1], "darkorang": [0, 1], "red": [0, 1, 13], "alpha": [0, 1], "format": [0, 1, 2, 12, 15], "yaxi": [0, 1], "set_major_formatt": [0, 1], "funcformatt": [0, 1], "lambda": [0, 1], "loc": [0, 1], "int": [0, 1], "xaxi": [0, 1], "set_major_loc": [0, 1], "weekdayloc": [0, 1], "gcf": [0, 1], "autofmt_xd": [0, 1], "label": [0, 1], "set_xlabel": [0, 1], "set_ylabel": [0, 1], "custom": [0, 1], "legend": [0, 1], "titl": [0, 1], "lower": [0, 1], "center": [0, 1, 4, 6], "bbox_to_anchor": [0, 1], "5": [0, 1, 13], "ncol": [0, 1], "frameon": [0, 1], "framealpha": [0, 1], "edgecolor": [0, 1], "borderpad": [0, 1], "75": [0, 1, 2, 15], "get_text": [0, 1], "set_text": [0, 1], "No": [0, 1], "none": [0, 1, 3, 11, 12], "2": [0, 1, 2, 11, 13, 14, 15], "warn": [0, 1, 13], "3": [0, 1, 2, 11, 13, 15], "remov": [0, 1, 4, 13], "top": [0, 1, 11], "right": [0, 1, 4, 5, 6, 14], "spine": [0, 1], "despin": [0, 1], "fit": [0, 1, 6, 13, 14], "linear": [0, 1, 13], "percentag": [0, 1, 13], "forecast": [0, 1, 13], "predict": [0, 1, 13], "valu": [0, 1, 11, 12, 13], "follow": [0, 1, 2, 4, 13, 15], "pre": [0, 1], "defin": [0, 1, 2, 4, 12, 13, 15], "buffer": [0, 1], "period": [0, 1, 13], "includ": [0, 1, 4, 6, 13, 14], "term": [0, 1, 13, 14], "account": [0, 1, 4, 13], "dai": [0, 1, 5, 13, 14], "week": [0, 1, 11, 13], "effect": [0, 1, 13], "implement": [0, 1, 13], "doe": [0, 1, 4, 13], "NOT": [0, 1, 13], "holidai": [0, 1, 13], "alert_regress": [0, 1, 11, 13], "df_regress": [0, 1, 13], "filter": [0, 1], "df_regression_region": [0, 1], "gener": [0, 1, 13], "where": [0, 1, 4, 13], "adjust": [0, 1, 13], "r": [0, 1, 2, 4, 13, 14, 15], "squar": [0, 1, 13], "examin": [0, 1, 13], "see": [0, 1, 13, 14], "meet": [0, 1, 4, 5, 13], "threshold": [0, 1, 13], "60": [0, 1, 13], "If": [0, 1, 4, 5, 9, 11, 13, 14], "met": [0, 1, 13], "consid": [0, 1, 4, 5, 13, 14], "explain": [0, 1, 2, 13, 15], "well": [0, 1, 4, 13], "which": [0, 1, 13], "sparser": [0, 1, 13], "ar": [0, 1, 2, 5, 12, 13, 14, 15], "common": [0, 1, 13], "granular": [0, 1], "level": [0, 1, 8, 11, 13], "alert_switch": [0, 1, 11, 13], "df_switch": [0, 1, 13], "df_switch_region": [0, 1], "intend": [0, 1, 13], "weekli": [0, 1, 13], "span": [0, 1, 13], "year": [0, 1, 11, 13], "cyclic": [0, 1], "sine": [0, 1, 13], "cosin": [0, 1, 13], "inclus": [0, 1, 13], "season": [0, 1, 13], "multi": [0, 1, 13], "syndrom": [0, 1, 13, 14], "diseas": [0, 1, 4, 6, 13], "influenza": [0, 1, 13], "rsv": [0, 1, 13], "noroviru": [0, 1, 13], "each": [0, 1, 4, 13], "make": [0, 1, 4, 13], "all": [0, 1, 4, 5, 6, 13, 14], "One": [0, 1, 13], "side": [0, 1, 13], "upper": [0, 1, 13], "95": [0, 1, 13], "interv": [0, 1, 13], "bound": [0, 1, 13], "comput": [0, 1, 13], "alarm": [0, 1, 13], "signal": [0, 1, 13], "ani": [0, 1, 4, 5, 6, 13, 14], "dure": [0, 1, 13], "exce": [0, 1], "alert_nbinom": [0, 1, 11, 13], "exampl": [0, 1, 2, 11, 13, 14, 15], "our": [0, 1, 4, 5], "synthet": [0, 1], "scenario": [0, 1, 11], "synthes": [0, 1], "scenario1": [0, 1], "synth_ts1": [0, 1], "get_scenario1": [0, 1, 10, 11], "head": [0, 1, 2, 13, 14, 15], "df_nbinom": [0, 1, 13], "baseline_end": [0, 1, 13], "2021": [0, 1], "26": [0, 1], "fig": [0, 1], "subplot": [0, 1], "1a476f": [0, 1], "a50026": [0, 1], "add": [0, 1], "vertic": [0, 1], "line": [0, 1, 4, 5, 8], "axvlin": [0, 1], "to_datetim": [0, 1], "text": [0, 1], "annot": [0, 1], "2019": [0, 1], "01": [0, 1, 13], "left": [0, 1], "va": [0, 1], "bottom": [0, 1], "yearloc": [0, 1], "dateformatt": [0, 1], "tick": [0, 1], "paramet": [0, 1, 2, 11, 12, 13, 15], "tick_param": [0, 1], "both": [0, 1, 2, 4, 5, 15], "length": [0, 1, 13], "width": [0, 1], "theme": [0, 1], "ideal": [0, 1, 13], "fourier": [0, 1, 13], "approach": [0, 1, 13], "method": [0, 1, 2, 11, 12, 13, 14, 15], "between": [0, 1, 5, 13], "octob": [0, 1, 13], "mai": [0, 1, 2, 5, 6, 9, 13, 14, 15], "end": [0, 1, 11, 13, 14], "epidem": [0, 1, 2, 13, 15], "prior": [0, 1, 13], "alert_serfl": [0, 1, 11, 13], "same": [0, 1, 2, 15], "simul": [0, 1, 11], "previou": [0, 1], "df_serfl": [0, 1, 13], "goal": [2, 15], "facilit": [2, 15], "access": [2, 4, 6, 15], "electron": [2, 4, 15], "surveil": [2, 5, 14, 15], "system": [2, 15], "earli": [2, 15], "notif": [2, 15], "commun": [2, 4, 14, 15], "base": [2, 4, 12, 15], "via": [2, 12, 15], "secur": [2, 4, 11, 15], "simplifi": [2, 15], "interfac": [2, 15], "provid": [2, 4, 6, 11, 15], "streamlin": [2, 15], "pull": [2, 8, 14, 15], "abstract": [2, 15], "complex": [2, 15], "code": [2, 4, 5, 6, 14, 15], "variou": [2, 15], "png": [2, 11, 12, 15], "imag": [2, 11, 15], "util": [2, 10, 14, 15], "step": [2, 15], "here": [2, 15], "abov": [2, 4, 13, 15], "need": [2, 4, 15], "execut": [2, 15], "onli": [2, 4, 6, 14, 15], "onc": [2, 9, 15], "upon": [2, 15], "prompt": [2, 11, 15], "hi": [2, 15], "usernam": [2, 11, 12, 14, 15], "password": [2, 4, 11, 12, 14, 15], "come": [2, 15], "get_api_respons": [2, 10, 11, 12, 15], "get_api_data": [2, 10, 11, 12, 14, 15], "get_api_graph": [2, 10, 11, 12, 15], "get_api_tsgraph": [2, 15], "altern": [2, 9, 14, 15], "serv": [2, 14, 15], "wrapper": [2, 15], "respect": [2, 15], "section": [2, 15], "show": [2, 11, 12, 15], "seven": [2, 5, 15], "9feb2021": [2, 15], "medicalgroup": [2, 15], "injuri": [2, 15], "noperc": [2, 11, 15], "detector": [2, 8, 10, 11, 15], "11nov2020": [2, 15], "455": [2, 15], "api_data_t": [2, 15], "column": [2, 11, 13, 14, 15], "extract": [2, 14, 15], "embed": [2, 15], "json_norm": [2, 14, 15], "timeseriesdata": [2, 15], "preview": [2, 15], "achiev": [2, 15], "outcom": [2, 15], "directli": [2, 15], "when": [2, 4, 15], "retriev": [2, 11, 15], "insert": [2, 15], "graphtitl": [2, 15], "nation": [2, 4, 14, 15], "20": [2, 15], "20injuri": [2, 15], "20syndrom": [2, 15], "20daili": [2, 15], "20count": [2, 15], "xaxislabel": [2, 15], "yaxislabel": [2, 15], "api_data_graph": [2, 15], "check": [2, 15], "type": [2, 4, 11, 12, 13, 15], "print": [2, 4, 15], "locat": [2, 15], "variabl": [2, 13, 15], "apigraph": [2, 11, 12, 15], "interact": [2, 4, 15], "mode": [2, 15], "call": [2, 4, 15], "contain": [2, 4, 10, 11, 13, 14, 15], "exaust": [2, 15], "pleas": [2, 5, 14, 15], "document": [2, 4, 15], "tabular": [2, 15], "seen": [2, 15], "other": [2, 4, 5, 14, 15], "hand": [2, 15], "long": [2, 15], "pivot": [2, 15], "subsect": [2, 15], "demonstr": [2, 15], "tablebuild": [2, 15], "31dec2020": [2, 15], "cdc": [2, 4, 5, 6, 14, 15], "20opioid": [2, 15], "20overdos": [2, 15], "20v3": [2, 15], "nodetectordetector": [2, 15], "1oct2020": [2, 15], "agench": [2, 15], "11": [2, 15], "14": [2, 15], "15": [2, 15], "24": [2, 11, 15], "25": [2, 13, 15], "34": [2, 15], "35": [2, 15], "44": [2, 15], "45": [2, 15], "54": [2, 15], "55": [2, 15], "64": [2, 15], "65": [2, 15], "74": [2, 15], "84": [2, 15], "85": [2, 15], "1000": [2, 15], "unknown": [2, 15], "monthli": [2, 15], "rowfield": [2, 15], "geographyhospitaldhhsregion": [2, 15], "columnfield": [2, 15], "api_data_tb_csv": [2, 15], "fromcsv": [2, 11, 12, 15], "while": [2, 15], "equal": [2, 4, 13, 15], "retrun": [2, 15], "advantag": [2, 15], "requir": [2, 4, 15], "extra": [2, 15], "pars": [2, 15], "second": [2, 15], "api_data_tb_json": [2, 15], "similarli": [2, 15], "output": [2, 15], "datadetail": [2, 15], "20i": [2, 15], "31jan2021": [2, 14, 15], "api_data_dd_csv": [2, 15], "api_data_dd_json": [2, 15], "summarydata": [2, 15], "va_hosp": [2, 11, 15], "29jan2021": [2, 14, 15], "api_data_ss": [2, 15], "sinc": [2, 15], "programmat": [2, 15], "two": [2, 15], "regionsyndromealert": [2, 14, 15], "end_dat": [2, 11, 14, 15], "start_dat": [2, 11, 14, 15], "api_data_alr": [2, 15], "hospitalsyndromealert": [2, 15], "api_data_alh": [2, 15], "februari": [2, 15], "2023": [2, 8, 15], "avail": [2, 4, 6, 14, 15], "therefor": [2, 14, 15], "differ": [2, 4, 15], "one": [2, 15], "you": [2, 4, 5, 6, 9, 14, 15], "recreat": [2, 15], "anoth": [2, 15], "assum": [2, 4, 15], "20pneumonia": [2, 15], "20ccdd": [2, 15], "20v1": [2, 15], "api_data_tssh": [2, 15], "gbedegnon": 3, "roser": 3, "azondekon": 3, "yet": [3, 7], "why": 3, "aspir": 4, "peopl": [4, 5], "work": [4, 5, 14], "joyfulli": 4, "openli": 4, "about": 4, "thing": 4, "matter": 4, "great": 4, "servic": [4, 6], "global": 4, "would": [4, 5], "like": [4, 5, 6], "team": [4, 5], "govern": [4, 6, 14], "privat": 4, "sector": 4, "reflect": 4, "divers": 4, "kind": [4, 6], "just": 4, "protect": [4, 6, 14], "law": [4, 14], "foster": [4, 5], "creativ": 4, "perspect": 4, "solut": 4, "challeng": 4, "face": 4, "conduct": [4, 5, 6, 14], "adapt": [4, 13], "from": [4, 5, 8, 11, 13, 14], "18f": 4, "employ": 4, "opportun": [4, 5], "everyon": 4, "adher": 4, "gsa": 4, "anti": 4, "harras": 4, "polici": [4, 5, 14], "even": [4, 14], "thei": 4, "control": [4, 6], "prevent": [4, 6], "everi": 4, "mention": 4, "what": 4, "encourag": [4, 5, 14], "support": 4, "room": 4, "quieter": 4, "voic": 4, "contribut": [4, 8], "offer": 4, "enthusiasm": 4, "idea": 4, "leverag": 4, "low": 4, "cost": 4, "experiment": 4, "colleagu": 4, "take": 4, "care": 4, "acknowledg": 4, "origin": [4, 13], "look": [4, 5], "wai": [4, 5], "collabor": [4, 5, 14], "situat": 4, "normal": 4, "wouldn": 4, "share": [4, 6], "knowledg": 4, "skill": 4, "priorit": 4, "input": 4, "those": 4, "who": 4, "tradition": 4, "exclud": 4, "civic": 4, "process": [4, 5, 9], "understand": 4, "must": [4, 5, 6, 13], "complet": 4, "awar": 4, "record": [4, 8], "manag": [4, 8], "train": [4, 13], "annual": [4, 13], "order": [4, 5, 13], "compli": [4, 5, 14], "latest": 4, "also": [4, 5], "inform": [4, 6, 13, 14], "resourc": 4, "store": [4, 6, 12], "sensit": [4, 6, 13, 14], "data": [4, 6, 8, 10, 12, 13, 14], "health": [4, 6], "statu": [4, 6], "provis": [4, 6], "payment": [4, 6], "healthcar": [4, 6], "pii": [4, 6], "etc": [4, 6], "under": [4, 5, 6, 14], "circumst": [4, 6], "knowingli": 4, "conceal": 4, "falsifi": 4, "non": [4, 6, 11, 12, 13, 14], "publicli": [4, 6, 14], "up": [4, 6], "steward": 4, "reserv": [4, 6], "moder": [4, 6], "own": 4, "safe": 4, "space": [4, 5], "list": [4, 8], "instanc": [4, 13], "guid": [4, 9], "think": 4, "through": [4, 5, 9, 14], "respond": [4, 5], "experi": 4, "whether": [4, 6, 11, 12, 13], "direct": [4, 6], "ourselv": 4, "unsur": 4, "someth": [4, 5], "probabl": 4, "person": [4, 6], "them": [4, 5], "impact": 4, "than": [4, 13], "intent": 4, "ensur": 4, "neg": [4, 13], "respons": [4, 11, 12, 13], "problem": 4, "usual": 4, "aris": [4, 6], "norm": 4, "offens": 4, "remark": 4, "race": 4, "religion": 4, "sex": 4, "ag": 4, "disabl": 4, "genetr": 4, "sexual": 4, "orient": 4, "gender": 4, "ident": 4, "parent": 4, "maritu": 4, "polit": 4, "affili": 4, "express": [4, 6], "mental": 4, "ill": 4, "socioeconom": 4, "neuro": 4, "typic": 4, "physic": 4, "appear": 4, "bodi": 4, "cloth": 4, "attent": 4, "feel": [4, 5], "alien": 4, "sustain": 4, "disrupt": 4, "talk": 4, "discuss": [4, 5], "chatroom": 4, "patron": 4, "languag": [4, 5], "aggres": 4, "unconstruct": 4, "critic": 4, "correct": 4, "improv": 4, "convers": 4, "sometim": 4, "refer": 4, "actual": 4, "repeatedli": 4, "interrupt": 4, "over": 4, "someon": 4, "els": [4, 6], "feign": 4, "surpris": 4, "lack": 4, "topic": 4, "subtl": 4, "prejudic": 4, "misidentifi": 4, "reject": 4, "valid": 4, "incorrect": 4, "pronoun": 4, "form": 4, "address": [4, 5], "misgend": 4, "retali": 4, "against": 4, "anyon": [4, 6, 14], "formal": 4, "complaint": 4, "violat": 4, "scientif": 4, "clearanc": 4, "obtain": 4, "approv": 4, "offici": 4, "product": 4, "releas": [4, 5, 8, 14], "public": [4, 8, 9], "extern": 4, "partner": 4, "oral": 4, "materi": [4, 6, 14], "employe": 4, "author": [4, 14], "co": 4, "publish": [4, 14], "outsid": 4, "contractor": 4, "develop": [4, 8, 14], "content": [4, 5, 10], "behalf": 4, "put": 4, "project": [4, 5, 8], "blog": 4, "wiki": 4, "forum": 4, "bug": [4, 14], "track": 4, "site": [4, 14], "deem": 4, "necessari": [4, 5], "individu": 4, "within": [4, 5, 14], "dissemin": 4, "qualiti": 4, "agenc": 4, "allow": 4, "real": 4, "among": 4, "wa": [4, 13], "mainli": 4, "informat": 4, "unit": [4, 5, 6, 14], "d": 4, "lab": 4, "enforc": 4, "commiss": 4, "discrimin": 4, "prohibit": 4, "new": [4, 11, 14], "propos": 4, "regul": 4, "thank": 5, "question": 5, "doubt": 5, "don": [5, 9], "afraid": 5, "send": 5, "appreci": 5, "forward": 5, "transpar": 5, "environ": 5, "read": 5, "licens": [5, 8], "readm": 5, "found": 5, "inquiri": 5, "answer": 5, "free": [5, 14], "contact": 5, "u": [5, 6], "state": [5, 6, 14], "copyright": [5, 14], "worldwid": [5, 14], "waiv": [5, 14], "cc0": [5, 14], "univers": [5, 14], "dedic": [5, 14], "By": [5, 14], "agre": [5, 14], "waiver": [5, 14], "interest": [5, 14], "merg": 5, "design": 5, "suggest": 5, "affect": 5, "off": 5, "abid": [5, 6], "specif": 5, "member": 5, "soon": 5, "possibl": [5, 6], "Be": [5, 8], "least": [5, 13], "old": 5, "held": 5, "longer": 5, "assess": 5, "receiv": [5, 12, 14], "major": 5, "associ": 5, "signific": 5, "dissent": 5, "plan": 5, "action": [5, 6], "identifi": 6, "phi": 6, "administr": 6, "cannot": 6, "subject": [6, 14], "revok": 6, "embodi": 6, "softwar": [6, 14], "without": [6, 14], "warranti": [6, 14], "impli": [6, 14], "otherwis": 6, "particular": [6, 14], "event": [6, 13], "shall": 6, "liabl": 6, "special": 6, "incident": 6, "indirect": 6, "consequenti": 6, "damag": 6, "whatsoev": 6, "loss": 6, "profit": 6, "revenu": 6, "claim": 6, "third": 6, "parti": 6, "advis": 6, "howev": 6, "caus": 6, "theori": 6, "liabil": 6, "out": 6, "connect": 6, "possess": 6, "pypi": 7, "upcom": 7, "instal": 8, "usag": [8, 11], "get": [8, 12], "domain": 8, "standard": [8, 12, 13], "notic": 8, "privaci": 8, "version": [8, 14], "introduct": 8, "nssp": [8, 11, 13, 14], "graph": [8, 11, 12], "essenc": [8, 11, 13, 14], "builder": 8, "result": [8, 13], "detail": [8, 14], "summari": 8, "stat": 8, "detect": [8, 13], "stratifi": 8, "histor": 8, "anomali": [8, 13], "note": [8, 13], "credit": 8, "lead": 8, "contributor": 8, "disclaim": [8, 14], "cultur": 8, "innov": 8, "empow": 8, "rule": 8, "behavior": 8, "boundari": 8, "background": 8, "relev": 8, "legal": 8, "consider": [8, 13], "index": 8, "search": 8, "page": [8, 14], "pip": [9, 14], "command": 9, "termin": 9, "git": [9, 14], "github": [9, 14], "com": [9, 11, 14], "cdcgov": [9, 14], "download": 9, "repo": 9, "either": 9, "clone": 9, "repositori": [9, 14], "Or": 9, "tarbal": 9, "curl": 9, "ojl": 9, "master": 9, "copi": [9, 14], "setup": 9, "py": 9, "subpackag": 10, "core": [10, 11], "submodul": 10, "constant": [10, 11], "modul": 10, "token": [10, 11, 14], "ewma": [10, 11], "farrington": [10, 11], "nbinom": [10, 11], "regress": [10, 11], "serfl": [10, 11], "switch": [10, 11], "get_scenario2": [10, 11], "load_nssp_stopword": [10, 11], "load_simulated_t": [10, 11], "change_d": [10, 11, 14], "create_token_profil": [10, 11, 14], "nsspcontain": [11, 12], "ewma_loop": [11, 13], "nb_model": [11, 13], "adaptive_regress": [11, 13], "serfling_model": [11, 13], "return": [11, 12, 13], "scenario1_t": 11, "scenario2_t": 11, "curat": 11, "stopword": 11, "null": 11, "dtype": 11, "word": 11, "835": 11, "int64": 11, "memori": 11, "13": 11, "kb": 11, "simulated_t": 11, "626": 11, "id": 11, "main": 11, "chang": 11, "given": 11, "str": 11, "A": [11, 12, 13, 14], "replac": 11, "exist": 11, "default": [11, 12, 13], "modifi": [11, 14], "01jan2022": 11, "31dec2022": 11, "01jan2021": 11, "31dec2021": 11, "string": [11, 12, 13], "repres": [11, 12, 13], "enter": 11, "access_token": [11, 12], "bearer": [11, 12], "auth_typ": 11, "mytokenprofil": 11, "encod": [11, 12], "utf": [11, 12], "8": [11, 12], "kwarg": [11, 12], "endpoint": 11, "boolean": 11, "indic": [11, 13], "should": [11, 14], "csv": [11, 12], "keyword": [11, 12], "argument": [11, 12], "pass": [11, 12], "httpbin": 11, "json": [11, 12, 14], "file_ext": [11, 12], "extens": [11, 12], "api_graph": 11, "25jun2022": 11, "25jun2021": 11, "3751": 11, "hospitalregion": 11, "path": 12, "displai": 12, "encapsul": 12, "kei": 12, "logic": [12, 13], "read_csv": 12, "empti": 12, "charact": 12, "vector": [12, 13], "pkl": 12, "df": 13, "b": 13, "28": 13, "g": 13, "w1": 13, "w2": 13, "9": 13, "exponenti": 13, "weight": 13, "move": 13, "averag": 13, "statist": 13, "student": 13, "yield": 13, "p": 13, "less": 13, "greater": 13, "strictli": 13, "05": 13, "yellow": 13, "occur": 13, "grei": 13, "did": 13, "e": 13, "were": 13, "unavail": 13, "name": 13, "numer": 13, "number": 13, "calcul": 13, "roll": 13, "deviat": 13, "match": 13, "guardband": 13, "separ": 13, "current": 13, "smooth": 13, "coeffici": 13, "gradual": 13, "recommend": 13, "sudden": 13, "7": 13, "approxim": 13, "c2": 13, "numpi": 13, "np": 13, "date_rang": 13, "2020": 13, "31": 13, "random": 13, "randint": 13, "101": 13, "366": 13, "loop": 13, "outli": 13, "mu": 13, "include_tim": 13, "binomi": 13, "fall": 13, "estim": 13, "binari": 13, "field": 13, "2014": 13, "2022": 13, "02": 13, "freq": 13, "w": 13, "poisson": 13, "lam": 13, "len": 13, "03": 13, "helper": 13, "monitor": 13, "multipl": 13, "liner": 13, "later": [13, 14], "guard": 13, "band": 13, "divid": 13, "error": 13, "feder": [13, 14], "counti": 13, "fix": 13, "analyz": 13, "program": 14, "its": 14, "practic": 14, "collect": 14, "advanc": 14, "rnssp": 14, "datetim": 14, "timedelta": 14, "updat": 14, "todai": 14, "30": 14, "structur": 14, "glimps": 14, "dataset": 14, "want": 14, "submit": 14, "push": 14, "request": 14, "ad": 14, "gazondekon": 14, "gov": 14, "encount": 14, "clear": 14, "email": 14, "issu": 14, "minim": 14, "reproduc": 14, "constitut": 14, "domest": 14, "17": 14, "usc": 14, "105": 14, "apach": 14, "asl": 14, "redistribut": 14, "option": 14, "distribut": 14, "hope": 14, "merchant": 14, "FOR": 14, "along": 14, "www": 14, "html": 14, "fork": 14, "open": 14, "inherit": 14, "particip": 14, "cover": 14, "might": 14, "basic": 14, "grant": 14, "world": 14, "wide": 14, "royalti": 14, "perpetu": 14, "irrevoc": 14, "exclus": 14, "transfer": 14, "comment": 14, "messag": 14, "submiss": 14, "act": 14, "archiv": 14, "learn": 14, "increas": 14, "potenti": 14, "web": 14}, "objects": {"": [[11, 0, 0, "-", "pynssp"]], "pynssp": [[12, 0, 0, "-", "core"], [11, 0, 0, "-", "data"], [13, 0, 0, "-", "detectors"], [11, 0, 0, "-", "pynssp"], [11, 0, 0, "-", "utils"]], "pynssp.core": [[12, 0, 0, "-", "constants"], [12, 0, 0, "-", "container"], [12, 0, 0, "-", "credentials"], [12, 0, 0, "-", "token"]], "pynssp.core.container": [[12, 1, 1, "", "APIGraph"], [12, 1, 1, "", "NSSPContainer"]], "pynssp.core.container.APIGraph": [[12, 2, 1, "", "plot"], [12, 2, 1, "", "show"]], "pynssp.core.credentials": [[12, 1, 1, "", "Credentials"]], "pynssp.core.credentials.Credentials": [[12, 2, 1, "", "get_api_data"], [12, 2, 1, "", "get_api_graph"], [12, 2, 1, "", "get_api_response"], [12, 2, 1, "", "pickle"]], "pynssp.core.token": [[12, 1, 1, "", "Token"]], "pynssp.core.token.Token": [[12, 2, 1, "", "get_api_data"], [12, 2, 1, "", "get_api_graph"], [12, 2, 1, "", "get_api_response"], [12, 2, 1, "", "pickle"]], "pynssp.data": [[11, 3, 1, "", "get_scenario1"], [11, 3, 1, "", "get_scenario2"], [11, 3, 1, "", "load_nssp_stopwords"], [11, 3, 1, "", "load_simulated_ts"]], "pynssp.detectors": [[13, 0, 0, "-", "ewma"], [13, 0, 0, "-", "farrington"], [13, 0, 0, "-", "nbinom"], [13, 0, 0, "-", "regression"], [13, 0, 0, "-", "serfling"], [13, 0, 0, "-", "switch"], [13, 0, 0, "-", "trend"]], "pynssp.detectors.ewma": [[13, 3, 1, "", "alert_ewma"], [13, 3, 1, "", "ewma_loop"]], "pynssp.detectors.nbinom": [[13, 3, 1, "", "alert_nbinom"], [13, 3, 1, "", "nb_model"]], "pynssp.detectors.regression": [[13, 3, 1, "", "adaptive_regression"], [13, 3, 1, "", "alert_regression"]], "pynssp.detectors.serfling": [[13, 3, 1, "", "alert_serfling"], [13, 3, 1, "", "serfling_model"]], "pynssp.detectors.switch": [[13, 3, 1, "", "alert_switch"]], "pynssp.utils": [[11, 3, 1, "", "change_dates"], [11, 3, 1, "", "create_profile"], [11, 3, 1, "", "create_token_profile"], [11, 3, 1, "", "get_api_data"], [11, 3, 1, "", "get_api_graph"], [11, 3, 1, "", "get_api_response"], [11, 3, 1, "", "get_essence_data"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"]}, "titleterms": {"anomali": [0, 1], "detect": [0, 1, 2, 15], "introduct": [0, 1, 2, 15], "data": [0, 1, 2, 11, 15], "pull": [0, 1, 5], "from": [0, 1, 2, 9, 15], "nssp": [0, 1, 2, 15], "essenc": [0, 1, 2, 15], "exponenti": [0, 1], "weight": [0, 1], "move": [0, 1], "averag": [0, 1], "ewma": [0, 1, 13], "adapt": [0, 1], "multipl": [0, 1], "regress": [0, 1, 13], "switch": [0, 1, 13], "neg": [0, 1], "binomi": [0, 1], "origin": [0, 1], "serfl": [0, 1, 13], "detector": [0, 1, 13], "appli": 1, "get": [2, 14, 15], "start": [2, 15], "creat": [2, 4, 15], "an": [2, 5, 15], "user": [2, 15], "profil": [2, 15], "time": [2, 15], "seri": [2, 15], "tabl": [2, 8, 15], "graph": [2, 15], "builder": [2, 15], "result": [2, 15], "csv": [2, 15], "option": [2, 15], "json": [2, 15], "detail": [2, 15], "line": [2, 15], "level": [2, 15], "summari": [2, 15], "stat": [2, 15], "alert": [2, 15], "list": [2, 15], "patient": [2, 15], "region": [2, 15], "hospit": [2, 15], "stratifi": [2, 15], "histor": [2, 15], "essence2": [2, 15], "credit": [3, 4], "develop": [3, 9], "lead": 3, "contributor": 3, "cultur": 4, "innov": 4, "Be": 4, "empow": 4, "rule": 4, "behavior": 4, "boundari": 4, "here": 4, "ar": 4, "some": 4, "exampl": 4, "unaccept": 4, "background": 4, "relev": 4, "legal": 4, "consider": 4, "contribut": [5, 14], "welcom": [5, 8], "public": [5, 14], "domain": [5, 14], "request": 5, "chang": 5, "open": 5, "issu": 5, "repositori": 5, "submit": 5, "disclaim": 6, "releas": 7, "note": 7, "0": 7, "1": 7, "2023": 7, "pynssp": [8, 9, 10, 11, 12, 13, 14], "": 8, "document": 8, "overview": [8, 14], "hand": 8, "On": 8, "tutori": 8, "modul": [8, 11, 12, 13], "about": [8, 14], "indic": 8, "instal": [9, 14], "version": 9, "sourc": 9, "packag": [11, 12, 13], "subpackag": 11, "submodul": [11, 12, 13], "util": 11, "content": [11, 12, 13], "core": 12, "constant": 12, "contain": 12, "credenti": 12, "token": 12, "farrington": 13, "nbinom": 13, "trend": 13, "usag": 14, "thi": 14, "project": 14, "help": 14, "standard": 14, "notic": 14, "licens": 14, "privaci": 14, "record": 14, "manag": 14}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.viewcode": 1, "sphinx": 57}, "alltitles": {"Anomaly Detection": [[0, "anomaly-detection"], [0, "id1"], [1, "anomaly-detection"]], "Introduction": [[0, "introduction"], [1, "introduction"], [2, "introduction"], [15, "introduction"]], "Data Pull from NSSP-ESSENCE": [[0, "data-pull-from-nssp-essence"], [1, "data-pull-from-nssp-essence"]], "Exponentially Weighted Moving Average (EWMA)": [[0, "exponentially-weighted-moving-average-ewma"], [1, "exponentially-weighted-moving-average-ewma"]], "Adaptive Multiple Regression": [[0, "adaptive-multiple-regression"], [1, "adaptive-multiple-regression"]], "Regression/EWMA Switch": [[0, "regression-ewma-switch"], [1, "regression-ewma-switch"]], "Negative Binomial Regression": [[0, "negative-binomial-regression"], [1, "negative-binomial-regression"]], "Original Serfling detector": [[0, "original-serfling-detector"], [1, "original-serfling-detector"]], "Applying Detectors": [[1, "applying-detectors"]], "Getting started": [[2, "getting-started"], [15, "getting-started"]], "Creating an NSSP user profile": [[2, "creating-an-nssp-user-profile"], [15, "creating-an-nssp-user-profile"]], "Time Series Data Table": [[2, "time-series-data-table"], [15, "time-series-data-table"]], "Time Series Graph from ESSENCE": [[2, "time-series-graph-from-essence"], [15, "time-series-graph-from-essence"]], "Table Builder Results": [[2, "table-builder-results"], [15, "table-builder-results"]], "CSV option": [[2, "csv-option"], [2, "id1"], [15, "csv-option"], [15, "id1"]], "JSON option": [[2, "json-option"], [2, "id2"], [15, "json-option"], [15, "id2"]], "Data Details (line level)": [[2, "data-details-line-level"], [15, "data-details-line-level"]], "Summary Stats": [[2, "summary-stats"], [15, "summary-stats"]], "Alert List Detection Table": [[2, "alert-list-detection-table"], [15, "alert-list-detection-table"]], "Alert List Detection Table by Patient Region": [[2, "alert-list-detection-table-by-patient-region"], [15, "alert-list-detection-table-by-patient-region"]], "Alert List Detection Table by Hospital Region": [[2, "alert-list-detection-table-by-hospital-region"], [15, "alert-list-detection-table-by-hospital-region"]], "Time series data table with stratified, historical alerts (from ESSENCE2)": [[2, "time-series-data-table-with-stratified-historical-alerts-from-essence2"], [15, "time-series-data-table-with-stratified-historical-alerts-from-essence2"]], "Credits": [[3, "credits"]], "Development Lead": [[3, "development-lead"]], "Contributors": [[3, "contributors"]], "Creating a Culture of Innovation": [[4, "creating-a-culture-of-innovation"]], "Be Empowering": [[4, "be-empowering"]], "Rules of Behavior": [[4, "rules-of-behavior"]], "Boundaries": [[4, "boundaries"]], "Here are some examples of unacceptable behavior": [[4, "here-are-some-examples-of-unacceptable-behavior"]], "Background": [[4, "background"]], "Credit": [[4, "credit"]], "Relevant Legal Considerations": [[4, "relevant-legal-considerations"]], "Contributing": [[5, "contributing"]], "Welcome!": [[5, "welcome"]], "Public Domain": [[5, "public-domain"]], "Requesting Changes": [[5, "requesting-changes"]], "Open an issue in the repository": [[5, "open-an-issue-in-the-repository"]], "Submit a pull request": [[5, "submit-a-pull-request"]], "Disclaimer": [[6, "disclaimer"]], "Release Notes": [[7, "release-notes"]], "0.1.0 (2023-??-??)": [[7, "id1"]], "Welcome to pynssp\u2019s documentation!": [[8, "welcome-to-pynssp-s-documentation"]], "Overview:": [[8, null]], "Hands-On Tutorials:": [[8, null]], "Modules:": [[8, null]], "About:": [[8, null]], "Indices and tables": [[8, "indices-and-tables"]], "Installing pynssp": [[9, "installing-pynssp"]], "Development version": [[9, "development-version"]], "From sources": [[9, "from-sources"]], "pynssp": [[10, "pynssp"]], "pynssp package": [[11, "pynssp-package"]], "Subpackages": [[11, "subpackages"]], "Submodules": [[11, "submodules"], [12, "submodules"], [13, "submodules"]], "pynssp.data module": [[11, "module-pynssp.data"]], "pynssp.pynssp module": [[11, "module-pynssp.pynssp"]], "pynssp.utils module": [[11, "module-pynssp.utils"]], "Module contents": [[11, "module-pynssp"], [12, "module-pynssp.core"], [13, "module-pynssp.detectors"]], "pynssp.core package": [[12, "pynssp-core-package"]], "pynssp.core.constants module": [[12, "module-pynssp.core.constants"]], "pynssp.core.container module": [[12, "module-pynssp.core.container"]], "pynssp.core.credentials module": [[12, "module-pynssp.core.credentials"]], "pynssp.core.token module": [[12, "module-pynssp.core.token"]], "pynssp.detectors package": [[13, "pynssp-detectors-package"]], "pynssp.detectors.ewma module": [[13, "module-pynssp.detectors.ewma"]], "pynssp.detectors.farrington module": [[13, "module-pynssp.detectors.farrington"]], "pynssp.detectors.nbinom module": [[13, "module-pynssp.detectors.nbinom"]], "pynssp.detectors.regression module": [[13, "module-pynssp.detectors.regression"]], "pynssp.detectors.serfling module": [[13, "module-pynssp.detectors.serfling"]], "pynssp.detectors.switch module": [[13, "module-pynssp.detectors.switch"]], "pynssp.detectors.trend module": [[13, "module-pynssp.detectors.trend"]], "About pynssp": [[14, "about-pynssp"]], "Overview": [[14, "overview"]], "Installation": [[14, "installation"]], "Usage": [[14, "usage"]], "Contributing to this project": [[14, "contributing-to-this-project"]], "Getting Help": [[14, "getting-help"]], "Public Domain Standard Notice": [[14, "public-domain-standard-notice"]], "License Standard Notice": [[14, "license-standard-notice"]], "Privacy Standard Notice": [[14, "privacy-standard-notice"]], "Contributing Standard Notice": [[14, "contributing-standard-notice"]], "Records Management Standard Notice": [[14, "records-management-standard-notice"]]}, "indexentries": {"change_dates() (in module pynssp.utils)": [[11, "pynssp.utils.change_dates"]], "create_profile() (in module pynssp.utils)": [[11, "pynssp.utils.create_profile"]], "create_token_profile() (in module pynssp.utils)": [[11, "pynssp.utils.create_token_profile"]], "get_api_data() (in module pynssp.utils)": [[11, "pynssp.utils.get_api_data"]], "get_api_graph() (in module pynssp.utils)": [[11, "pynssp.utils.get_api_graph"]], "get_api_response() (in module pynssp.utils)": [[11, "pynssp.utils.get_api_response"]], "get_essence_data() (in module pynssp.utils)": [[11, "pynssp.utils.get_essence_data"]], "get_scenario1() (in module pynssp.data)": [[11, "pynssp.data.get_scenario1"]], "get_scenario2() (in module pynssp.data)": [[11, "pynssp.data.get_scenario2"]], "load_nssp_stopwords() (in module pynssp.data)": [[11, "pynssp.data.load_nssp_stopwords"]], "load_simulated_ts() (in module pynssp.data)": [[11, "pynssp.data.load_simulated_ts"]], "module": [[11, "module-pynssp"], [11, "module-pynssp.data"], [11, "module-pynssp.pynssp"], [11, "module-pynssp.utils"], [12, "module-pynssp.core"], [12, "module-pynssp.core.constants"], [12, "module-pynssp.core.container"], [12, "module-pynssp.core.credentials"], [12, "module-pynssp.core.token"], [13, "module-pynssp.detectors"], [13, "module-pynssp.detectors.ewma"], [13, "module-pynssp.detectors.farrington"], [13, "module-pynssp.detectors.nbinom"], [13, "module-pynssp.detectors.regression"], [13, "module-pynssp.detectors.serfling"], [13, "module-pynssp.detectors.switch"], [13, "module-pynssp.detectors.trend"]], "pynssp": [[11, "module-pynssp"]], "pynssp.data": [[11, "module-pynssp.data"]], "pynssp.pynssp": [[11, "module-pynssp.pynssp"]], "pynssp.utils": [[11, "module-pynssp.utils"]], "apigraph (class in pynssp.core.container)": [[12, "pynssp.core.container.APIGraph"]], "credentials (class in pynssp.core.credentials)": [[12, "pynssp.core.credentials.Credentials"]], "nsspcontainer (class in pynssp.core.container)": [[12, "pynssp.core.container.NSSPContainer"]], "token (class in pynssp.core.token)": [[12, "pynssp.core.token.Token"]], "get_api_data() (pynssp.core.credentials.credentials method)": [[12, "pynssp.core.credentials.Credentials.get_api_data"]], "get_api_data() (pynssp.core.token.token method)": [[12, "pynssp.core.token.Token.get_api_data"]], "get_api_graph() (pynssp.core.credentials.credentials method)": [[12, "pynssp.core.credentials.Credentials.get_api_graph"]], "get_api_graph() (pynssp.core.token.token method)": [[12, "pynssp.core.token.Token.get_api_graph"]], "get_api_response() (pynssp.core.credentials.credentials method)": [[12, "pynssp.core.credentials.Credentials.get_api_response"]], "get_api_response() (pynssp.core.token.token method)": [[12, "pynssp.core.token.Token.get_api_response"]], "pickle() (pynssp.core.credentials.credentials method)": [[12, "pynssp.core.credentials.Credentials.pickle"]], "pickle() (pynssp.core.token.token method)": [[12, "pynssp.core.token.Token.pickle"]], "plot() (pynssp.core.container.apigraph method)": [[12, "pynssp.core.container.APIGraph.plot"]], "pynssp.core": [[12, "module-pynssp.core"]], "pynssp.core.constants": [[12, "module-pynssp.core.constants"]], "pynssp.core.container": [[12, "module-pynssp.core.container"]], "pynssp.core.credentials": [[12, "module-pynssp.core.credentials"]], "pynssp.core.token": [[12, "module-pynssp.core.token"]], "show() (pynssp.core.container.apigraph method)": [[12, "pynssp.core.container.APIGraph.show"]], "adaptive_regression() (in module pynssp.detectors.regression)": [[13, "pynssp.detectors.regression.adaptive_regression"]], "alert_ewma() (in module pynssp.detectors.ewma)": [[13, "pynssp.detectors.ewma.alert_ewma"]], "alert_nbinom() (in module pynssp.detectors.nbinom)": [[13, "pynssp.detectors.nbinom.alert_nbinom"]], "alert_regression() (in module pynssp.detectors.regression)": [[13, "pynssp.detectors.regression.alert_regression"]], "alert_serfling() (in module pynssp.detectors.serfling)": [[13, "pynssp.detectors.serfling.alert_serfling"]], "alert_switch() (in module pynssp.detectors.switch)": [[13, "pynssp.detectors.switch.alert_switch"]], "ewma_loop() (in module pynssp.detectors.ewma)": [[13, "pynssp.detectors.ewma.ewma_loop"]], "nb_model() (in module pynssp.detectors.nbinom)": [[13, "pynssp.detectors.nbinom.nb_model"]], "pynssp.detectors": [[13, "module-pynssp.detectors"]], "pynssp.detectors.ewma": [[13, "module-pynssp.detectors.ewma"]], "pynssp.detectors.farrington": [[13, "module-pynssp.detectors.farrington"]], "pynssp.detectors.nbinom": [[13, "module-pynssp.detectors.nbinom"]], "pynssp.detectors.regression": [[13, "module-pynssp.detectors.regression"]], "pynssp.detectors.serfling": [[13, "module-pynssp.detectors.serfling"]], "pynssp.detectors.switch": [[13, "module-pynssp.detectors.switch"]], "pynssp.detectors.trend": [[13, "module-pynssp.detectors.trend"]], "serfling_model() (in module pynssp.detectors.serfling)": [[13, "pynssp.detectors.serfling.serfling_model"]]}})