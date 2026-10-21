from src.utils.constants import LOG_DIR

def write_log(msg: str, module: str, lvl: str = 'INFO'):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / (module + '.log')

    with open(log_file, 'a', encoding='utf-8') as lf:
        lf.write('[' + module + ']\t' + lvl + '\t| ' + msg + '\n')