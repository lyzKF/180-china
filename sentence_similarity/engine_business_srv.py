# _*_ coding:utf-8 -*-

import sys
import time
import os
reload(sys)
sys.setdefaultencoding( "utf-8" )
sys.path.append('gen-py')
#sys.path.insert(0, glob.glob('../../lib/py/build/lib*')[0])
from engine_caengineer_handler import CAEngineeHandler
import logging
import time
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)

import signal

def sigint_handler(signum, frame):
  pid=os.getpid()
  os.kill(pid,signal.SIGKILL)



if __name__ == '__main__':
    # is_sigint_stop = False
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGHUP, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    handler = CAEngineeHandler()
    time.sleep(2)
    handler.serverStart(9090)
    #handler.serverProc()
    while True:
     time.sleep(1)



