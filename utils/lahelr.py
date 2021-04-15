import time

def lahelr_print(*args,sep=None,end=None):
    pass
    print(*args,sep=sep,end=end)
    # print()

def lahelr_time():
    return time.asctime(time.localtime(time.time()))
