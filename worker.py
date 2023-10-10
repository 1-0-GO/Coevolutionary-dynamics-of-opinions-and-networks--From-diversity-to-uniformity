from multiprocessing import Process, freeze_support
from my_bussiness import mean
if __name__ == '__main__':
    freeze_support()
    Process(target=mean).start()
