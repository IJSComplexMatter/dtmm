from __future__ import absolute_import, print_function, division
import sys, time


def print_message(message, level = 1):
    if level >= 1:
        print(message)
    
def print_layer_rate(n_frames, t0, t1 = None, message = "... processed", level = 2):
    if level >= 2:
        if t1 is None:
            t1 = time.time()#take current time
        print (message + " {0} layers with an average rate {1:.2f}".format(n_frames, n_frames/(t1-t0)))
        
def print_progress (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '=', level = 2):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if level >= 1:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    #        sys.stdout.write(s)
    #        sys.stdout.flush()
        # Print New Line on Complete
        if iteration == total: 
            print()
        
def test_progress():
    import time
    print("start")
    for i in range(100):
        print_progress(i,100, fill = "=")
        time.sleep(0.02)
    print_progress(100,100)
    print("stop")