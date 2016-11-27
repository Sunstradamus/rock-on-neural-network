import sys

def progress_bar(percent):
    print "\r[",
    print "\b" + "#"*(percent/5) + " "*(20-percent/5) + "] %d %%" %(percent),
    sys.stdout.flush()

