import sys
import time

for i in range(99999999):
    print("%d \r" % i, end="")
    sys.stdout.flush()
