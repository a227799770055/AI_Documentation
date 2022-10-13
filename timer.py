import time

time_start = time.time()
seconds = 0
minutes = 0

for i in range(70):
    time.sleep(1)
    seconds = int(time.time() - time_start) - minutes * 60
    if seconds >= 60:
            minutes += 1
            seconds = 0
    m = "%02d" %minutes
    s = "%02d" %seconds
    print("{}:{}".format(m, s))
