import numpy
from time import sleep

print("sleeping....")
sleep(10000)
exit(0)

# count the number of processes spawned by a process matching a pattern, in this case "2023-"
# watch -n 3 "for pid in \$(ps aux | grep 2023[-] | awk '{print \$2}'); do echo -n 'PID '\$pid': '; count=\$(pstree -p \$pid | grep -o '([[:digit:]]\+' | wc -l); echo \$((count-1)); done"