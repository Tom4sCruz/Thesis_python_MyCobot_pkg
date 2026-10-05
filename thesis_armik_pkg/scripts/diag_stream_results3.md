##### 1. ENVIRONMENT #####
  pymycobot version        : 4.0.4b5
  raw object               : MyCobot280
  send_angles has _async   : True
  gripper no-wait possible : True
  serial read timeout      : 0.1
  fresh mode               : 1
  CONTROL_RATE_HZ / GAIN   : 25.0 / 1.6

homing (one firmware command)...
  !! did not reach HOME within the timeout -- results below may be off

##### 2. CALL LATENCY, ARM STILL #####
  get_angles               : min    7.4  median   14.8  p95   19.1  max   22.7  ms  (n=40)
  send_angles  synchronous : min    2.9  median    8.0  p95   10.3  max  514.6  ms  (n=40)
  send_angles  _async=True : min    0.3  median    3.1  p95    3.6  max    3.7  ms  (n=40)
  gripper      synchronous : min    3.1  median    6.4  p95    7.7  max    7.7  ms  (n=12)
  gripper      no wait     : min    0.1  median    0.5  p95    0.8  max    0.9  ms  (n=12)

##### 3. THE SAME MOVE, SEVERAL WAYS -- watch the arm #####

=== M1  one firmware command each way (no streaming) ===
  M1: send call 8.0 ms, arrived after TIMEOUT (firmware speed 49)
  M1: send call 10.1 ms, arrived after TIMEOUT (firmware speed 49)

=== M2  armik stream as it is today (arm.move_joints, synchronous sends) ===
  M2: planned 4.00s actual 4.02s, LATE ticks 0
      gap between sends: min   30.1  median   40.0  p95   44.4  max   46.8  ms  (n=98)

=== M3  the same stream, asynchronous sends ===
  M3: 100 setpoints, tick 40 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    1.6  median    3.7  p95    4.8  max    5.9  ms  (n=100)
      gap between sends: min   34.6  median   40.0  p95   43.1  max   44.8  ms  (n=98)

=== M4  asynchronous stream, setpoints 5 ms apart (deliberately too fast) ===
  M4: 800 setpoints, tick 5 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.3  median    1.1  p95    1.7  max    3.2  ms  (n=800)
      gap between sends: min    2.0  median    5.0  p95    5.7  max    6.9  ms  (n=798)

=== M5a  M3 + gripper commands mid-move, gripper sent the BLOCKING way ===
  M5a: 100 setpoints, tick 40 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.8  median    2.7  p95    3.8  max    4.0  ms  (n=100)
      gap between sends: min   35.7  median   40.0  p95   42.5  max   44.5  ms  (n=98)
      gripper call     : min    6.9  median    8.1  p95    9.1  max    9.3  ms  (n=2)

=== M5b  M3 + gripper commands mid-move, gripper written WITHOUT waiting ===
  M5b: 100 setpoints, tick 40 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.4  median    3.0  p95    3.8  max    4.8  ms  (n=100)
      gap between sends: min   34.5  median   40.0  p95   42.9  max   46.1  ms  (n=98)
      gripper call     : min    0.7  median    0.9  p95    1.1  max    1.1  ms  (n=2)

##### DONE #####
Paste everything above, and say which of M1, M2, M3, M4, M5a, M5b LOOKED smooth
and which stuttered (and, for M5a/M5b, whether the gripper actually moved).
