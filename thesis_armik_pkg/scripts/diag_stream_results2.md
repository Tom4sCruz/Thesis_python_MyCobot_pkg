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
  get_angles               : min    7.2  median   16.4  p95   22.0  max   25.2  ms  (n=40)
  send_angles  synchronous : min    1.8  median    7.4  p95   10.3  max   11.2  ms  (n=40)
  send_angles  _async=True : min    0.3  median    3.3  p95    3.7  max    4.0  ms  (n=40)
  gripper      synchronous : min    1.7  median    6.0  p95    8.6  max    9.4  ms  (n=12)
  gripper      no wait     : min    0.2  median    0.6  p95    1.5  max    1.7  ms  (n=12)

##### 3. THE SAME MOVE, SEVERAL WAYS -- watch the arm #####

=== M1  one firmware command each way (no streaming) ===
  M1: send call 6.1 ms, arrived after TIMEOUT (firmware speed 49)
  M1: send call 6.3 ms, arrived after TIMEOUT (firmware speed 49)

=== M2  armik stream as it is today (arm.move_joints, synchronous sends) ===
  M2: planned 4.00s actual 4.02s, LATE ticks 0
      gap between sends: min   30.8  median   40.0  p95   45.0  max   47.2  ms  (n=98)

=== M3  the same stream, asynchronous sends ===
  M3: 100 setpoints, tick 40 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.3  median    3.7  p95    4.2  max    5.5  ms  (n=100)
      gap between sends: min   34.4  median   40.0  p95   42.6  max   44.9  ms  (n=98)

=== M4  asynchronous stream, setpoints 5 ms apart (deliberately too fast) ===
  M4: 800 setpoints, tick 5 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.3  median    1.1  p95    1.7  max    3.5  ms  (n=800)
      gap between sends: min    2.0  median    5.0  p95    5.5  max   11.0  ms  (n=798)

=== M5a  M3 + gripper commands mid-move, gripper sent the BLOCKING way ===
  M5a: 100 setpoints, tick 40 ms, planned 4.00s actual 4.62s, LATE ticks 64
      time inside send : min    0.2  median    0.4  p95    3.6  max    4.0  ms  (n=100)
      gap between sends: min    0.3  median    0.6  p95   40.4  max 1614.0  ms  (n=98)
      gripper call     : min 1515.5  median 1562.4  p95 1604.6  max 1609.3  ms  (n=2)

=== M5b  M3 + gripper commands mid-move, gripper written WITHOUT waiting ===
  M5b: 100 setpoints, tick 40 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.3  median    2.9  p95    3.6  max    3.7  ms  (n=100)
      gap between sends: min   34.2  median   40.0  p95   43.1  max   43.8  ms  (n=98)
      gripper call     : min    0.6  median    0.6  p95    0.7  max    0.7  ms  (n=2)

##### DONE #####
Paste everything above, and say which of M1, M2, M3, M4, M5a, M5b LOOKED smooth
and which stuttered (and, for M5a/M5b, whether the gripper actually moved).
