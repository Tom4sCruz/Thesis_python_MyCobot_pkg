
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
  get_angles               : min    8.4  median   15.8  p95   19.8  max   21.8  ms  (n=40)
  send_angles  synchronous : min    3.3  median    8.1  p95   10.1  max   11.3  ms  (n=40)
  send_angles  _async=True : min    1.0  median    3.5  p95    3.8  max    5.2  ms  (n=40)
  gripper      synchronous : min    1.7  median    7.3  p95    8.5  max    8.6  ms  (n=12)
  gripper      no wait     : min    0.5  median    1.5  p95    1.9  max    2.0  ms  (n=12)

##### 3. THE SAME MOVE, SEVERAL WAYS -- watch the arm #####

=== M1  one firmware command each way (no streaming) ===
  M1: send call 7.5 ms, arrived after TIMEOUT (firmware speed 49)
  M1: send call 7.1 ms, arrived after TIMEOUT (firmware speed 49)

=== M2  armik stream as it is today (arm.move_joints, synchronous sends) ===
  M2: planned 4.00s actual 4.03s, LATE ticks 0
      gap between sends: min   31.8  median   40.0  p95   43.2  max   45.3  ms  (n=98)

=== M3  the same stream, asynchronous sends ===
  M3: 100 setpoints, tick 40 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.7  median    3.2  p95    3.7  max    3.9  ms  (n=100)
      gap between sends: min   36.3  median   40.0  p95   42.4  max   43.4  ms  (n=98)

=== M4  asynchronous stream, setpoints 5 ms apart (deliberately too fast) ===
  M4: 800 setpoints, tick 5 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.6  median    1.1  p95    1.7  max    3.7  ms  (n=800)
      gap between sends: min    2.5  median    5.0  p95    5.6  max    8.8  ms  (n=798)

=== M5a  M3 + gripper commands mid-move, gripper sent the BLOCKING way ===
  M5a: 100 setpoints, tick 40 ms, planned 4.00s actual 4.35s, LATE ticks 44
      time inside send : min    0.2  median    2.0  p95    3.5  max    3.7  ms  (n=100)
      gap between sends: min    0.3  median   38.7  p95   42.8  max 1590.8  ms  (n=98)
      gripper call     : min  535.5  median 1060.1  p95 1532.3  max 1584.7  ms  (n=2)

=== M5b  M3 + gripper commands mid-move, gripper written WITHOUT waiting ===
  M5b: 100 setpoints, tick 40 ms, planned 4.00s actual 4.01s, LATE ticks 0
      time inside send : min    0.8  median    3.1  p95    3.6  max    4.0  ms  (n=100)
      gap between sends: min   36.2  median   40.0  p95   40.6  max   43.9  ms  (n=98)
      gripper call     : min    0.7  median    1.0  p95    1.4  max    1.4  ms  (n=2)

##### DONE #####
Paste everything above, and say which of M1, M2, M3, M4, M5a, M5b LOOKED smooth
and which stuttered (and, for M5a/M5b, whether the gripper actually moved).
