#!/usr/bin/env python3
"""
Send paul_car drive commands to BallDetector_0D over the ST-Link VCP.

Protocol (single char): 0=stop 1=forward 2=fwd-right 3=fwd-left 4=backward 5=stop

Usage:
    python3 car_cmd.py 1            # one command then exit
    python3 car_cmd.py             # interactive: type 0-5 + Enter, 'q' to quit
"""
import sys
try:
    import serial
except ImportError:
    sys.exit("pyserial required:  pip install pyserial")

PORT, BAUD = "/dev/ttyACM0", 921600


def main():
    ser = serial.Serial(PORT, BAUD)
    if len(sys.argv) > 1:
        ser.write(sys.argv[1][:1].encode())
        print(f"sent {sys.argv[1][:1]!r}")
        return
    print("commands: 0 stop | 1 fwd | 2 fwd-right | 3 fwd-left | 4 back | q quit")
    try:
        while True:
            c = input("> ").strip()
            if c[:1] in ("q", "Q"):
                ser.write(b"0")            # stop on exit
                break
            if c[:1] in "012345":
                ser.write(c[:1].encode())
    except (KeyboardInterrupt, EOFError):
        ser.write(b"0")
        print("\nstopped")


if __name__ == "__main__":
    main()
