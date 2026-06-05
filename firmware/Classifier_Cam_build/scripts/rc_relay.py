#!/usr/bin/env python3
"""
rc_relay.py -- PC-Bruecke: U585-Gestenklassifikator -> Bluetooth -> RC-Auto (HC-06).

Fluss:
    U585  --UART(VCP)-->  PC (dieses Skript)  --BT-->  HC-06  --UART(9600)-->  Auto-MCU

Die U585 rechnet die Inferenz on-device und gibt auf USART1 (ST-LINK VCP) Zeilen aus:
    PRED rock conf=0.92 -> BT '1' (FORWARD) | infer=12ms
Dieses Skript zieht den Befehl ('0'..'5') aus der Zeile und schickt ihn per
RFCOMM (Bluetooth SPP) an den HC-06. Der HC-06 gibt das Byte mit 9600 Baud an
den Auto-MCU weiter (gleiche '0'..'5'-Befehle wie die Handy-Steuerung).

Voraussetzungen:
  * HC-06 ist mit dem PC gepairt+trusted (PIN 1234) und FREI (Handy-BT aus).
  * U585 laeuft mit der normalen Classifier-Firmware (HC05_AT_BRIDGE = 0),
    nicht mit der AT-Bridge.

Beenden mit Ctrl-C.
"""

import argparse
import re
import socket
import sys
import time

import serial  # pyserial

# "... -> BT '1' (FORWARD) ..."  -> faengt das einzelne Befehlszeichen
CMD_RE = re.compile(rb"-> BT '(.)'")
# "... conf=0.92 ..."  -> Konfidenz, um schwache Vorhersagen nicht anzuzeigen
CONF_RE = re.compile(rb"conf=([0-9.]+)")
CONF_DISPLAY_MIN = 0.5

CMD_NAME = {
    '0': "STOP", '1': "FORWARD", '2': "FWD-RIGHT",
    '3': "FWD-LEFT", '4': "BACKWARD", '5': "OTHER/STOP",
}


def connect_bt(mac, channel, retry_s=2.0):
    """RFCOMM-Socket zum HC-06 (PC = Master). Blockiert bis es klappt."""
    while True:
        try:
            s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM,
                              socket.BTPROTO_RFCOMM)
            s.settimeout(15)
            print(f"[BT] verbinde mit {mac} (Kanal {channel}) ...")
            s.connect((mac, channel))
            s.settimeout(None)
            print("[BT] verbunden (HC-06-LED sollte dauerhaft leuchten)")
            return s
        except OSError as e:
            print(f"[BT] Verbindung fehlgeschlagen ({e}); neuer Versuch in {retry_s}s "
                  "(HC-06 an? frei? gepairt?)")
            time.sleep(retry_s)


def open_uart(port, baud, retry_s=2.0):
    while True:
        try:
            u = serial.Serial(port, baud, timeout=0.1)
            print(f"[UART] {port} @ {baud} offen")
            return u
        except serial.SerialException as e:
            print(f"[UART] {port} nicht da ({e}); neuer Versuch in {retry_s}s "
                  "(U585 angesteckt? richtige Firmware?)")
            time.sleep(retry_s)


def main():
    ap = argparse.ArgumentParser(description="U585-Geste -> Bluetooth -> RC-Auto Relay")
    ap.add_argument("--port", default="/dev/ttyACM1", help="U585 VCP (Default /dev/ttyACM1)")
    ap.add_argument("--baud", type=int, default=921600, help="USART1-Baud (Default 921600)")
    ap.add_argument("--mac", default="98:DA:60:0F:EE:0B", help="HC-06 Bluetooth-MAC")
    ap.add_argument("--channel", type=int, default=1, help="RFCOMM-Kanal (SPP=1)")
    ap.add_argument("--heartbeat", type=float, default=0.3,
                    help="gleichen Befehl spaetestens alle N s erneut senden (0=aus)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="jedes Kommando N-mal hintereinander senden (Redundanz; Default 3)")
    args = ap.parse_args()

    uart = open_uart(args.port, args.baud)
    bt = connect_bt(args.mac, args.channel)

    last_cmd = None
    last_send = 0.0
    buf = b""

    def send(cmd):
        nonlocal bt, last_cmd, last_send
        payload = (cmd * args.repeat).encode()   # z.B. "111" -- Redundanz gegen verlorene Bytes
        while True:
            try:
                bt.sendall(payload)
                last_cmd, last_send = cmd, time.time()
                return
            except OSError as e:
                print(f"[BT] Sendefehler ({e}); reconnect ...")
                try:
                    bt.close()
                except OSError:
                    pass
                bt = connect_bt(args.mac, args.channel)

    print("[relay] laeuft -- Ctrl-C beendet\n")
    try:
        while True:
            try:
                chunk = uart.read(256)
            except serial.SerialException as e:
                print(f"[UART] Lesefehler ({e}); reconnect ...")
                try:
                    uart.close()
                except serial.SerialException:
                    pass
                uart = open_uart(args.port, args.baud)
                continue

            if chunk:
                buf += chunk
                # alle vollstaendigen Zeilen verarbeiten, Rest im Puffer lassen
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    m = CMD_RE.search(line)
                    if not m:
                        continue
                    cmd = m.group(1).decode(errors="replace")
                    if cmd not in CMD_NAME:
                        continue
                    mc = CONF_RE.search(line)
                    conf = float(mc.group(1)) if mc else 0.0
                    if cmd != last_cmd:
                        # schwache Vorhersagen (< 0.5) NICHT anzeigen; der Befehl
                        # (STOP) wird trotzdem gesendet, damit das Auto haelt.
                        if conf >= CONF_DISPLAY_MIN:
                            print(f">> '{cmd}' ({CMD_NAME[cmd]})   {line.decode(errors='replace').strip()}")
                        send(cmd)
                if len(buf) > 4096:        # Schutz gegen Muellanhaeufung
                    buf = buf[-512:]

            # Heartbeat: gleichen Befehl periodisch erneut senden (Selbstheilung
            # bei verlorenem Byte; STOP bleibt zuverlaessig).
            if args.heartbeat and last_cmd is not None and \
               (time.time() - last_send) >= args.heartbeat:
                send(last_cmd)
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[relay] beende, sende STOP")
        try:
            bt.sendall(b"0")
        except OSError:
            pass
    finally:
        try:
            bt.close()
        except OSError:
            pass
        uart.close()


if __name__ == "__main__":
    sys.exit(main())
