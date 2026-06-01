#!/usr/bin/env bash
# Headless fault inspection: load the FSBL into RAM via ST-LINK GDB, run it,
# trigger one inference from the host, and when the NPU register access faults
# (vectors to 0x0 / a fault handler) dump the Cortex-M fault status registers
# (CFSR/HFSR/BFAR/MMFAR/SFSR/SFAR) + stacked frame + backtrace.
#
# PREREQ: no other debugger (CubeIDE) is attached to the ST-LINK.
set -uo pipefail

CUBE=/opt/st/stm32cubeide_2.1.1
GDBSRV="$CUBE/plugins/com.st.stm32cube.ide.mcu.externaltools.stlink-gdb-server.linux64_2.2.400.202601091506/tools/bin/ST-LINK_gdbserver"
GDB="$CUBE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin/arm-none-eabi-gdb"
CP="$CUBE/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJ="$(cd "$HERE/.." && pwd)"
ELF="$PROJ/FSBL/Debug/BallDetector_N6_FSBL.elf"
HOSTPY="$PROJ/../Host/host_balldetector_n6.py"
VAL="$PROJ/../../software/ball_detection/splits/val.txt"
PORT=61234

pkill -f ST-LINK_gdbserver 2>/dev/null; sleep 1

echo "==> starting ST-LINK gdbserver"
"$GDBSRV" -p $PORT -cp "$CP" -d -e --halt -k --pend-halt-timeout 2000 -r 15 >/tmp/gdbsrv.out 2>&1 &
sleep 4
if ! grep -qi "Listening\|Waiting" /tmp/gdbsrv.out; then
  echo "!! gdbserver not listening — is CubeIDE still attached? log:"; cat /tmp/gdbsrv.out; exit 1
fi

cat > /tmp/fault.gdb <<'EOF'
set pagination off
set confirm off
target extended-remote :61234
monitor halt
load
# fault catchers: handlers (secure) + the bad 0x0 vector + the reporter
break *0x0
break HardFault_Handler
break BusFault_Handler
break MemManage_Handler
break UsageFault_Handler
break SecureFault_Handler
break balld_fault_report
continue
echo \n=== TARGET STOPPED (fault) ===\n
printf "CFSR =%08x  HFSR=%08x  BFAR=%08x  MMFAR=%08x\n", *(unsigned*)0xE000ED28, *(unsigned*)0xE000ED2C, *(unsigned*)0xE000ED38, *(unsigned*)0xE000ED34
printf "SFSR =%08x  SFAR=%08x  SHCSR=%08x  AIRCR=%08x\n", *(unsigned*)0xE000EDE4, *(unsigned*)0xE000EDE8, *(unsigned*)0xE000ED24, *(unsigned*)0xE000ED0C
printf "MSP=%08x PSP=%08x LR=%08x PC=%08x\n", $msp, $psp, $lr, $pc
echo --- stacked exception frame (@MSP): r0 r1 r2 r3 r12 lr pc xpsr ---\n
x/8xw $msp
echo --- backtrace ---\n
bt
quit
EOF

echo "==> launching gdb (loads + runs; will block until fault)"
"$GDB" "$ELF" -x /tmp/fault.gdb >/tmp/fault.out 2>&1 &
GDBPID=$!
sleep 8   # let gdb load + continue; firmware boots to "ready" and waits on UART

echo "==> sending one image from host to trigger inference -> fault"
( cd "$PROJ/../.." && python firmware/Host/host_balldetector_n6.py --port /dev/ttyACM0 img --image-list software/ball_detection/splits/val.txt --limit 1 ) >/tmp/host.out 2>&1 || true

# give gdb a moment to catch + dump
for i in $(seq 1 20); do kill -0 $GDBPID 2>/dev/null || break; sleep 1; done

echo; echo "================ GDB FAULT DUMP ================"; cat /tmp/fault.out
echo; echo "================ HOST OUTPUT ================";    tail -20 /tmp/host.out
pkill -f ST-LINK_gdbserver 2>/dev/null
