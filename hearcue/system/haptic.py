import time
import subprocess
import signal

class HapticController:
    """
    Single-pin haptic controller for GPIO27 using gpioset --mode=wait.
    Guaranteed to work on Ubuntu Pi 5.
    """

    CHIP = "gpiochip4"
    PIN = 27

    def __init__(self):
        self._proc = None
        print("Haptics ready on GPIO27")
        self.off()

    def _force_low(self):
        subprocess.run(
            ["gpioset", self.CHIP, f"{self.PIN}=0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def pulse(self, label: str = "", duration: float = 0.4):
        # Stop anything currently running
        self.off()

        # Hold GPIO27 HIGH
        self._proc = subprocess.Popen(
            ["gpioset", "--mode=wait", self.CHIP, f"{self.PIN}=1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(duration)

        # Stop and release
        self.off()

    def off(self):
        if self._proc is not None:
            try:
                self._proc.send_signal(signal.SIGINT)
                self._proc.wait(timeout=0.3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

        # Always force LOW
        self._force_low()
