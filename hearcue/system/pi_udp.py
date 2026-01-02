import socket
import json
import time
import threading

# ---------- NETWORK (Pi side) ----------
PC_HOSTNAME = "Mertspc.local"   # <-- your PC mDNS name from avahi output
PC_PORT = 5005                  # Pi -> PC
PI_PORT = 5006                  # PC -> Pi

def resolve_host(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except Exception as e:
        print("mDNS resolve failed:", host, e)
        return host

class PiUDP:
    def __init__(self):
        self.pc_ip = resolve_host(PC_HOSTNAME)

        self.sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx.bind(("", PI_PORT))
        self.sock_rx.settimeout(1.0)

        self.mode = "home"
        self._lock = threading.Lock()

        threading.Thread(target=self._rx_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _heartbeat_loop(self):
        while True:
            self.send({"type": "heartbeat", "ts": time.time()})
            time.sleep(1.0)

    def _rx_loop(self):
        while True:
            try:
                data, _ = self.sock_rx.recvfrom(2048)
            except socket.timeout:
                continue
            except Exception:
                continue

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            # PC UI sends: {"type":"mode","mode":"home"}
            if msg.get("type") == "mode" and "mode" in msg:
                with self._lock:
                    self.mode = msg["mode"]
                print(f"[MODE] -> {self.mode}")

    def get_mode(self) -> str:
        with self._lock:
            return self.mode

    def _refresh_pc_ip(self):
        self.pc_ip = resolve_host(PC_HOSTNAME)

    def send(self, msg: dict):
        payload = json.dumps(msg).encode("utf-8")
        try:
            self.sock_tx.sendto(payload, (self.pc_ip, PC_PORT))
        except Exception as e:
            print("UDP send failed:", e)
            self._refresh_pc_ip()
            try:
                self.sock_tx.sendto(payload, (self.pc_ip, PC_PORT))
            except Exception as e2:
                print("UDP send retry failed:", e2)

    # convenience wrappers for your audio loop:
    def send_idle(self):
        self.send({"type": "status", "state": "idle", "ts": time.time()})

    def send_trigger(self, label: str, conf: float | None = None):
        msg = {"type": "event", "trigger": label, "ts": time.time()}
        if conf is not None:
            msg["conf"] = float(conf)
        self.send(msg)
