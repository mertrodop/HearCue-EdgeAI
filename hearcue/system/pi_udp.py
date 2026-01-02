import socket
import json
import time
import threading

PC_IP = "172.20.10.2"
PC_PORT = 5005
PI_PORT = 5006

class PiUDP:
    def __init__(self):
        self.sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx.bind(("", PI_PORT))
        self.mode = "library"

        threading.Thread(target=self._rx_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _heartbeat_loop(self):
        while True:
            self.send({"type": "heartbeat"})
            time.sleep(1.0)

    def _rx_loop(self):
        while True:
            data, _ = self.sock_rx.recvfrom(1024)
            msg = json.loads(data.decode())
            if msg["type"] == "set_mode":
                self.mode = msg["mode"]
                print(f"[MODE] -> {self.mode}")

    def send(self, msg: dict):
        self.sock_tx.sendto(
            json.dumps(msg).encode(),
            (PC_IP, PC_PORT)
        )
