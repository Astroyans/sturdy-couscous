from __future__ import annotations

import argparse
import socket
import threading
import time

# Edita aqui la IP de la Laptop Rover destino (donde corre main.py / main_expo.py)
TARGET_IP = "192.168.1.100"
TARGET_PORT = 8766
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8766


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Puente UDP zero-copy: Raspi 3 -> Laptop Puente -> Laptop Rover (sturdy-couscous)"
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=LISTEN_PORT,
        help="Puerto UDP donde esta laptop puente recibe los paquetes desde la Raspi 3.",
    )
    parser.add_argument(
        "--target-ip",
        default=TARGET_IP,
        help=f"IP LAN de la Laptop Rover destino (por defecto TARGET_IP={TARGET_IP}).",
    )
    parser.add_argument(
        "--target-port",
        type=int,
        default=TARGET_PORT,
        help=f"Puerto UDP destino en la Laptop Rover (por defecto {TARGET_PORT}).",
    )
    parser.add_argument(
        "--listen-host",
        default=LISTEN_HOST,
        help="Interfaz local donde escuchar (por defecto 0.0.0.0).",
    )
    return parser.parse_args(argv)


def run_udp_bridge(
    listen_host: str,
    listen_port: int,
    target_ip: str,
    target_port: int,
    stop_event: threading.Event | None = None,
) -> None:
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for s in (rx_sock, tx_sock):
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        except OSError:
            pass

    rx_sock.bind((listen_host, listen_port))
    rx_sock.settimeout(0.2)
    tx_sock.bind(("0.0.0.0", 0))
    tx_sock.settimeout(0.2)

    target_addr = (target_ip, target_port)
    addr_lock = threading.Lock()
    last_raspi_addr: list[tuple[str, int] | None] = [None]
    stats = {"packets": 0, "bytes": 0, "acks": 0}
    local_stop = stop_event or threading.Event()

    def _ack_relay_loop() -> None:
        while not local_stop.is_set():
            try:
                ack_data, _ = tx_sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            with addr_lock:
                dst = last_raspi_addr[0]
            if dst is not None:
                try:
                    rx_sock.sendto(ack_data, dst)
                    stats["acks"] += 1
                except OSError:
                    pass

    ack_thread = threading.Thread(target=_ack_relay_loop, daemon=True)
    ack_thread.start()

    t0 = time.perf_counter()
    print(
        f"[BRIDGE-UDP] Escuchando en {listen_host}:{listen_port} -> "
        f"reenviando a {target_ip}:{target_port}"
    )

    try:
        while not local_stop.is_set():
            try:
                data, src_addr = rx_sock.recvfrom(4096)
                with addr_lock:
                    last_raspi_addr[0] = src_addr
                tx_sock.sendto(data, target_addr)
                stats["packets"] += 1
                stats["bytes"] += len(data)
            except socket.timeout:
                pass
            except OSError:
                break

            now = time.perf_counter()
            dt = now - t0
            if dt >= 2.0:
                kbps = (stats["bytes"] * 8.0 / 1000.0) / dt
                pps = stats["packets"] / dt
                with addr_lock:
                    src = last_raspi_addr[0]
                src_str = f"{src[0]}:{src[1]}" if src is not None else "esperando..."
                print(
                    f"[BRIDGE-UDP] origen={src_str} | {pps:.0f} pkt/s | "
                    f"{kbps:.1f} kbps | acks={stats['acks']}"
                )
                stats["packets"] = 0
                stats["bytes"] = 0
                stats["acks"] = 0
                t0 = now
    except KeyboardInterrupt:
        print("\n[BRIDGE-UDP] Puente detenido.")
    finally:
        local_stop.set()
        ack_thread.join(timeout=1.0)
        rx_sock.close()
        tx_sock.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_udp_bridge(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        target_ip=args.target_ip,
        target_port=args.target_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
