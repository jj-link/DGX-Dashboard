#!/usr/bin/env python3
import socket, subprocess, threading, os, signal
HOST="0.0.0.0"; PORT=8000; CONTAINER="sglang-qwen36-dflash"; TARGET_PORT="30000"
def pump_sock_to_stdin(sock, stdin):
    try:
        while True:
            data=sock.recv(65536)
            if not data: break
            stdin.write(data); stdin.flush()
    except Exception: pass
    try: stdin.close()
    except Exception: pass
def pump_stdout_to_sock(stdout, sock):
    try:
        while True:
            data=stdout.read(65536)
            if not data: break
            sock.sendall(data)
    except Exception: pass
    try: sock.close()
    except Exception: pass
def handle(conn):
    p=subprocess.Popen(["docker","exec","-i",CONTAINER,"python3","-c",f"import socket,sys; s=socket.create_connection((127.0.0.1,{TARGET_PORT})); import threading; threading.Thread(target=lambda: [s.sendall(d) for d in iter(lambda: sys.stdin.buffer.read(65536), b)], daemon=True).start(); [sys.stdout.buffer.write(d) or sys.stdout.buffer.flush() for d in iter(lambda: s.recv(65536), b)]"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
    threading.Thread(target=pump_sock_to_stdin,args=(conn,p.stdin),daemon=True).start()
    pump_stdout_to_sock(p.stdout,conn)
    try: p.kill()
    except Exception: pass
def main():
    s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind((HOST,PORT)); s.listen(128)
    print(f"listening {HOST}:{PORT} -> {CONTAINER}:{TARGET_PORT}",flush=True)
    while True:
        c,_=s.accept(); threading.Thread(target=handle,args=(c,),daemon=True).start()
if __name__=="__main__": main()
