"""仅将本机发布端口的TCP字节转发给内网Streamlit，支持HTTP和WebSocket。"""

import select
import socket
from socketserver import BaseRequestHandler, ThreadingTCPServer


class StreamlitConnection(BaseRequestHandler):
    def handle(self):
        try:
            with socket.create_connection(("app", 8501), timeout=5) as upstream:
                upstream.settimeout(None)
                # 不解析请求、不访问模型或文档，只在两条连接之间转发字节。
                peers = {self.request: upstream, upstream: self.request}
                while True:
                    ready, _, _ = select.select(list(peers), [], [])
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        peers[source].sendall(data)
        except OSError as error:
            print(f"入口连接结束：{type(error).__name__}", flush=True)


if __name__ == "__main__":
    # 每条浏览器连接独立转发；退出入口时无需等待长连接。
    ThreadingTCPServer.allow_reuse_address = True
    ThreadingTCPServer.daemon_threads = True
    with ThreadingTCPServer(("0.0.0.0", 8501), StreamlitConnection) as server:
        server.serve_forever()
