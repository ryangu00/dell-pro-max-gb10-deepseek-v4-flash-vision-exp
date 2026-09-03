#!/usr/bin/env python3
"""dsv4f-thinking-proxy - DSV4F Vision-Exp 双机 thinking 分档(2026-09-01,派生自 thinking_proxy.py/ADR-0101)。
一个 vLLM 双机引擎(head :8899)+两个薄代理端口,固定注入 chat_template_kwargs:
  :8901 = off (thinking=false)  - fleet 主力快档
  :8902 = on  (thinking=true)   - 质量档(agentic 语义/日期计算,实测 6-13x 时间)
端口语义强制覆盖调用方;/v1/chat/completions 之外全透传。纯 stdlib,SSE 透传。"""
import asyncio, json, sys

UPSTREAM_HOST, UPSTREAM_PORT = "127.0.0.1", 8899
PORTS = {
    8901: {"thinking": False},
    8902: {"thinking": True},
}

async def read_chunked(reader, buf):
    """RFC7230 chunked body 解码;返回完整 body 字节(trailer 丢弃)。"""
    async def fill(cond):
        nonlocal buf
        while not cond(buf):
            d = await reader.read(65536)
            if not d:
                raise ConnectionError("client EOF mid-chunked")
            buf += d
    body = b""
    while True:
        await fill(lambda b: b"\r\n" in b)
        line, _, buf = buf.partition(b"\r\n")
        size = int(line.split(b";")[0].strip() or b"0", 16)
        if size == 0:
            while True:  # trailer 直到空行
                await fill(lambda b: b"\r\n" in b)
                line, _, buf = buf.partition(b"\r\n")
                if not line:
                    return body
        await fill(lambda b, n=size: len(b) >= n + 2)
        body += buf[:size]
        buf = buf[size + 2:]

async def pump(reader, writer):
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass

async def handle(client_r, client_w, kwargs):
    try:
        # 读请求头
        head = b""
        while b"\r\n\r\n" not in head:
            b_ = await client_r.read(65536)
            if not b_:
                return
            head += b_
        header_blob, _, body_start = head.partition(b"\r\n\r\n")
        lines = header_blob.split(b"\r\n")
        request_line = lines[0].decode("latin1")
        method, path, _ = request_line.split(" ", 2)
        headers = {}
        for ln in lines[1:]:
            k, _, v = ln.decode("latin1").partition(":")
            headers[k.strip().lower()] = v.strip()
        if "chunked" in headers.get("transfer-encoding", "").lower():
            body = await read_chunked(client_r, body_start)  # 解块重算 content-length(对抗审 #6)
        else:
            clen = int(headers.get("content-length", "0"))
            body = body_start
            while len(body) < clen:
                chunk = await client_r.read(65536)
                if not chunk:
                    return  # client 半途断开:EOF 不退会死循环烧 CPU
                body += chunk

        inject = method == "POST" and path.startswith("/v1/chat/completions")
        if inject:
            try:
                obj = json.loads(body.decode("utf-8"))
                ck = obj.get("chat_template_kwargs") or {}
                ck.update(kwargs)  # 端口档位强制覆盖
                obj["chat_template_kwargs"] = ck
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            except Exception:
                pass  # 非 JSON 原样透传

        up_r, up_w = await asyncio.open_connection(UPSTREAM_HOST, UPSTREAM_PORT)
        out = [f"{method} {path} HTTP/1.1"]
        for k, v in headers.items():
            if k in ("content-length", "host", "connection", "transfer-encoding"):
                continue
            out.append(f"{k}: {v}")
        out.append(f"host: {UPSTREAM_HOST}:{UPSTREAM_PORT}")
        out.append("connection: close")
        out.append(f"content-length: {len(body)}")
        up_w.write(("\r\n".join(out) + "\r\n\r\n").encode("latin1") + body)
        await up_w.drain()
        await pump(up_r, client_w)  # 响应流式回传(SSE 原样)
        up_w.close()
    except Exception as e:
        try:
            msg = json.dumps({"error": {"message": f"thinking-proxy: {e}"}}).encode()
            client_w.write(b"HTTP/1.1 502 Bad Gateway\r\ncontent-type: application/json\r\ncontent-length: " + str(len(msg)).encode() + b"\r\nconnection: close\r\n\r\n" + msg)
            await client_w.drain()
        except Exception:
            pass
    finally:
        try:
            client_w.close()
        except Exception:
            pass

async def main():
    servers = []
    for port, kwargs in PORTS.items():
        srv = await asyncio.start_server(
            lambda r, w, kw=kwargs: handle(r, w, kw), "0.0.0.0", port)
        servers.append(srv)
        print(f"thinking-proxy: :{port} -> :{UPSTREAM_PORT} inject={kwargs}", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
