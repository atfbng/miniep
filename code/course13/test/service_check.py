"""Check health, nonstreaming, streaming, and concurrent completion requests."""

import argparse
import concurrent.futures
import json
import urllib.request


def request(base: str, stream: bool = False) -> dict:
    payload = dict(
        model="miniep-qwen",
        prompt="The capital of France is",
        temperature=0,
        max_tokens=8,
        stream=stream,
    )
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        if not stream:
            result = json.load(response)
            assert result["choices"] and result["usage"]["completion_tokens"] > 0
            return dict(stream=False, text=result["choices"][0]["text"])
        pieces = []
        finished = False
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                finished = True
            elif line.startswith("data: "):
                item = json.loads(line[6:])
                pieces.extend(c.get("text", "") for c in item["choices"])
        assert finished and pieces
        return dict(stream=True, text="".join(pieces))


def chat_request(base: str) -> dict:
    payload = dict(
        model="miniep-qwen",
        messages=[
            dict(role="user", content="Explain mixture of experts in one sentence.")
        ],
        temperature=0,
        max_tokens=16,
        chat_template_kwargs={"enable_thinking": False},
    )
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    assert (
        result["choices"][0]["message"]["content"]
        and result["usage"]["completion_tokens"] > 0
    )
    return dict(chat=True, text=result["choices"][0]["message"]["content"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8133")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with urllib.request.urlopen(a.url + "/health", timeout=10) as response:
        assert response.status == 200
    results = [request(a.url), request(a.url, True), chat_request(a.url)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results.extend(pool.map(lambda _: request(a.url), range(4)))
    from pathlib import Path

    Path(a.output).write_text(json.dumps(dict(passed=True, requests=results), indent=2))
    print("PASS: health, completion, streaming, chat, four concurrent requests")


if __name__ == "__main__":
    main()
