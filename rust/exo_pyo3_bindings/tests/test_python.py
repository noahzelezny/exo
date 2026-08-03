import asyncio

import pytest
from _pytest.capture import CaptureFixture
from exo_pyo3_bindings import (
    Keypair,
    NetworkingHandle,
    NoPeersSubscribedToTopicError,
    Pidfile,
    PyFromSwarm,
)


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle(Keypair.generate(), [], 0)

    rt = asyncio.create_task(_await_recv(h))

    # sleep for 4 ticks
    for i in range(4):
        await asyncio.sleep(1)

        try:
            await h.gossipsub_publish("topic", b"somehting or other")
        except NoPeersSubscribedToTopicError as e:
            print("caught it", e)


def test_pidfile(capsys: CaptureFixture[str]):
    with capsys.disabled():
        print("\nbefore python")
        scoped_lock_file()
        print("after python")


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case PyFromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case PyFromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")


def scoped_lock_file():
    a = Pidfile("/tmp/lock.pid", 0o0600)


async def _wait_connected(h: NetworkingHandle, deadline: float) -> str:
    """Drain events until a Connection{connected=True}; return the peer id."""
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError("no Connection event before deadline")
        event = await asyncio.wait_for(h.recv(), timeout=remaining)
        match event:
            case PyFromSwarm.Connection(connected=True) as c:
                return c.peer_id
            case _:
                continue


@pytest.mark.asyncio
async def test_bootstrap_pairing_without_mdns() -> None:
    """Peers must pair via bootstrap dialing alone, with mDNS disabled and
    listeners scoped to loopback (listen_ips) — the sealed-ring configuration."""
    port_a, port_b = 42750, 42751
    a = NetworkingHandle(Keypair.generate(), [], port_a, ["127.0.0.1"], False)
    b = NetworkingHandle(
        Keypair.generate(),
        [f"/ip4/127.0.0.1/tcp/{port_a}"],
        port_b,
        ["127.0.0.1"],
        False,
    )

    deadline = asyncio.get_event_loop().time() + 20.0
    peer_of_a, peer_of_b = await asyncio.gather(
        _wait_connected(a, deadline), _wait_connected(b, deadline)
    )
    assert peer_of_a and peer_of_b


@pytest.mark.asyncio
async def test_legacy_three_arg_constructor_still_works() -> None:
    """Callers that predate listen_ips/enable_mdns keep the historical behavior."""
    h = NetworkingHandle(Keypair.generate(), [], 0)
    assert h is not None
