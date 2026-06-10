"""
WebSocket client-correlation tests.

Two clients connect with distinct client_ids over /ws/{client_id}. A personal
message addressed to client A must reach ONLY client A; client B must not see
it. We also assert send_personal_message tolerates unknown client_ids (no
raise) — the regression that progress updates are correlated per-client rather
than broadcast to everyone.
"""

import pytest


def test_personal_message_targets_only_named_client(client):
    with client.websocket_connect("/ws/clientA") as wsA, \
         client.websocket_connect("/ws/clientB") as wsB:
        from app.services.websocket_manager import manager

        msg = {"type": "progress_update", "job_id": "gen_A", "status": "processing"}

        # Drive a personal send to A from within the app's portal so it runs on
        # the same event loop the connections live on.
        client.portal.call(manager.send_personal_message, msg, "clientA")

        received = wsA.receive_json()
        assert received["job_id"] == "gen_A"
        assert received["type"] == "progress_update"

        # B must not receive A's message. Send B its own message afterwards and
        # confirm B's first received frame is its own (proves A's did not leak).
        msgB = {"type": "progress_update", "job_id": "gen_B", "status": "processing"}
        client.portal.call(manager.send_personal_message, msgB, "clientB")
        receivedB = wsB.receive_json()
        assert receivedB["job_id"] == "gen_B"


def test_send_to_unknown_client_is_tolerated(client):
    """send_personal_message to a client that never connected must not raise."""
    with client.websocket_connect("/ws/onlyclient") as ws:
        from app.services.websocket_manager import manager

        # No connection for "ghost" -> should be a silent no-op.
        client.portal.call(manager.send_personal_message, {"x": 1}, "ghost")

        # The live client is unaffected and still receives its own message.
        client.portal.call(manager.send_personal_message, {"hello": "onlyclient"}, "onlyclient")
        assert ws.receive_json() == {"hello": "onlyclient"}


def test_broadcast_reaches_all_connected_clients(client):
    with client.websocket_connect("/ws/c1") as ws1, \
         client.websocket_connect("/ws/c2") as ws2:
        from app.services.websocket_manager import manager

        client.portal.call(manager.broadcast, {"type": "ping"})

        assert ws1.receive_json() == {"type": "ping"}
        assert ws2.receive_json() == {"type": "ping"}
