"""Tests for SocraticDialogue.stream_agent_reply (M2 dialogue-side wiring)."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from openbiliclaw.agent.loop import AgentEvent, AgentLoop
from openbiliclaw.agent.tools import Tool, ToolRegistry
from openbiliclaw.llm.base import LLMResponse
from openbiliclaw.llm.service import LLMResponseContentError
from openbiliclaw.soul.dialogue import (
    DialogueLearningMode,
    SocraticDialogue,
)


class FakeAgentLLM:
    """Service-shaped double returning queued LLMResponses."""

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self._responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_with_native_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        caller: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        bypass_semaphore: bool = False,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "caller": caller,
            }
        )
        item = self._responses.popleft()
        if isinstance(item, Exception):
            raise item
        return item


class RecordingSettlementQueue:
    """Settlement-queue double capturing learn submissions."""

    def __init__(self) -> None:
        self.submissions: list[tuple[Any, dict[str, object]]] = []

    def submit(self, kind: Any, payload: dict[str, object]) -> object:
        self.submissions.append((kind, payload))
        return object()


def _dialogue(
    service: Any,
    *,
    mode: DialogueLearningMode = DialogueLearningMode.REPLY_ONLY_TEST,
    queue: Any = None,
) -> SocraticDialogue:
    return SocraticDialogue(
        llm=None,
        soul_engine=object(),
        llm_service=service,
        session="popup",
        learning_mode=mode,
        settlement_queue=queue,
    )


def _loop(
    responses: list[LLMResponse | Exception],
    *,
    tools: ToolRegistry | None = None,
) -> tuple[AgentLoop, FakeAgentLLM]:
    llm = FakeAgentLLM(responses)
    registry = tools or ToolRegistry(
        [
            Tool(
                name="get_profile",
                description="读取画像",
                handler=lambda _args: "画像：喜欢机械键盘",
            )
        ]
    )
    return AgentLoop(llm, registry), llm


async def _collect(stream: Any) -> list[AgentEvent]:
    return [event async for event in stream]


class TestStreamAgentReply:
    async def test_streams_loop_events_and_records_history(self) -> None:
        loop, llm = _loop(
            [
                LLMResponse(
                    content="我先看看你的画像",
                    tool_calls=[{"id": "c1", "name": "get_profile", "arguments": {}}],
                ),
                LLMResponse(content="你最近很喜欢机械键盘相关的视频呢"),
            ]
        )
        dialogue = _dialogue(object())

        events = await _collect(dialogue.stream_agent_reply(loop, "我最近表现怎么样？"))

        assert [event.type for event in events] == [
            "thinking",
            "tool_call",
            "tool_result",
            "final",
        ]
        assert events[-1].text == "你最近很喜欢机械键盘相关的视频呢"
        # The system prompt comes from the socratic persona builder.
        first_call = llm.calls[0]
        assert first_call["messages"][0]["role"] == "system"
        assert "OpenBiliClaw" in first_call["messages"][0]["content"]
        # History recorded exactly once: user turn + agent reply.
        history = dialogue.history
        assert [turn.role for turn in history] == ["user", "agent"]
        assert history[0].content == "我最近表现怎么样？"
        assert history[1].content == "你最近很喜欢机械键盘相关的视频呢"

    async def test_queued_learning_receives_turn_payload(self) -> None:
        loop, _llm = _loop([LLMResponse(content="好的")])
        queue = RecordingSettlementQueue()
        dialogue = _dialogue(
            object(),
            mode=DialogueLearningMode.QUEUED,
            queue=queue,
        )

        events = await _collect(
            dialogue.stream_agent_reply(
                loop,
                "帮我记一下",
                session="desktop",
                scope="chat",
                turn_id="turn-1",
            )
        )

        assert [event.type for event in events] == ["final"]
        assert len(queue.submissions) == 1
        _kind, payload = queue.submissions[0]
        assert payload == {
            "user_message": "帮我记一下",
            "assistant_reply": "好的",
            "session": "desktop",
            "scope": "chat",
            "turn_id": "turn-1",
        }

    async def test_llm_failure_rolls_back_history(self) -> None:
        loop, _llm = _loop([RuntimeError("provider down")])
        dialogue = _dialogue(object())

        with pytest.raises(RuntimeError, match="provider down"):
            await _collect(dialogue.stream_agent_reply(loop, "你好"))

        assert dialogue.history == []

    async def test_empty_final_is_an_error_and_rolls_back(self) -> None:
        loop, _llm = _loop([LLMResponse(content="   ")])
        dialogue = _dialogue(object())

        with pytest.raises(LLMResponseContentError):
            await _collect(dialogue.stream_agent_reply(loop, "你好"))

        assert dialogue.history == []

    async def test_system_prompt_carries_agent_ground_rules(self) -> None:
        loop, llm = _loop([LLMResponse(content="好的")])
        dialogue = _dialogue(object())

        await _collect(dialogue.stream_agent_reply(loop, "你好"))

        system = llm.calls[0]["messages"][0]["content"]
        # 工具纪律: never narrate/fabricate tool calls in prose (issue 4).
        assert "工具纪律" in system
        assert "严禁编造工具调用" in system
        # 记忆归属: shared memory base, no unfounded session claims (issue 5).
        assert "记忆归属" in system
        assert "跨会话共享" in system
        # 会话边界: "本对话/第一回合" means the current session (issue 6).
        assert "会话边界" in system
        assert "当前会话" in system


async def test_agent_context_isolated_per_conversation() -> None:
    loop, llm = _loop(
        [
            LLMResponse(content="记住了，A 的暗号是海豚"),
            LLMResponse(content="B 是新对话"),
            LLMResponse(content="A 的暗号是海豚"),
        ]
    )
    dialogue = _dialogue(object())
    await _collect(dialogue.stream_agent_reply(loop, "A 的暗号是海豚", session_id="A"))
    await _collect(dialogue.stream_agent_reply(loop, "B 的第一条消息", session_id="B"))
    assert not any("海豚" in item["content"] for item in llm.calls[1]["messages"])
    await _collect(dialogue.stream_agent_reply(loop, "我的暗号是什么", session_id="A"))
    assert any("海豚" in item["content"] for item in llm.calls[2]["messages"])
    assert not any("B 的第一条" in item["content"] for item in llm.calls[2]["messages"])


async def test_agent_restores_only_own_durable_conversation(tmp_path: Any) -> None:
    from openbiliclaw.storage.database import Database

    database = Database(tmp_path / "dialogue.db")
    database.initialize()
    for session_id in ("A", "B"):
        database.create_chat_session(session_id=session_id)
        database.create_chat_turn(
            turn_id=session_id, session_id=session_id, message=f"{session_id} 的独有历史"
        )
        database.complete_chat_turn(session_id, reply=f"收到 {session_id}")
    loop, llm = _loop([LLMResponse(content="好的")])
    dialogue = SocraticDialogue(
        llm=None,
        soul_engine=object(),
        llm_service=object(),
        database=database,
        session="popup",
        learning_mode=DialogueLearningMode.REPLY_ONLY_TEST,
    )
    await _collect(dialogue.stream_agent_reply(loop, "继续", session_id="B"))
    assert any("B 的独有历史" in item["content"] for item in llm.calls[0]["messages"])
    assert not any("A 的独有历史" in item["content"] for item in llm.calls[0]["messages"])


async def test_agent_bound_reply_keeps_context_and_learning_anchor() -> None:
    from openbiliclaw.soul.dialogue_turn_context import DialogueTurnBinding, DialogueTurnContext

    binding = DialogueTurnBinding.from_context(
        DialogueTurnContext(
            reply_to_turn_id="card-1",
            source_type="card",
            kind="hypothesis",
            ref="h1",
            generation=1,
            anchor_origin_turn_id="card-1",
            title="你喜欢安静的科普",
        )
    )

    class BoundQueue(RecordingSettlementQueue):
        def submit(self, kind: Any, payload: dict[str, object], **kwargs: Any) -> object:
            self.anchor = kwargs["_server_frozen_anchor_snapshot"]
            return super().submit(kind, payload)

    queue = BoundQueue()
    dialogue = _dialogue(object(), mode=DialogueLearningMode.QUEUED, queue=queue)
    loop, llm = _loop([LLMResponse(content="我记住了")])
    await _collect(dialogue.stream_agent_reply(loop, "是的", dialogue_binding=binding))
    assert "你喜欢安静的科普" in llm.calls[0]["messages"][-1]["content"]
    assert queue.submissions[0][1]["dialogue_binding"] == binding.to_mapping()
    assert queue.anchor.ref == "h1"
