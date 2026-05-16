"""
Multi-user, multi-conversation Locust workload for the HP4-indexed RAG chat.

Rate-tuned for NVIDIA NIM's 40 RPM cap on the LLM + embedding models:
  N users * (60 / wait_seconds) ≈ 40 RPM

Defaults (--users 8, wait 10-15s) → ~38-48 RPM. Keep --users ≤ 8 to avoid 429s.

Run:
  locust -f mt_rag_locust.py --host http://localhost:8000 \
         --users 8 --spawn-rate 1 -t 10m --headless

  # Or with the UI:
  locust -f mt_rag_locust.py --host http://localhost:8000

Each locust user:
  - Picks a sticky `user_external_id` from a pool of N personas.
  - Maintains a list of conversation_ids and rotates between them (turning each
    into a multi-turn session). 10% of turns start a fresh conversation.
  - 10% of turns are sent with debug=true (those traces will show
    retrieved/reranked/prompt in the Langfuse UI debug section).
"""

import os
import random

from locust import HttpUser, between, task

# ---------------------------------------------------------------------------
# Tunables (override via env if you like)
# ---------------------------------------------------------------------------
USER_POOL_SIZE = int(os.getenv("LOCUST_USER_POOL_SIZE", "20"))
MAX_CONVERSATIONS_PER_USER = int(os.getenv("LOCUST_MAX_CONVOS_PER_USER", "3"))
NEW_CONVERSATION_PROB = float(os.getenv("LOCUST_NEW_CONV_PROB", "0.10"))
DEBUG_PROB = float(os.getenv("LOCUST_DEBUG_PROB", "0.10"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "docs")

# Personas the locust user pool will draw from. Multiple locust workers can
# end up sharing a persona — that's the multi-session-per-user behavior.
USER_NAMES = [
    "alice", "bob", "carol", "dave", "eve", "frank", "grace", "harry",
    "ivy", "jack", "kate", "leo", "maya", "noah", "olivia", "peter",
    "quinn", "rachel", "sam", "tina",
][:USER_POOL_SIZE]


# ---------------------------------------------------------------------------
# HP4-only question pool, calibrated to stress different retrieval modes.
# ---------------------------------------------------------------------------

# Single-chunk factual (BM25 strong)
FACTUAL = [
    "Who are the three Triwizard champions originally selected by the Goblet?",
    "What is the password to the prefects' bathroom?",
    "What does the Hungarian Horntail look like?",
    "What spell does Harry use against his dragon in the first task?",
    "What is in the egg Harry receives after the first task?",
    "Who actually puts Harry's name in the Goblet of Fire?",
]

# Proper-noun heavy (hybrid retrieval should shine)
PROPER_NOUN = [
    "What does Ludo Bagman do for a living?",
    "Tell me about Bertha Jorkins.",
    "Who is Walden Macnair?",
    "What does Hassan Mostafa do?",
    "Who is Mr. Roberts?",
    "What happens to Winky in the woods?",
    "Tell me about Madame Maxime.",
    "What is the Sneakoscope and how is it used?",
    "Tell me about the Foe-Glass.",
    "What is the Pensieve?",
]

# Multi-chunk synthesis (rerank + bigger TOP_K helps)
SYNTHESIS = [
    "Walk me through the three Triwizard tasks in order.",
    "How does Barty Crouch Jr. impersonate Mad-Eye Moody throughout the year?",
    "Describe the events at the Quidditch World Cup before the Death Eater attack.",
    "How does Voldemort return to a body in the graveyard?",
    "What is the role of the Ministry of Magic throughout the book?",
]

# Character / relationship
CHARACTER = [
    "What is the relationship between Hermione and Viktor Krum?",
    "Why is Ron upset with Harry after the Goblet picks him?",
    "How does Rita Skeeter describe Harry in her articles?",
    "How does Mad-Eye Moody act in his first Defence Against the Dark Arts class?",
    "How does the wizarding community react to the events at the Quidditch World Cup?",
]

# Conversational follow-ups (tests history handling — see FOLLOWUP_THREADS below)
FOLLOWUP_THREADS = [
    [
        "Tell me about the Yule Ball.",
        "Who did Harry take?",
        "What about Ron?",
        "How did the evening end?",
    ],
    [
        "Describe the first task of the Triwizard Tournament.",
        "What dragon did Harry get?",
        "How did he get past it?",
        "What did he win?",
    ],
    [
        "Who is Cedric Diggory?",
        "How does he die?",
        "How does Harry react to it?",
    ],
]

SINGLE_TURN_QUESTIONS = FACTUAL + PROPER_NOUN + SYNTHESIS + CHARACTER


class PotterRAGUser(HttpUser):
    # ~12.5s mean wait → 4.8 req/min per user
    # With 8 users that's ~38 req/min, comfortably under NVIDIA's 40 RPM cap.
    wait_time = between(10, 15)
    network_timeout = 130
    connection_timeout = 30

    def on_start(self):
        # Pick a sticky persona from the pool. Multiple locust workers may
        # land on the same persona — that's intentional multi-session behavior.
        self.user_external_id = random.choice(USER_NAMES)
        # Tracks the conversations this locust worker has opened.
        self.conversation_ids: list[str] = []
        # When in the middle of a scripted multi-turn thread:
        self.active_thread: list[str] | None = None
        self.thread_index: int = 0

    # --------------------------- helpers ----------------------------------

    def _pick_or_create_conversation(self) -> str | None:
        """
        With prob NEW_CONVERSATION_PROB or when we have no conversations yet,
        return None (the API will mint a new conversation_id and we'll capture
        it from the response). Otherwise rotate through the existing ones.
        """
        if not self.conversation_ids:
            return None
        if random.random() < NEW_CONVERSATION_PROB and len(self.conversation_ids) < MAX_CONVERSATIONS_PER_USER:
            return None
        return random.choice(self.conversation_ids)

    def _pick_question(self) -> str:
        """
        ~25% of the time, walk through a scripted follow-up thread to stress
        history handling. Rest of the time, single-turn question.
        """
        # Continue an ongoing thread
        if self.active_thread is not None and self.thread_index < len(self.active_thread):
            q = self.active_thread[self.thread_index]
            self.thread_index += 1
            if self.thread_index >= len(self.active_thread):
                self.active_thread = None
                self.thread_index = 0
            return q

        # Start a new thread sometimes
        if random.random() < 0.25:
            self.active_thread = random.choice(FOLLOWUP_THREADS)
            self.thread_index = 1
            return self.active_thread[0]

        return random.choice(SINGLE_TURN_QUESTIONS)

    # --------------------------- main task --------------------------------

    @task
    def rag_turn(self):
        conv_id = self._pick_or_create_conversation()
        question = self._pick_question()
        debug = random.random() < DEBUG_PROB

        payload = {
            "user_external_id": self.user_external_id,
            "question": question,
            "collection_name": COLLECTION_NAME,
            "debug": debug,
        }
        if conv_id is not None:
            payload["conversation_id"] = conv_id

        # Tag the locust UI breakdown by question type for easier eyeballing.
        name = "hp_rag_turn_debug" if debug else "hp_rag_turn"

        with self.client.post(
            "/chat",
            json=payload,
            name=name,
            catch_response=True,
        ) as resp:
            if resp.status_code == 0:
                resp.failure(f"NETWORK ERROR: {repr(getattr(resp, 'error', None))}")
                return

            if resp.status_code == 429:
                # Hit the NIM rate limit — slow down naturally via wait_time.
                resp.failure("429 from upstream (NVIDIA RPM cap)")
                return

            if resp.status_code != 200:
                text = (resp.text or "")[:200]
                resp.failure(f"status={resp.status_code}, body={text}")
                return

            try:
                data = resp.json()
            except Exception as e:
                resp.failure(f"json error: {repr(e)}")
                return

            new_conv_id = data.get("conversation_id")
            if new_conv_id and new_conv_id not in self.conversation_ids:
                if len(self.conversation_ids) >= MAX_CONVERSATIONS_PER_USER:
                    self.conversation_ids.pop(0)
                self.conversation_ids.append(new_conv_id)

            answer = data.get("answer", "")
            if not answer:
                resp.failure("empty answer")
