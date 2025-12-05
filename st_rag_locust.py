from locust import HttpUser, task, between
import random

GOBLET_OF_FIRE_QUERIES = [
    "Describe all three tasks of the Triwizard Tournament and what Harry has to do in each.",
    "How does Harry end up being selected as the fourth Triwizard champion despite being underage?",
    "List the dragons used in the first Triwizard task and explain which one Harry faces.",
    "How does Harry prepare for facing the dragon in the first task, and who helps him?",
    "Explain the clues and events that lead Harry to understand how to breathe underwater for the second task.",
    "Who are the hostages in the second Triwizard task and why is each one chosen?",
    "Describe the obstacles Harry encounters inside the maze during the third task.",
    "How does the Triwizard Cup turn out to be a Portkey and who arranged it?",
    "Summarize what happens in the graveyard when Voldemort returns to full strength.",
    "Explain the duel between Harry and Voldemort in the graveyard and the significance of the golden connection between their wands.",
    "How does Barty Crouch Jr. impersonate Moody and what clues are dropped throughout the book?",
    "Describe the role of Rita Skeeter in the story and how she gathers her information.",
    "How is house-elf rights and Hermione’s S.P.E.W. campaign portrayed in this book?",
    "What tensions arise between Harry and Ron after Harry’s name comes out of the Goblet of Fire?",
    "Describe the Yule Ball: how the partners are chosen, who goes with whom, and why the evening is important for character relationships.",
    "How does Dumbledore react to Harry’s recounting of the events in the graveyard and Voldemort’s return?",
    "Explain the significance of the Pensieve scene with Barty Crouch Jr.’s trial.",
    "What role does Ludo Bagman play in the book and why is he a suspicious character?",
    "How does the Quidditch World Cup set the tone for the return of Voldemort and his supporters?",
    "Describe the events surrounding the Dark Mark appearing at the Quidditch World Cup.",
]

DEATHLY_HALLOWS_QUERIES = [
    "Summarize the contents of Dumbledore’s will and what he leaves to Harry, Ron, and Hermione.",
    "Explain the significance of the seven Potters plan and what happens during the escape from Privet Drive.",
    "Describe the events at Bill and Fleur’s wedding and how the Death Eaters’ takeover interrupts it.",
    "How do Harry, Ron, and Hermione infiltrate the Ministry of Magic and what are they trying to retrieve?",
    "Explain how the trio discovers that the locket Horcrux is with Umbridge and how they steal it.",
    "Describe the effect of the locket Horcrux on Harry, Ron, and Hermione over time.",
    "Why does Ron leave the tent, and what brings him back to the group later?",
    "Explain how the Sword of Gryffindor appears to Harry in the frozen lake and how it destroys the locket.",
    "Summarize the story of the Three Brothers and how it explains the Deathly Hallows.",
    "Detail what each of the Deathly Hallows is and how they are connected to Harry and Voldemort.",
    "Describe how the trio breaks into Gringotts, including the role of Griphook and the protections in Bellatrix’s vault.",
    "Explain the events leading up to the Battle of Hogwarts and how Hogwarts prepares for the fight.",
    "List the Horcruxes destroyed in Deathly Hallows and who destroys each one.",
    "Describe Snape’s memories in the Pensieve and how they change Harry’s understanding of Snape.",
    "Explain Harry’s walk into the forest and his meeting with Voldemort there.",
    "How does Harry survive the Killing Curse in the forest, and what is revealed in the King’s Cross scene?",
    "Describe the final confrontation between Harry and Voldemort in the Great Hall and why the Elder Wand turns against Voldemort.",
    "Summarize what happens to the main characters after the Battle of Hogwarts as shown in the epilogue.",
    "Explain how the relationship between Harry and Dumbledore is reinterpreted through Rita Skeeter’s book and Aberforth’s comments.",
    "Describe the role of Kreacher and the other house-elves in Deathly Hallows, especially during the Battle of Hogwarts.",
]

# If you want one combined pool for Locust:
HP_QUERIES = GOBLET_OF_FIRE_QUERIES + DEATHLY_HALLOWS_QUERIES


class PotterRAGUser(HttpUser):
    wait_time = between(5, 10)

    # (optional but explicit)
    # host = "http://localhost:8000"
    # give server enough time (your NIM timeout is 120s)
    network_timeout = 130
    connection_timeout = 30

    def on_start(self):
        self.conversation_id = None
        self.user_external_id = f"locust-user-{id(self)}"

    @task
    def rag_turn(self):
        payload = {
            "user_external_id": self.user_external_id,
            "conversation_id": self.conversation_id,
            "question": random.choice(HP_QUERIES),
            "collection_name": "docs",
        }

        with self.client.post(
            "/chat",
            json=payload,
            name="hp_rag_turn",
            catch_response=True,
        ) as resp:

            # 1️⃣ network / timeout / connection error: status == 0
            if resp.status_code == 0:
                # resp.error holds the underlying requests exception
                resp.failure(f"NETWORK ERROR: {repr(getattr(resp, 'error', None))}")
                return

            # 2️⃣ normal HTTP errors (what your InferenceError → HTTPException path will produce)
            if resp.status_code != 200:
                text = resp.text or ""
                resp.failure(f"status={resp.status_code}, body={text[:200]}")
                return

            # 3️⃣ parse JSON and verify answer
            try:
                data = resp.json()
            except Exception as e:
                text = resp.text or ""
                resp.failure(f"json error: {repr(e)}, body={text[:200]}")
                return

            answer = data.get("answer") or data.get("content", "")
            if not answer:
                resp.failure("empty answer")