# Eval run — 2026-05-20 05:09 UTC

- **Generator:** `google/gemma-4-31b-it`
- **Judge:** `meta/llama-3.3-70b-instruct (default)` (different model family — see [docs/PROGRESS.md](../PROGRESS.md))
- **Q&A count:** 18

## Aggregate

| Recall@5 | MRR | Groundedness | Accuracy | Completeness | p50 latency | p95 latency |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.60 | 3.89/5 | 3.94/5 | 2.11/5 | 8370ms | 39755ms |

## By book

| Book | Count | Recall@5 | MRR | Ground | Acc | Comp |
|---|---:|---:|---:|---:|---:|---:|
| `hp4` | 9 | 0.83 | 0.94 | 4.78 | 4.89 | 3.44 |
| `hp7` | 9 | 0.17 | 0.26 | 3.00 | 3.00 | 0.78 |

## By category

| Category | Count | Recall@5 | Ground | Acc | Comp |
|---|---:|---:|---:|---:|---:|
| `character` | 4 | 0.62 | 3.75 | 4.00 | 2.75 |
| `factual` | 1 | 1.00 | 5.00 | 5.00 | 3.00 |
| `magic` | 4 | 0.50 | 5.00 | 5.00 | 2.50 |
| `plot` | 7 | 0.36 | 3.43 | 3.43 | 1.29 |
| `reasoning` | 2 | 0.50 | 3.00 | 3.00 | 2.50 |

## Per question

| ID | Book | Category | Recall@5 | Ground | Acc | Comp | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| `hp4-001` | hp4 | character | 1.00 | 5 | 5 | 5 | 55356ms |
| `hp4-002` | hp4 | plot | 1.00 | 5 | 5 | 1 | 37123ms |
| `hp4-003` | hp4 | magic | 1.00 | 5 | 5 | 4 | 13532ms |
| `hp4-004` | hp4 | plot | 0.50 | 4 | 4 | 2 | 31447ms |
| `hp4-005` | hp4 | character | 0.50 | 4 | 5 | 4 | 39755ms |
| `hp4-006` | hp4 | factual | 1.00 | 5 | 5 | 3 | 16672ms |
| `hp4-007` | hp4 | magic | 1.00 | 5 | 5 | 4 | 7599ms |
| `hp4-008` | hp4 | plot | 0.50 | 5 | 5 | 4 | 11485ms |
| `hp4-009` | hp4 | reasoning | 1.00 | 5 | 5 | 4 | 5053ms |
| `hp7-001` | hp7 | magic | 0.00 | 5 | 5 | 1 | 22261ms |
| `hp7-002` | hp7 | plot | 0.00 | 5 | 5 | 1 | 5936ms |
| `hp7-003` | hp7 | magic | 0.00 | 5 | 5 | 1 | 5541ms |
| `hp7-004` | hp7 | character | 0.50 | 5 | 5 | 1 | 4430ms |
| `hp7-005` | hp7 | plot | 0.50 | 5 | 5 | 1 | 4831ms |
| `hp7-006` | hp7 | character | 0.50 | 1 | 1 | 1 | 8370ms |
| `hp7-007` | hp7 | reasoning | 0.00 | 1 | 1 | 1 | 15370ms |
| `hp7-008` | hp7 | plot | 0.00 | 0 | 0 | 0 | 0ms |
| `hp7-009` | hp7 | plot | 0.00 | 0 | 0 | 0 | 0ms |

## Failures (11)

Rows where retrieval recall@5 < 50% or any judge sub-score < 3/5.

### `hp4-002` (hp4, plot)

**Question:** What are the three tasks of the Triwizard Tournament in book 4?

**Answer:** The context doesn't specify the exact three tasks of the Triwizard Tournament in book 4, but it does mention that there will be three tasks, spaced throughout the school year, and they will test the champions in many different ways, including their magical prowess, daring, powers of deduction, and ability to cope with danger. It also mentions that the third task takes place in the evening, but it doesn't provide any details about what the tasks entail.

**Scores:** recall@5=1.00, ground=5, acc=5, comp=1

**Judge reasoning:** The answer is grounded and accurate because it correctly states that the context does not specify the exact tasks, but it lacks completeness as it does not mention any of the expected keywords such as dragon, lake, maze, or merpeople, which are implied to be part of the tasks.

### `hp4-004` (hp4, plot)

**Question:** How does Voldemort return to a physical body at the end of book 4?

**Answer:** Voldemort returns to a physical body at the end of book 4 with the help of Wormtail, who follows his instructions to obtain the essential ingredients for his rebirth. Although the exact process is not fully detailed in the provided context, it's mentioned that Wormtail's actions allow Voldemort to inhabit a rudimentary, weak body of his own, which he can use until he can gather the necessary ingredients for a true rebirth.

**Scores:** recall@5=0.50, ground=4, acc=4, comp=2

**Judge reasoning:** The answer is grounded in the context as it mentions Wormtail's role in Voldemort's return, but it lacks completeness as it doesn't touch on the expected keywords like Pettigrew, ritual, bone, flesh, and blood, which are crucial to understanding the process of Voldemort's rebirth.

### `hp7-001` (hp7, magic)

**Question:** What is a Horcrux and how many did Voldemort create?

**Answer:** Unfortunately, the provided context doesn't explicitly explain what a Horcrux is or how many Voldemort created. However, it does mention Voldemort's attempts to regain power and his connection to Harry through the curse that failed, which might be related to the concept of Horcruxes. To answer your question, I would need more information from the document, but based on the given context, I can't provide a direct answer about Horcruxes or their number.

**Scores:** recall@5=0.00, ground=5, acc=5, comp=1

**Judge reasoning:** The answer is grounded and accurate because it correctly states that the provided context does not explicitly explain what a Horcrux is or how many Voldemort created, but it lacks completeness as it fails to provide any information about Horcruxes or their relation to Voldemort's immortality.

### `hp7-002` (hp7, plot)

**Question:** What is the Battle of the Seven Potters and what is its purpose?

**Answer:** The Battle of the Seven Potters is not mentioned in the provided context. The context appears to be excerpts from a story about Harry Potter, but it does not mention a specific event called the Battle of the Seven Potters. It discusses various scenes and conversations between characters, including Harry, Dumbledore, and Snape, but does not provide information about this particular battle. If you're looking for information about a specific event in the Harry Potter series, you might want to ask about a different topic or provide more context.

**Scores:** recall@5=0.00, ground=5, acc=5, comp=1

**Judge reasoning:** The answer is grounded and accurate because it correctly states that the Battle of the Seven Potters is not mentioned in the provided context, but it lacks completeness as it does not provide any information about the actual event, which is a significant plot point in the Harry Potter series involving Polyjuice, decoys, Death Eaters, and transport.

### `hp7-003` (hp7, magic)

**Question:** What are the three Deathly Hallows and what does each one do?

**Answer:** The context provided doesn't mention the Deathly Hallows or their functions. It discusses the Unforgivable Curses, the Triwizard Tournament, and other topics, but doesn't give any information about the Deathly Hallows. If you're looking for information on the Deathly Hallows, you might want to ask a different question or provide more context.

**Scores:** recall@5=0.00, ground=5, acc=5, comp=1

**Judge reasoning:** The answer is grounded and accurate because it correctly states that the provided context does not mention the Deathly Hallows, but it is incomplete because it does not provide any information about the Deathly Hallows, which are the Elder Wand, Resurrection Stone, and Invisibility Cloak.

### `hp7-004` (hp7, character)

**Question:** How does Dobby die in the Deathly Hallows?

**Answer:** From what's here, we can't tell how Dobby dies in the Deathly Hallows. The passages provided show Dobby's interactions with Harry, but they don't mention his death. To find out how Dobby dies, you might want to ask about a specific scene or chapter in the book.

**Scores:** recall@5=0.50, ground=5, acc=5, comp=1

**Judge reasoning:** The answer is grounded and accurate because it correctly states that the provided passages do not mention Dobby's death, but it is incomplete because it does not provide any information about Dobby's death, which is the main question being asked, and does not touch on the expected keywords.

### `hp7-005` (hp7, plot)

**Question:** How does the trio break into Gringotts and what Horcrux do they retrieve?

**Answer:** The provided context doesn't mention the trio breaking into Gringotts or retrieving a Horcrux. It appears to be a collection of scenes from Harry Potter's life at Hogwarts, including interactions with friends and professors, but it doesn't contain information about a Gringotts heist or Horcrux retrieval. If you're looking for information on that specific event, it might be found in a different part of the document.

**Scores:** recall@5=0.50, ground=5, acc=5, comp=1

**Judge reasoning:** The answer is grounded and accurate because it correctly states that the provided context does not mention the trio breaking into Gringotts or retrieving a Horcrux, but it is incomplete because it does not provide any information about the event, which is not present in the context, and does not touch on the expected keywords.

### `hp7-006` (hp7, character)

**Question:** What does Snape's memory reveal about his loyalty in The Prince's Tale?

**Answer:** In The Prince's Tale, Snape's memory reveals that he was a spy for Dumbledore, working against Voldemort at great personal risk. This suggests that Snape's loyalty ultimately lay with Dumbledore and the Order of the Phoenix, rather than with the Death Eaters or Voldemort. Although his expression and behavior often seemed sour and unpleasant, his actions and Dumbledore's trust in him imply that he was working to undermine Voldemort's plans.

**Scores:** recall@5=0.50, ground=1, acc=1, comp=1

**Judge reasoning:** No context or answer was provided to evaluate.

### `hp7-007` (hp7, reasoning)

**Question:** Why is Harry the true master of the Elder Wand at the end of book 7?

**Answer:** From what's here, it appears that Harry becomes the true master of the Elder Wand at the end of book 7 because of a complex series of events involving wand lore and the concept of Priori Incantatem. When Voldemort's wand and Harry's wand connected, it caused a flood of memories to emerge, indicating a deep magical connection between the two. However, the exact reason why Harry becomes the master of the Elder Wand isn't explicitly stated in these passages. It's likely related to the fact that Draco Malfoy disarmed Dumbledore, making him the master of the Elder Wand, and then Harry disarmed Draco, transferring the mastery to him. But this specific information isn't mentioned in the provided context.

**Scores:** recall@5=0.00, ground=1, acc=1, comp=1

**Judge reasoning:** No context or answer was provided to evaluate.

### `hp7-008` (hp7, plot)

**Question:** How is the locket Horcrux destroyed?

**Answer:** 

**Scores:** recall@5=0.00, ground=0, acc=0, comp=0

**Judge reasoning:** 

### `hp7-009` (hp7, plot)

**Question:** What happens to Harry in the Forbidden Forest when he confronts Voldemort in book 7?

**Answer:** 

**Scores:** recall@5=0.00, ground=0, acc=0, comp=0

**Judge reasoning:** 

