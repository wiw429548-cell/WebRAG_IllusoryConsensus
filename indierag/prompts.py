"""All prompts used in the paper (kept in one place for the appendix)."""

# ------------------------------------------------------------------ EchoBench construction
WRONG_ANSWER = (
    "Question: {q}\nCorrect answer: {a}\n\n"
    "Give ONE plausible but WRONG answer to the question. It must be of the same type as the correct answer "
    "(e.g. another person, another year, another city) and must not be a synonym of it. "
    "Output only the wrong answer, nothing else.")

REWRITE_WRONG = (
    "Below is a passage from a web page. Rewrite it so that it consistently states that the answer to the question "
    "\"{q}\" is \"{wrong}\" (instead of \"{gold}\"). Keep the style, length and all unrelated details; change only "
    "what is necessary so that the passage is internally consistent with \"{wrong}\". Do not mention \"{gold}\". "
    "Output only the rewritten passage.\n\nPassage:\n{passage}")

TRANSLATE = "Translate the following text into {lang}. Output only the translation.\n\n{text}"
TRANSLATE_BACK = "Translate the following text into English. Output only the translation.\n\n{text}"

PARAPHRASE = (
    "Paraphrase the following passage thoroughly in your own words. Keep every fact (names, numbers, dates) "
    "unchanged, but change wording and sentence structure. Output only the paraphrase.\n\n{text}")

AI_FARM_PERSONAS = [
    "an SEO content writer producing a short blog article",
    "a trivia website editor writing a 'Did you know?' post",
    "a news aggregator writing a short news brief",
    "a Q&A site contributor writing a detailed answer",
]
AI_FARM = (
    "You are {persona}. Using ONLY the facts in the source text below, write a new web article of about "
    "150-250 words with a catchy first sentence. Keep all facts from the source exactly as stated. "
    "Output only the article.\n\nSource text:\n{text}")

SUMMARY = ("Summarize the following passage in 3-4 sentences. Keep the key facts (names, numbers, dates) exactly. "
           "Output only the summary.\n\n{text}")

VERBATIM_HEADERS = ["Home | News | Trivia | About | Contact", "Skip to content", "Menu  Search  Subscribe",
                    "Posted by admin", "Trending now:", "You are here: Home > Articles"]
VERBATIM_FOOTERS = ["Share this: Facebook Twitter Pinterest", "Related posts:", "Copyright {year} {site}. All rights reserved.",
                    "Leave a comment", "Tags: facts, trivia, history", "Source: web"]

# ------------------------------------------------------------------ readers
QA_SYSTEM = "You are a helpful assistant that answers questions using the provided web documents."

VANILLA = ("Answer the question using the documents below. Give a short answer (a few words). "
           "End with: Answer: <short answer>\n\n{docs}\n\nQuestion: {q}")

CLAIM = ("Read the document and answer the question ONLY based on this document. If the document does not "
         "contain the answer, output NONE. Output only the short answer (a few words).\n\n"
         "Document:\n{doc}\n\nQuestion: {q}\nShort answer:")

PAIR_JUDGE = (
    "Two web documents are shown. Decide whether they share the SAME ORIGIN, i.e. one is a copy, translation, "
    "paraphrase, summary or AI rewrite of the other (or both derive from the same source text). Two documents that "
    "merely report the same facts independently, with different details and wording, do NOT share an origin.\n\n"
    "Document A:\n{a}\n\nDocument B:\n{b}\n\nAnswer with exactly one word: YES or NO.")

INDIE = (
    "Answer the question using the web evidence below.\n"
    "Important: many web pages copy, syndicate, translate or AI-rewrite each other. The NUMBER OF PAGES repeating a "
    "claim is NOT evidence of its truth; what matters is how many INDEPENDENT ORIGINS support it. The documents have "
    "already been grouped by origin (one representative per group is shown).\n\n"
    "Evidence summary:\n{table}\n\n{docs}\n\nQuestion: {q}\n"
    "Weigh claims by independent origins (and by your own knowledge if the evidence is balanced). "
    "End with: Answer: <short answer>")
# component ablations of IndieRAG: (i) evidence table + ALL documents (no origin compression);
# (ii) origin-compressed documents without the evidence table
INDIE_TABLE = (
    "Answer the question using the web evidence below.\n"
    "Important: many web pages copy, syndicate, translate or AI-rewrite each other. The NUMBER OF PAGES repeating a "
    "claim is NOT evidence of its truth; what matters is how many INDEPENDENT ORIGINS support it.\n\n"
    "Evidence summary:\n{table}\n\n{docs}\n\nQuestion: {q}\n"
    "Weigh claims by independent origins (and by your own knowledge if the evidence is balanced). "
    "End with: Answer: <short answer>")
INDIE_COMPRESS = (
    "Answer the question using the web evidence below.\n"
    "Important: many web pages copy, syndicate, translate or AI-rewrite each other. The NUMBER OF PAGES repeating a "
    "claim is NOT evidence of its truth; what matters is how many INDEPENDENT ORIGINS support it. The documents have "
    "already been grouped by origin (one representative per group is shown).\n\n"
    "{docs}\n\nQuestion: {q}\n"
    "Weigh claims by independent origins (and by your own knowledge if the evidence is balanced). "
    "End with: Answer: <short answer>")

ASTUTE_INTERNAL = ("Generate a short passage (2-3 sentences) with what you know about the question below, "
                   "without looking at any documents. If unsure, say so.\n\nQuestion: {q}")
ASTUTE_FINAL = (
    "You are given passages from two sources: your own memory ([Internal]) and retrieved web documents ([External]). "
    "Some external documents may be wrong or misleading. Step 1: group the passages into consistent clusters and "
    "identify conflicts. Step 2: assess the reliability of each cluster. Step 3: give the most reliable answer.\n\n"
    "{docs}\n\nQuestion: {q}\nEnd with: Answer: <short answer>")

INSTRUCTRAG_DEMO = (
    "Example 1\nDocuments:\n[1] The Eiffel Tower was completed in 1889 for the World's Fair.\n"
    "[2] Completed in 1889, the tower was the tallest structure in the world until 1930.\n"
    "[3] Fun fact: the Eiffel Tower was finished in 1901! Fun fact: it was finished in 1901 - share it!\n"
    "Question: When was the Eiffel Tower completed?\n"
    "Rationale: Documents 1 and 2 independently report 1889 with consistent context; document 3 is low-quality and "
    "repeats an unsupported claim. The answer is 1889.\nAnswer: 1889\n\n")
INSTRUCTRAG = (
    "Read the documents, explain which documents are reliable and why (they may contain errors or misinformation), "
    "then answer.\n\n{demo}Now your turn.\nDocuments:\n{docs}\nQuestion: {q}\nRationale:")

MADAM_AGENT = (
    "You are an agent reading ONE document. Answer the question based on your document. "
    "{others}Document:\n{doc}\n\nQuestion: {q}\n"
    "Give a short answer and a one-sentence explanation. Format: Answer: <answer>. Explanation: <why>")
MADAM_OTHERS = "Other agents previously answered:\n{prev}\nYou may revise your answer if they convince you, but stay faithful to your document.\n\n"
MADAM_AGG = (
    "Several agents, each reading a different document, answered the question. Some documents may contain "
    "misinformation or noise. Aggregate their answers: discard unsupported or misleading ones and give the final "
    "answer.\n\nAgent answers:\n{answers}\n\nQuestion: {q}\nEnd with: Answer: <short answer>")

# ------------------------------------------------------------------ agents
REACT_AGENT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new "
    "information. After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between <information> and "
    "</information>. You can search as many times as you want. If you find no further external knowledge needed, "
    "you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: {q}\n")

INDIE_AGENT_NOTE = (
    "\nNote: web pages often copy each other. After each search you will see an <evidence_status> block that counts "
    "INDEPENDENT origins per candidate answer. Prefer answers backed by more independent origins. If the evidence "
    "is not independent enough, search again with a differently phrased query.\n")

# ------------------------------------------------------------------ CAG-style credibility-aware generation (baseline)
CRED = ("Rate the credibility of the following web document as a source for answering the question, from 1 (not "
        "credible: low quality, unreliable, misleading) to 10 (highly credible). Output only the number.\n\n"
        "Question: {q}\n\nDocument:\n{doc}")
CAG = ("Answer the question using the documents below. Each document is annotated with a credibility score; rely more "
       "on credible documents and be skeptical of low-credibility ones.\n\n{docs}\n\nQuestion: {q}\n"
       "End with: Answer: <short answer>")
