"""Поиск положений: BM25 по стемам + оценка релевантности (доля «веса» слов запроса, найденных в положении)."""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass

from app.config import ROLES
from app.services.kb import Chunk, KnowledgeBase
from app.services.nlp import content_stems, stems_match


@dataclass
class Hit:
    chunk: Chunk
    bm25: float
    relevance: float     # 0..1
    matched: int = 0     # сколько слов запроса нашлось в положении
    qlen: int = 0        # сколько значимых слов в запросе


class Retriever:
    K1, B = 1.5, 0.75

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._vocab_key: int | None = None
        self._vocab: list[str] = []

    def _vocabulary(self) -> list[str]:
        """Словарь стемов базы (пересобирается после перезагрузки базы)."""
        key = id(self.kb.chunks)
        if key != self._vocab_key:
            self._vocab = sorted({s for c in self.kb.chunks for s in c.stems if len(s) >= 4})
            self._vocab_key = key
        return self._vocab

    def _fix_typos(self, q: list[str]) -> list[str]:
        """Слова с опечатками (нет в базе, но очень похожи на слово из неё) заменяем на ближайшее слово базы."""
        vocab = self._vocabulary()
        if not vocab:
            return q
        known = set(vocab)
        out: list[str] = []
        for s in q:
            if len(s) >= 6 and s not in known and not any(stems_match(s, v) for v in vocab):
                near = [v for v in difflib.get_close_matches(s, vocab, n=3, cutoff=0.86)
                        if v[0] == s[0] and abs(len(v) - len(s)) <= 1]
                if near:
                    s = near[0]
            if s not in out:
                out.append(s)
        return out

    def _idf(self, stem: str, chunks: list[Chunk]) -> float:
        df = sum(1 for c in chunks if any(stems_match(stem, s) for s in c.stems))
        n = len(chunks) or 1
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, role: str = "employee", k: int = 4) -> tuple[list[Hit], float]:
        allowed = ROLES.get(role, {"public"})
        chunks = [c for c in self.kb.chunks if c.access in allowed]
        q = self._fix_typos(content_stems(query))
        if not q or not chunks:
            return [], 0.0
        avg_len = sum(len(c.stems) for c in chunks) / len(chunks)
        max_idf = math.log(1 + (len(chunks) + 0.5) / 0.5)
        idf = {s: self._idf(s, chunks) for s in q}
        # слово запроса, которого нет нигде в базе, считается «неизвестным» и тянет релевантность вниз
        weight = {s: (idf[s] if any(stems_match(s, cs) for c in chunks for cs in c.stems) else max_idf) for s in q}
        total_w = sum(weight.values())

        hits: list[Hit] = []
        for c in chunks:
            score, matched_w, matched_n = 0.0, 0.0, 0
            for s in q:
                tf = sum(1 for cs in c.stems if stems_match(s, cs))
                if tf:
                    matched_w += weight[s]
                    matched_n += 1
                    score += idf[s] * tf * (self.K1 + 1) / (tf + self.K1 * (1 - self.B + self.B * len(c.stems) / avg_len))
            if score > 0:
                hits.append(Hit(c, score, matched_w / total_w if total_w else 0.0, matched_n, len(q)))
        hits.sort(key=lambda h: h.bm25, reverse=True)
        top = hits[:k]
        overall = max((h.relevance for h in top), default=0.0)
        return top, overall
