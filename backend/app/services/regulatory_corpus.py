from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from backend.app.core.config import Settings
from backend.app.core.schemas import EvidenceChunk

logger = logging.getLogger(__name__)


@dataclass
class SourceManifest:
    source: str
    jurisdiction: str
    url: str
    document_version: str
    fetched_at: str
    content_sha256: str
    character_count: int
    chunk_count: int


@dataclass
class CorpusLoadResult:
    chunks: list[EvidenceChunk]
    warnings: list[str]
    source_manifests: list[dict]
    corpus_sha256: str


class RegulatoryCorpusLoader:
    """Fetch official FCA/EUR-Lex/CFPB text and chunk it only in RAM."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._encoding = None

    def load(self) -> CorpusLoadResult:
        chunks: list[EvidenceChunk] = []
        warnings: list[str] = []
        manifests: list[SourceManifest] = []
        headers = {
            "User-Agent": "FinLingoPP-AcademicResearch/1.0",
            "Accept": "text/html,application/pdf,text/plain;q=0.9,*/*;q=0.8",
        }
        with httpx.Client(
            timeout=self.settings.regulatory_request_timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            for spec in self.settings.regulatory_sources:
                try:
                    count = 0
                    for url, text, version, fetched_at, content_hash in self._fetch_source_documents(client, spec):
                        document_chunks = list(
                            self._chunk_document(
                                spec,
                                url,
                                text,
                                version,
                                fetched_at=fetched_at,
                                content_hash=content_hash,
                            )
                        )
                        chunks.extend(document_chunks)
                        count += len(document_chunks)
                        manifests.append(
                            SourceManifest(
                                source=spec["source"],
                                jurisdiction=spec["jurisdiction"],
                                url=url,
                                document_version=version,
                                fetched_at=fetched_at,
                                content_sha256=content_hash,
                                character_count=len(text),
                                chunk_count=len(document_chunks),
                            )
                        )
                    if not count:
                        warnings.append(f"No usable regulatory text extracted from {spec['source']}")
                    logger.info("Loaded %d in-memory chunks from %s", count, spec["source"])
                except Exception as exc:
                    message = f"Could not load {spec['source']} from {spec['url']}: {exc}"
                    logger.warning(message)
                    warnings.append(message)
        if not chunks:
            raise RuntimeError("No live regulatory corpus could be loaded; hardcoded evidence is never substituted")
        corpus_hash = hashlib.sha256(
            "\n".join(sorted(manifest.content_sha256 for manifest in manifests)).encode("utf-8")
        ).hexdigest()
        return CorpusLoadResult(chunks, warnings, [asdict(item) for item in manifests], corpus_hash)

    def _fetch_source_documents(
        self, client: httpx.Client, spec: dict[str, str]
    ) -> list[tuple[str, str, str, str, str]]:
        root_url = spec["url"]
        cache_path_value = spec.get("cache_path")
        if cache_path_value:
            cache_path = self.settings.resolve(Path(cache_path_value))
            if not cache_path.exists():
                raise FileNotFoundError(f"Configured regulatory cache does not exist: {cache_path}")
            raw = cache_path.read_bytes()
            version = spec.get("cache_version") or cache_path.stat().st_mtime_ns.__str__()
            fetched_at = datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc).isoformat()
            text = self._xml_text(raw.decode("utf-8")) if cache_path.suffix.lower() == ".xml" else raw.decode("utf-8")
            return [
                (root_url, text, version, fetched_at, hashlib.sha256(text.encode("utf-8")).hexdigest())
            ]
        response = client.get(root_url)
        response.raise_for_status()
        version = (
            response.headers.get("last-modified")
            or response.headers.get("etag")
            or datetime.now(timezone.utc).date().isoformat()
        )
        content_type = response.headers.get("content-type", "").lower()
        fetched_at = datetime.now(timezone.utc).isoformat()
        if "pdf" in content_type or root_url.lower().endswith(".pdf"):
            text = self._pdf_text(response.content)
            return [
                (root_url, text, version, fetched_at, hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest())
            ]
        html = response.text
        root_text = self._xml_text(html) if "xml" in content_type else self._html_text(html)
        documents = [
            (
                root_url,
                root_text,
                version,
                fetched_at,
                hashlib.sha256(root_text.encode("utf-8", errors="ignore")).hexdigest(),
            )
        ]
        if "xml" in content_type:
            return documents
        for link in self._relevant_links(root_url, html, spec["source"])[
            : max(self.settings.regulatory_max_pages_per_source - 1, 0)
        ]:
            try:
                child = client.get(link)
                child.raise_for_status()
                child_version = child.headers.get("last-modified") or child.headers.get("etag") or version
                child_type = child.headers.get("content-type", "").lower()
                text = (
                    self._pdf_text(child.content)
                    if "pdf" in child_type or link.lower().endswith(".pdf")
                    else self._html_text(child.text)
                )
                if text.strip():
                    child_fetched_at = datetime.now(timezone.utc).isoformat()
                    documents.append(
                        (
                            link,
                            text,
                            child_version,
                            child_fetched_at,
                            hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
                        )
                    )
            except Exception as exc:
                logger.debug("Skipping regulatory child page %s: %s", link, exc)
        return documents

    def _relevant_links(self, root_url: str, html: str, source_name: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        root = urlparse(root_url)
        output, seen = [], {root_url}
        for anchor in soup.find_all("a", href=True):
            candidate = urljoin(root_url, anchor["href"])
            parsed = urlparse(candidate)
            if parsed.netloc != root.netloc or candidate in seen:
                continue
            lower = candidate.lower()
            if any(lower.endswith(ext) for ext in (".jpg", ".png", ".css", ".js", ".zip")):
                continue
            relevant = (
                ("FCA" in source_name.upper() and "/handbook/conc/" in lower)
                or ("CFPB" in source_name.upper() and "/rules-policy/regulations/" in lower)
                or (("EUR" in source_name.upper() or "EU " in source_name.upper()) and "32023l2225" in lower)
            )
            if relevant:
                seen.add(candidate)
                output.append(candidate)
        return output

    def _html_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "form"]):
            tag.decompose()
        lines, heading = [], ""
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
            text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if not text:
                continue
            if element.name.startswith("h"):
                heading = text
                lines.append(f"\n## {text}\n")
            else:
                lines.append(f"[{heading}] {text}" if heading else text)
        return "\n".join(lines)[: self.settings.regulatory_max_chars_per_source]

    def _xml_text(self, xml: str) -> str:
        """Extract headings and operative paragraphs from official eCFR XML."""
        soup = BeautifulSoup(xml, "xml")
        lines, heading = [], ""
        paragraph_tags = {"P", "FP", "FP-1", "FP-2", "FP-DASH"}
        for element in soup.find_all(["HEAD", "HD1", "HD2", "HD3", *sorted(paragraph_tags)]):
            text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if not text:
                continue
            if element.name.startswith("H"):
                heading = text
                lines.append(f"\n## {text}\n")
            else:
                lines.append(f"[{heading}] {text}" if heading else text)
        return "\n".join(lines)[: self.settings.regulatory_max_chars_per_source]

    @staticmethod
    def _pdf_text(content: bytes) -> str:
        import fitz

        with fitz.open(stream=content, filetype="pdf") as document:
            return "\n".join(page.get_text("text") for page in document)

    def _encoding_for_model(self):
        if self._encoding is None:
            try:
                import tiktoken

                try:
                    self._encoding = tiktoken.encoding_for_model(self.settings.embedding_model)
                except KeyError:
                    self._encoding = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._encoding = _WhitespaceEncoding()
        return self._encoding

    def _chunk_document(
        self,
        spec: dict[str, str],
        url: str,
        text: str,
        version: str,
        *,
        fetched_at: str,
        content_hash: str,
    ):
        encoding = self._encoding_for_model()
        tokens = encoding.encode(text)
        size, overlap = self.settings.chunk_size_tokens, self.settings.chunk_overlap_tokens
        step = size - overlap
        source_hash = hashlib.sha256(url.encode()).hexdigest()[:10]
        for index, start in enumerate(range(0, len(tokens), step)):
            token_slice = tokens[start : start + size]
            if len(token_slice) < self.settings.regulatory_min_chunk_tokens:
                continue
            chunk_text = encoding.decode(token_slice).strip()
            section_match = re.search(r"\[([^\]]{2,160})\]", chunk_text)
            yield EvidenceChunk(
                chunk_id=f"{source_hash}-{index:05d}",
                source=spec["source"],
                jurisdiction=spec["jurisdiction"],
                source_url=url,
                section=section_match.group(1) if section_match else None,
                source_anchor=(f"{url}#page={index + 1}" if url.lower().endswith(".pdf") else url),
                document_version=version,
                fetched_at=fetched_at,
                content_sha256=content_hash,
                text=chunk_text,
                token_start=start,
                token_end=start + len(token_slice),
            )
            if start + size >= len(tokens):
                break


class _WhitespaceEncoding:
    """Dependency-safe fallback; production requirements include tiktoken."""

    @staticmethod
    def encode(text: str):
        return text.split()

    @staticmethod
    def decode(tokens):
        return " ".join(tokens)
