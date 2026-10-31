"""Extraction agent (06 §5, DECISIONS D2): page images → strict `ExtractionResult`.

Modes of the returned outcome (stored in `invoices.extraction_mode`):
- `deterministic` — zero-cost intake from embedded or standalone EN 16931 XML (UBL/CII).
- `live`          — gpt-4.1-mini vision via the Agents SDK, metered by the gateway.
- `simulated`     — the caller passed `fixture` (e.g. the corpus' `simulated_extraction` for replay docs).
- `recorded`      — a recorded hero output from `tools/corpus/fixtures/heroes/<invoice>.json`.

Confidences returned here are the model's raw confidences; calibration (`app.domain.gate.
calibrate_fallback`) and the gate are applied by the orchestrator.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any

from agents import Agent

from .deterministic import extract_deterministic_invoice
from .gateway import BudgetExceeded, LlmGateway, LlmUnavailable, get_gateway
from .paths import PROMPTS_DIR, heroes_dir
from .schemas import ExtractionResult, normalize_extraction

log = logging.getLogger("ledgic.ai.extraction")

PROMPT_VERSION = "p-1"
MAX_PAGES = 3
MAX_SIDE = 1600
PDF_RENDER_SCALE = 150 / 72  # 150 dpi; downscaled to MAX_SIDE afterwards
# Token estimate for the budget reservation (gpt-4.1-mini image patches at ~1600px ≈ 2.5k tokens/page).
EST_TOKENS_PER_PAGE = 2600
EST_PROMPT_TOKENS = 1500
MAX_OUTPUT_TOKENS = 8000


class ExtractionUnavailable(Exception):
    """No live call possible (fixture mode / no key / budget) and no fixture to fall back to."""


@dataclass(frozen=True)
class ExtractionOutcome:
    result: ExtractionResult
    mode: str  # deterministic | live | recorded | simulated
    model: str | None
    prompt_version: str
    cost_usd: Decimal
    pages_sent: int = 0
    fallback_reason: str | None = None  # why a live call was not made (budget, fixture mode, ...)


@lru_cache
def extraction_prompt(version: str = PROMPT_VERSION) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")


def build_agent() -> Agent[Any]:
    return Agent(
        name="Ledgic extraction",
        instructions=extraction_prompt(),
        output_type=ExtractionResult,
    )


# --------------------------------------------------------------------------------------- page images
def _downscale_png(img: Any) -> bytes:
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    longest = max(img.size)
    if longest > MAX_SIDE:
        ratio = MAX_SIDE / longest
        img = img.resize((max(1, round(img.width * ratio)), max(1, round(img.height * ratio))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _render_pdf(data: bytes, max_pages: int) -> list[bytes]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        out = []
        for i in range(min(len(pdf), max_pages)):
            page = pdf[i]
            img = page.render(scale=PDF_RENDER_SCALE).to_pil()
            out.append(_downscale_png(img))
            page.close()
        return out
    finally:
        pdf.close()


def prepare_pages(pages: list[bytes], mime: str) -> list[str]:
    """Bytes → base64 PNG data URLs: ≤ 3 pages, longest side ≤ 1600px.
    `pages` is either one PDF (mime application/pdf) or one image per page."""
    from PIL import Image

    images: list[bytes] = []
    for blob in pages:
        if len(images) >= MAX_PAGES:
            break
        if mime == "application/pdf" or blob[:5] == b"%PDF-":
            images += _render_pdf(blob, MAX_PAGES - len(images))
        else:
            with Image.open(io.BytesIO(blob)) as img:
                img.load()
                images.append(_downscale_png(img))
    return ["data:image/png;base64," + base64.b64encode(b).decode("ascii") for b in images[:MAX_PAGES]]


def build_input(data_urls: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": f"Extract the accounts-payable fields from this document ({len(data_urls)} page(s)).",
        }
    ]
    content += [{"type": "input_image", "image_url": u, "detail": "high"} for u in data_urls]
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------------------- fixtures
def load_hero_fixture(invoice_number: str) -> dict[str, Any] | None:
    """Recorded extraction for a hero (`PS-88214`, `PR-2214`, `BD-104882`, `NV-0001`, `PR-2201`,
    `PR-2201-photo`)."""
    try:
        path = heroes_dir() / f"{invoice_number}.json"
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("extraction", data)


# --------------------------------------------------------------------------------------- entrypoint
async def extract_document(
    pages: list[bytes],
    *,
    mime: str,
    fixture: dict[str, Any] | None = None,
    api_key: str | None = None,
    org_id: str | None = None,
    run_id: str | None = None,
    hero: str | None = None,
    gateway: LlmGateway | None = None,
) -> ExtractionOutcome:
    """Extract one document.

    Deterministic when valid EN 16931 XML (standalone or embedded in hybrid PDF) is detected;
    live when `LLM_MODE=live`, a key is available (owner or BYOK `api_key`) and the budget allows;
    otherwise falls back to `fixture` (→ mode `simulated`) or the recorded hero file (`hero`, → mode
    `recorded`). Raises `ExtractionUnavailable` when nothing is available, `BudgetExceeded` never
    escapes when a fallback exists. LLM/API errors propagate (the orchestrator retries)."""
    # 0. Deterministic intake check (zero LLM cost, instant 1.0 confidence)
    if pages:
        deterministic_res = extract_deterministic_invoice(pages[0])
        if deterministic_res is not None:
            return ExtractionOutcome(
                result=deterministic_res,
                mode="deterministic",
                model=None,
                prompt_version="deterministic",
                cost_usd=Decimal(0),
                pages_sent=0,
                fallback_reason=None,
            )

    gw = gateway or get_gateway()
    reason: str | None = None
    if gw.can_call(api_key):
        try:
            return await _extract_live(pages, mime=mime, api_key=api_key, org_id=org_id, run_id=run_id, gw=gw)
        except BudgetExceeded as exc:
            reason = f"budget: {exc}"
        except LlmUnavailable as exc:
            reason = str(exc)
    else:
        reason = "fixture mode" if gw.mode != "live" else "no API key"

    if fixture is not None:
        return ExtractionOutcome(
            normalize_extraction(fixture), "simulated", None, PROMPT_VERSION, Decimal(0), 0, reason
        )
    recorded = load_hero_fixture(hero) if hero else None
    if recorded is not None:
        return ExtractionOutcome(
            normalize_extraction(recorded), "recorded", None, PROMPT_VERSION, Decimal(0), 0, reason
        )
    raise ExtractionUnavailable(reason or "no extraction available")


async def _extract_live(
    pages: list[bytes],
    *,
    mime: str,
    api_key: str | None,
    org_id: str | None,
    run_id: str | None,
    gw: LlmGateway,
) -> ExtractionOutcome:
    from .runtime import run_agent

    urls = prepare_pages(pages, mime)
    if not urls:
        raise ValueError("document has no renderable pages")
    est_in = EST_PROMPT_TOKENS + EST_TOKENS_PER_PAGE * len(urls)
    async with gw.metered(
        purpose="extraction",
        org_id=org_id,
        run_id=run_id,
        est_input_tokens=est_in,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        api_key=api_key,
    ) as meter:
        result = await run_agent(
            build_agent(),
            build_input(urls),
            meter,
            max_tokens=MAX_OUTPUT_TOKENS,
            max_turns=2,
            workflow="ledgic.extraction",
        )
    out = result.final_output
    if not isinstance(out, ExtractionResult):
        out = ExtractionResult.model_validate(out)
    return ExtractionOutcome(out, "live", meter.model, PROMPT_VERSION, meter.cost_usd, len(urls))