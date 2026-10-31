"""services/core/app/ai/deterministic.py

Deterministic e-invoice extraction for EN 16931 (UBL 2.1 & UN/CEFACT CII).
Complies strictly with contracts/extraction.schema.json via app.ai.schemas.ExtractionResult.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from defusedxml import ElementTree as ET

from app.ai.schemas import ExtractionResult

# XML Namespaces
NS = {
    "ubl_inv": "urn:oasis:names:specification:ubl:schema:xsd:Invoice-2",
    "ubl_cn": "urn:oasis:names:specification:ubl:schema:xsd:CreditNote-2",
    "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
    "rsm": "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100",
    "ram": "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100",
    "udt": "urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100",
    "qdt": "urn:un:unece:uncefact:data:standard:QualifiedDataType:100",
}


def _wrap(val: str | None) -> dict[str, Any]:
    if val is None or str(val).strip() == "":
        return {"value": None, "confidence": 0.0, "bbox": None}
    return {"value": str(val).strip(), "confidence": 1.0, "bbox": None}


def _clean_decimal(val: str | None) -> str | None:
    if not val:
        return None
    cleaned = re.sub(r"[^\d.-]", "", str(val).strip())
    try:
        d = Decimal(cleaned)
        return f"{d:.2f}"
    except (InvalidOperation, ValueError):
        return None


def _format_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    s = date_str.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    if re.match(r"^\d{8}$", s):
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


def parse_ubl(root: ET.Element) -> dict[str, Any]:
    is_credit_note = "CreditNote" in root.tag
    doc_type = "credit_memo" if is_credit_note else "invoice"

    def find_txt(path: str) -> str | None:
        node = root.find(path, NS)
        return node.text if node is not None and node.text else None

    doc_currency = find_txt("cbc:DocumentCurrencyCode")

    vendor = find_txt("cac:AccountingSupplierParty/cac:Party/cac:PartyName/cbc:Name")
    if not vendor:
        vendor = find_txt("cac:AccountingSupplierParty/cac:Party/cac:PartyLegalEntity/cbc:RegistrationName")

    inv_num = find_txt("cbc:ID")
    inv_date = _format_date(find_txt("cbc:IssueDate"))
    due_date = _format_date(find_txt("cbc:DueDate"))
    terms = find_txt("cac:PaymentTerms/cbc:Note")
    po_ref = find_txt("cac:OrderReference/cbc:ID")

    subtotal = _clean_decimal(find_txt("cac:LegalMonetaryTotal/cbc:TaxExclusiveAmount"))
    tax = _clean_decimal(find_txt("cac:TaxTotal/cbc:TaxAmount"))
    total = _clean_decimal(find_txt("cac:LegalMonetaryTotal/cbc:PayableAmount"))
    freight = _clean_decimal(find_txt("cac:LegalMonetaryTotal/cbc:ChargeTotalAmount"))

    remit_name = find_txt("cac:PayeeParty/cac:PartyName/cbc:Name") or vendor
    remit_addr = find_txt("cac:PayeeParty/cac:PostalAddress/cbc:StreetName")
    bank_routing = find_txt("cac:PaymentMeans/cac:PayeeFinancialAccount/cac:FinancialInstitutionBranch/cbc:ID")
    bank_account = find_txt("cac:PaymentMeans/cac:PayeeFinancialAccount/cbc:ID")

    lines = []
    line_nodes = root.findall("cac:InvoiceLine", NS) if not is_credit_note else root.findall("cac:CreditNoteLine", NS)

    for line in line_nodes:
        line_ext = line.find("cbc:LineExtensionAmount", NS)
        if line_ext is not None:
            curr_attr = line_ext.attrib.get("currencyID")
            if curr_attr and doc_currency and curr_attr != doc_currency:
                raise ValueError(f"EN 16931 BR-CL-04: Line currency {curr_attr} != doc currency {doc_currency}")

        desc = None
        item_node = line.find("cac:Item", NS)
        if item_node is not None:
            name_el = item_node.find("cbc:Name", NS)
            desc_el = item_node.find("cbc:Description", NS)
            desc = name_el.text if name_el is not None and name_el.text else (desc_el.text if desc_el is not None else None)

        qty_el = line.find("cbc:InvoicedQuantity", NS)
        if qty_el is None:
            qty_el = line.find("cbc:CreditedQuantity", NS)
        qty = _clean_decimal(qty_el.text) if qty_el is not None and qty_el.text else "1.00"
        uom = qty_el.attrib.get("unitCode", "EA") if qty_el is not None else "EA"

        unit_price = _clean_decimal(line.findtext("cac:Price/cbc:PriceAmount", namespaces=NS))
        line_amt = _clean_decimal(line_ext.text if line_ext is not None and line_ext.text else None)

        lines.append({
            "description": _wrap(desc),
            "qty": _wrap(qty),
            "uom": _wrap(uom),
            "unit_price": _wrap(unit_price),
            "amount": _wrap(line_amt)
        })

    return {
        "vendor_name": _wrap(vendor),
        "invoice_number": _wrap(inv_num),
        "invoice_date": _wrap(inv_date),
        "due_date": _wrap(due_date),
        "terms": _wrap(terms),
        "po_reference": _wrap(po_ref),
        "subtotal": _wrap(subtotal),
        "tax": _wrap(tax),
        "freight": _wrap(freight),
        "total": _wrap(total),
        "currency": _wrap(doc_currency),
        "remit_to_name": _wrap(remit_name),
        "remit_to_address": _wrap(remit_addr),
        "bank_routing": _wrap(re.sub(r"\D", "", bank_routing) if bank_routing else None),
        "bank_account": _wrap(re.sub(r"\D", "", bank_account) if bank_account else None),
        "lines": lines,
        "document_type": doc_type,
        "multiple_invoices_detected": False
    }


def parse_cii(root: ET.Element) -> dict[str, Any]:
    type_code = root.findtext("rsm:ExchangedDocument/ram:TypeCode", namespaces=NS)
    doc_type = "credit_memo" if type_code in ["381", "261"] else "invoice"

    trade_tx = root.find("rsm:SupplyChainTradeTransaction", NS)
    if trade_tx is None:
        raise ValueError("Invalid CII: Missing SupplyChainTradeTransaction")

    agr = trade_tx.find("ram:ApplicableHeaderTradeAgreement", NS)
    settle = trade_tx.find("ram:ApplicableHeaderTradeSettlement", NS)
    monetary = settle.find("ram:SpecifiedTradeSettlementHeaderMonetarySummation", NS) if settle is not None else None

    doc_currency = settle.findtext("ram:InvoiceCurrencyCode", namespaces=NS) if settle is not None else None
    vendor = agr.findtext("ram:SellerTradeParty/ram:Name", namespaces=NS) if agr is not None else None

    inv_num = root.findtext("rsm:ExchangedDocument/ram:ID", namespaces=NS)
    inv_date = _format_date(root.findtext("rsm:ExchangedDocument/ram:IssueDateTime/udt:DateTimeString", namespaces=NS))
    due_date = _format_date(settle.findtext("ram:SpecifiedTradePaymentTerms/ram:DueDateDateTime/udt:DateTimeString", namespaces=NS)) if settle is not None else None
    terms = settle.findtext("ram:SpecifiedTradePaymentTerms/ram:Description", namespaces=NS) if settle is not None else None
    po_ref = agr.findtext("ram:BuyerOrderReferencedDocument/ram:IssuerAssignedID", namespaces=NS) if agr is not None else None

    subtotal = _clean_decimal(monetary.findtext("ram:TaxBasisTotalAmount", namespaces=NS)) if monetary is not None else None
    tax = _clean_decimal(monetary.findtext("ram:TaxTotalAmount", namespaces=NS)) if monetary is not None else None
    total = _clean_decimal(monetary.findtext("ram:GrandTotalAmount", namespaces=NS)) if monetary is not None else None
    freight = _clean_decimal(monetary.findtext("ram:ChargeTotalAmount", namespaces=NS)) if monetary is not None else None

    bank_account = settle.findtext("ram:SpecifiedTradeSettlementPaymentMeans/ram:PayeePartyCreditorFinancialAccount/ram:IBANID", namespaces=NS) if settle is not None else None
    bank_routing = settle.findtext("ram:SpecifiedTradeSettlementPaymentMeans/ram:PayeeSpecifiedCreditorFinancialInstitution/ram:BICID", namespaces=NS) if settle is not None else None

    lines = []
    line_nodes = trade_tx.findall("ram:IncludedSupplyChainTradeLineItem", NS)

    for line in line_nodes:
        desc = line.findtext("ram:SpecifiedTradeProduct/ram:Name", namespaces=NS)
        qty_el = line.find("ram:SpecifiedLineTradeDelivery/ram:BilledQuantity", NS)
        qty = _clean_decimal(qty_el.text) if qty_el is not None and qty_el.text else "1.00"
        uom = qty_el.attrib.get("unitCode", "EA") if qty_el is not None else "EA"

        unit_price = _clean_decimal(line.findtext("ram:SpecifiedLineTradeAgreement/ram:NetPriceProductTradePrice/ram:ChargeAmount", namespaces=NS))
        if not unit_price:
            unit_price = _clean_decimal(line.findtext("ram:SpecifiedLineTradeAgreement/ram:GrossPriceProductTradePrice/ram:ChargeAmount", namespaces=NS))

        line_amt = _clean_decimal(line.findtext("ram:SpecifiedLineTradeSettlement/ram:SpecifiedTradeSettlementLineMonetarySummation/ram:LineTotalAmount", namespaces=NS))

        lines.append({
            "description": _wrap(desc),
            "qty": _wrap(qty),
            "uom": _wrap(uom),
            "unit_price": _wrap(unit_price),
            "amount": _wrap(line_amt)
        })

    return {
        "vendor_name": _wrap(vendor),
        "invoice_number": _wrap(inv_num),
        "invoice_date": _wrap(inv_date),
        "due_date": _wrap(due_date),
        "terms": _wrap(terms),
        "po_reference": _wrap(po_ref),
        "subtotal": _wrap(subtotal),
        "tax": _wrap(tax),
        "freight": _wrap(freight),
        "total": _wrap(total),
        "currency": _wrap(doc_currency),
        "remit_to_name": _wrap(vendor),
        "remit_to_address": _wrap(None),
        "bank_routing": _wrap(bank_routing),
        "bank_account": _wrap(bank_account),
        "lines": lines,
        "document_type": doc_type,
        "multiple_invoices_detected": False
    }


def try_extract_embedded_xml(pdf_bytes: bytes) -> bytes | None:
    """Extract embedded XML attachment from a PDF/A-3 file."""
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        n_attachments = doc.get_n_attachments()
        for i in range(n_attachments):
            attachment = doc.get_attachment(i)
            name = attachment.get_name().lower()
            if name.endswith(".xml") or "factur-x" in name or "zugferd" in name:
                return attachment.get_data()
    except Exception:
        pass

    # Direct byte pattern scan fallback
    xml_match = re.search(rb"(<\?xml[\s\S]*?</(?:rsm:CrossIndustryInvoice|Invoice)>)", pdf_bytes, re.IGNORECASE)
    if xml_match:
        return xml_match.group(1)

    return None


def extract_deterministic_invoice(raw_bytes: bytes) -> ExtractionResult | None:
    """Attempts deterministic intake from XML or hybrid PDF attachments.
    Returns ExtractionResult if parsed cleanly, or None to fall back to vision.
    """
    xml_data: bytes | None = None

    trimmed = raw_bytes.lstrip()
    if trimmed.startswith(b"<?xml") or trimmed.startswith(b"<"):
        xml_data = raw_bytes
    elif raw_bytes.startswith(b"%PDF"):
        xml_data = try_extract_embedded_xml(raw_bytes)

    if not xml_data:
        return None

    try:
        root = ET.fromstring(xml_data)
        tag = root.tag.lower()

        if "crossindustryinvoice" in tag:
            parsed = parse_cii(root)
        elif "invoice" in tag or "creditnote" in tag:
            parsed = parse_ubl(root)
        else:
            return None

        return ExtractionResult.model_validate(parsed)
    except Exception:
        return None