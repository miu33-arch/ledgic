"""services/core/tests/ai/test_deterministic.py

Unit tests for EN 16931 deterministic e-invoice extraction (UBL 2.1 & UN/CEFACT CII).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.ai.deterministic import extract_deterministic_invoice
from app.ai.extraction import extract_document
from app.ai.schemas import ExtractionResult

# Minimal EN 16931 compliant UBL 2.1 invoice snippet
UBL_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<Invoice xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"
         xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"
         xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2">
    <cbc:CustomizationID>urn:cen.eu:en16931:2017</cbc:CustomizationID>
    <cbc:ID>INV-2026-001</cbc:ID>
    <cbc:IssueDate>2026-10-31</cbc:IssueDate>
    <cbc:DueDate>2026-11-30</cbc:DueDate>
    <cbc:DocumentCurrencyCode>EUR</cbc:DocumentCurrencyCode>
    <cac:OrderReference>
        <cbc:ID>PO-99120</cbc:ID>
    </cac:OrderReference>
    <cac:AccountingSupplierParty>
        <cac:Party>
            <cac:PartyName>
                <cbc:Name>Miu Tech Solutions</cbc:Name>
            </cac:PartyName>
        </cac:Party>
    </cac:AccountingSupplierParty>
    <cac:PaymentMeans>
        <cac:PayeeFinancialAccount>
            <cbc:ID>DE89370400440532013000</cbc:ID>
            <cac:FinancialInstitutionBranch>
                <cbc:ID>DBEUMM2X</cbc:ID>
            </cac:FinancialInstitutionBranch>
        </cac:PayeeFinancialAccount>
    </cac:PaymentMeans>
    <cac:TaxTotal>
        <cbc:TaxAmount currencyID="EUR">19.00</cbc:TaxAmount>
    </cac:TaxTotal>
    <cac:LegalMonetaryTotal>
        <cbc:LineExtensionAmount currencyID="EUR">100.00</cbc:LineExtensionAmount>
        <cbc:TaxExclusiveAmount currencyID="EUR">100.00</cbc:TaxExclusiveAmount>
        <cbc:TaxInclusiveAmount currencyID="EUR">119.00</cbc:TaxInclusiveAmount>
        <cbc:PayableAmount currencyID="EUR">119.00</cbc:PayableAmount>
    </cac:LegalMonetaryTotal>
    <cac:InvoiceLine>
        <cbc:ID>1</cbc:ID>
        <cbc:InvoicedQuantity unitCode="EA">1</cbc:InvoicedQuantity>
        <cbc:LineExtensionAmount currencyID="EUR">100.00</cbc:LineExtensionAmount>
        <cac:Item>
            <cbc:Name>Software Integration License</cbc:Name>
        </cac:Item>
        <cac:Price>
            <cbc:PriceAmount currencyID="EUR">100.00</cbc:PriceAmount>
        </cac:Price>
    </cac:InvoiceLine>
</Invoice>
"""

# Minimal EN 16931 compliant UN/CEFACT CII (Cross Industry Invoice) snippet
CII_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rsm:CrossIndustryInvoice xmlns:rsm="urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100"
                          xmlns:ram="urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100"
                          xmlns:udt="urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100">
    <rsm:ExchangedDocument>
        <ram:ID>CII-88412</ram:ID>
        <ram:TypeCode>380</ram:TypeCode>
        <ram:IssueDateTime>
            <udt:DateTimeString format="102">20261031</udt:DateTimeString>
        </ram:IssueDateTime>
    </rsm:ExchangedDocument>
    <rsm:SupplyChainTradeTransaction>
        <ram:ApplicableHeaderTradeAgreement>
            <ram:SellerTradeParty>
                <ram:Name>Euro Cloud Services</ram:Name>
            </ram:SellerTradeParty>
            <ram:BuyerOrderReferencedDocument>
                <ram:IssuerAssignedID>PO-7721</ram:IssuerAssignedID>
            </ram:BuyerOrderReferencedDocument>
        </ram:ApplicableHeaderTradeAgreement>
        <ram:ApplicableHeaderTradeSettlement>
            <ram:InvoiceCurrencyCode>EUR</ram:InvoiceCurrencyCode>
            <ram:SpecifiedTradePaymentTerms>
                <ram:DueDateDateTime>
                    <udt:DateTimeString format="102">20261115</udt:DateTimeString>
                </ram:DueDateDateTime>
            </ram:SpecifiedTradePaymentTerms>
            <ram:SpecifiedTradeSettlementHeaderMonetarySummation>
                <ram:LineTotalAmount>250.00</ram:LineTotalAmount>
                <ram:TaxBasisTotalAmount>250.00</ram:TaxBasisTotalAmount>
                <ram:TaxTotalAmount currencyID="EUR">47.50</ram:TaxTotalAmount>
                <ram:GrandTotalAmount>297.50</ram:GrandTotalAmount>
            </ram:SpecifiedTradeSettlementHeaderMonetarySummation>
        </ram:ApplicableHeaderTradeSettlement>
        <ram:IncludedSupplyChainTradeLineItem>
            <ram:SpecifiedTradeProduct>
                <ram:Name>Dedicated Hosting Core</ram:Name>
            </ram:SpecifiedTradeProduct>
            <ram:SpecifiedLineTradeAgreement>
                <ram:NetPriceProductTradePrice>
                    <ram:ChargeAmount>250.00</ram:ChargeAmount>
                </ram:NetPriceProductTradePrice>
            </ram:SpecifiedLineTradeAgreement>
            <ram:SpecifiedLineTradeDelivery>
                <ram:BilledQuantity unitCode="EA">1.00</ram:BilledQuantity>
            </ram:SpecifiedLineTradeDelivery>
            <ram:SpecifiedLineTradeSettlement>
                <ram:SpecifiedTradeSettlementLineMonetarySummation>
                    <ram:LineTotalAmount>250.00</ram:LineTotalAmount>
                </ram:SpecifiedTradeSettlementLineMonetarySummation>
            </ram:SpecifiedLineTradeSettlement>
        </ram:IncludedSupplyChainTradeLineItem>
    </rsm:SupplyChainTradeTransaction>
</rsm:CrossIndustryInvoice>
"""

# Invalid line currency to test BR-CL-04 rule
UBL_INVALID_CURRENCY = UBL_SAMPLE.replace(b'LineExtensionAmount currencyID="EUR"', b'LineExtensionAmount currencyID="USD"')


def test_ubl_deterministic_extraction():
    res = extract_deterministic_invoice(UBL_SAMPLE)
    assert res is not None
    assert isinstance(res, ExtractionResult)
    assert res.invoice_number.value == "INV-2026-001"
    assert res.invoice_number.confidence == 1.0
    assert res.vendor_name.value == "Miu Tech Solutions"
    assert res.total.value == "119.00"
    assert res.tax.value == "19.00"
    assert res.subtotal.value == "100.00"
    assert res.currency.value == "EUR"
    assert res.po_reference.value == "PO-99120"
    assert len(res.lines) == 1
    assert res.lines[0].description.value == "Software Integration License"
    assert res.lines[0].amount.value == "100.00"


def test_cii_deterministic_extraction():
    res = extract_deterministic_invoice(CII_SAMPLE)
    assert res is not None
    assert isinstance(res, ExtractionResult)
    assert res.invoice_number.value == "CII-88412"
    assert res.invoice_date.value == "2026-10-31"
    assert res.due_date.value == "2026-11-15"
    assert res.vendor_name.value == "Euro Cloud Services"
    assert res.total.value == "297.50"
    assert res.subtotal.value == "250.00"
    assert res.tax.value == "47.50"
    assert res.document_type == "invoice"


def test_en16931_br_cl_04_currency_mismatch_fallback():
    # If line currency != doc currency, it must fail deterministic parsing gracefully
    res = extract_deterministic_invoice(UBL_INVALID_CURRENCY)
    assert res is None


def test_non_xml_returns_none():
    res = extract_deterministic_invoice(b"random-non-xml-binary-data")
    assert res is None


@pytest.mark.asyncio
async def test_extract_document_uses_deterministic_mode():
    outcome = await extract_document(
        pages=[UBL_SAMPLE],
        mime="application/xml",
    )
    assert outcome.mode == "deterministic"
    assert outcome.cost_usd == Decimal("0.00")
    assert outcome.pages_sent == 0
    assert outcome.result.invoice_number.value == "INV-2026-001"