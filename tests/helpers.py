"""Test utilities: build tiny valid text PDFs without any PDF-writing dependency."""
from __future__ import annotations


def make_pdf(pages: list[str]) -> bytes:
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        lines, y, ops = text.split("\n"), 760, []
        for ln in lines:
            safe = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"BT /F1 10 Tf 40 {y} Td ({safe}) Tj ET")
            y -= 13
        stream = "\n".join(ops).encode("latin-1", "replace")
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, start=1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


# Small multi-theme corpus: energy, medicine, finance (+ cross-document facts)
CORPUS = {
    "solar.txt": (
        "Solar power capacity in India reached 70 GW in 2023. The Ministry of New and Renewable Energy "
        "leads the national solar mission. Photovoltaic panels convert sunlight into electricity, and "
        "falling panel prices drove rapid adoption. Rajasthan hosts the Bhadla Solar Park, one of the "
        "largest solar parks in the world. Grid storage with lithium batteries stabilises solar output.\n\n"
        "Wind turbines complement solar generation because wind peaks at night. Offshore wind projects "
        "in Gujarat are planned for the next decade. Renewable energy auctions reduced tariffs sharply."),
    "wind.txt": (
        "Offshore wind farms generate electricity from strong coastal winds. Turbine blades exceed 100 "
        "metres in length. Gujarat and Tamil Nadu have the best wind resources in India. Wind energy "
        "tariffs fell after competitive auctions were introduced. Transmission lines must be expanded "
        "to carry renewable electricity from windy regions to cities."),
    "medicine.txt": (
        "Metformin is the first-line drug for type 2 diabetes. It lowers blood glucose by reducing "
        "hepatic glucose production. Clinical trials showed metformin reduces cardiovascular risk. "
        "Insulin therapy is required when oral drugs fail. Dr. Anita Rao published the landmark "
        "diabetes trial at AIIMS Delhi in 2019. Patients should monitor HbA1c every three months.\n\n"
        "Hypertension is treated with ACE inhibitors and beta blockers. Regular exercise lowers blood "
        "pressure. Clinical guidelines recommend reducing salt intake for hypertension patients."),
    "vaccines.txt": (
        "Vaccines train the immune system to recognise pathogens. Clinical trials for the measles "
        "vaccine enrolled 12000 children. Antibodies produced after vaccination protect against "
        "infection. Immunisation programmes reduced child mortality across India. Cold chain logistics "
        "keep vaccines effective during transport to rural clinics."),
    "finance.txt": (
        "The Reserve Bank of India sets the repo rate to control inflation. Banks lend to borrowers at "
        "interest rates linked to the repo rate. Inflation in India averaged 5 percent in 2023. "
        "Monetary policy committee meetings occur every two months. Mutual funds invest household "
        "savings in equity and bond markets. Stock market volatility rose after interest rate hikes."),
    "markets.txt": (
        "Equity markets reacted to the central bank rate decision. The Sensex index gained 2 percent "
        "after inflation data surprised analysts. Bond yields rose as investors expected higher interest "
        "rates. Foreign portfolio investors bought Indian equities worth 4 billion USD. Mutual fund "
        "inflows remained strong through systematic investment plans."),
}
