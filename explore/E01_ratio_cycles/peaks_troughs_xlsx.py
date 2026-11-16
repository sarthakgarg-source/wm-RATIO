"""
Build an Excel workbook that shows, with live formulas, how E01 finds the peaks and troughs of a ratio (the zig-zag rule,
point-in-time). EXPLORATORY teaching aid; the numbers come from the approved data only.

    python explore/E01_ratio_cycles/peaks_troughs_xlsx.py   → explore/E01_ratio_cycles/out/E01_peaks_troughs_explained.xlsx

Sheets: "How it works" (plain words), "Toy example" (21 made-up weeks, fixed 10% threshold), "Real data" (N100 ÷ SC100,
every step as a formula), "Turning points" (the list Excel finds, matched row by row against cycles.py's own result).
"""
import os
import sys

import numpy as np
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from analysis.approved_data import load_daily  # noqa: E402
from analysis.ratio_engine import pair_frame  # noqa: E402
import cycles as E  # noqa: E402  (the same point_in_time and zigzag functions the page uses)

PAIR = "N100|SC100"
F_BASE, F_BOLD, F_TITLE = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True), Font(name="Arial", size=14, bold=True)
F_INPUT, F_NOTE = Font(name="Arial", size=10, color="0000FF"), Font(name="Arial", size=9, italic=True, color="52514E")
HEAD_FILL, IN_FILL = PatternFill("solid", fgColor="E1E0D9"), PatternFill("solid", fgColor="FFFF00")
PEAK_FILL, TROUGH_FILL = PatternFill("solid", fgColor="FBD5C5"), PatternFill("solid", fgColor="C9EEDD")
THIN = Border(bottom=Side(style="thin", color="C3C2B7"))
WRAP = Alignment(wrap_text=True, vertical="top")


def header(ws, row, names, widths):
    for j, (n, w) in enumerate(zip(names, widths), start=1):
        c = ws.cell(row=row, column=j, value=n); c.font = F_BOLD; c.fill = HEAD_FILL; c.alignment = Alignment(wrap_text=True, vertical="bottom"); c.border = THIN
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[row].height = 54


def fmt(ws, col, first, last, f):
    for r in range(first, last + 1):
        ws[f"{col}{r}"].number_format = f


def all_font(ws):
    for row in ws.iter_rows():
        for c in row:
            if c.value is not None and c.font.name != "Arial":
                c.font = F_BASE


# ---------------------------------------------------------------------------------------------------- How it works
def how_it_works(wb, n_weeks, first, last):
    ws = wb.active; ws.title = "How it works"
    ws.column_dimensions["A"].width = 4; ws.column_dimensions["B"].width = 120
    ws["B1"] = "How the peaks and troughs of a ratio are found (E01, point-in-time)"; ws["B1"].font = F_TITLE
    lines = [
        ("", "EXPLORATORY teaching aid. Every number in 'Real data' and 'Turning points' is a live formula; the blue cells are the only inputs."),
        ("", ""),
        ("", "THE IDEA IN ONE LINE"),
        ("", "A peak is a high point that the ratio later falls away from by a set amount; a trough is a low point it later rises from by that amount. Small wiggles in between are ignored."),
        ("", ""),
        ("", "THE STEPS (one column each in 'Real data')"),
        ("1", "Ratio = N100 ÷ SC100 at the close of the last trading day of each week (Friday, unless it was a holiday)."),
        ("2", "Weekly change = LN(this week's ratio ÷ last week's ratio). It is the ratio's % change: 0.02 ≈ +2%."),
        ("3", "Drift = the AVERAGE of all EARLIER weekly changes (only the past, never later weeks). Taking it out stops a slow, steady trend from hiding the swings."),
        ("4", "Change minus drift, then a running total of it = the 'path': the ratio with its drift taken out (in log terms). Turning points are found on this path."),
        ("5", "Threshold = the standard deviation of all EARLIER weekly changes × √26 = one typical half-year move for this pair, known at that week. In plain %, a trough needs a RISE of e^h − 1 from the low and a peak a FALL of 1 − e^−h from the high (the same distance in log terms)."),
        ("6", "Warm-up: the first 52 weeks are only used to start the average and the standard deviation; the path starts after them."),
        ("", ""),
        ("", "THE ZIG-ZAG RULE (columns P to AB)"),
        ("a", "At the start: keep the highest and the lowest point of the path. As soon as they are one threshold apart, whichever came first is the first turning point (the low → a TROUGH, the high → a PEAK)."),
        ("b", "While RISING: keep track of the highest point since the last trough. If the path goes higher, that becomes the new high. If it falls one threshold below the high, the high was a PEAK — now we are FALLING, watching for a trough."),
        ("c", "While FALLING: keep track of the lowest point since the last peak. If the path goes lower, that becomes the new low. If it rises one threshold above the low, the low was a TROUGH — now we are RISING, watching for a peak."),
        ("d", "A turning point is only KNOWN later: on the week the reversal reaches the threshold (column X, 'confirmed this week'), several weeks after the actual high or low (column Y). Column AB marks the actual week."),
        ("", ""),
        ("", "WHERE TO LOOK"),
        ("→", "Toy example: 21 made-up weeks with a fixed 10% threshold — follow the rule by hand. Start here."),
        ("→", f"Real data: {n_weeks} weeks of N100 ÷ SC100, {first} → {last}, every step as a formula. Rows where a turning point is confirmed are coloured (orange = peak, green = trough)."),
        ("→", "Turning points: the list Excel finds, with each swing's length and size, matched row by row against the Python script (explore/E01_ratio_cycles/cycles.py). The cell at the top says whether they agree."),
        ("", ""),
        ("", "COLOURS: blue text = inputs (data, settings); black = formulas; yellow = settings you can change (warm-up, half-year length, toy threshold)."),
        ("", "Source of the data: data/index_ohlcv_all.csv through analysis/approved_data.load_daily() (fingerprint-checked), on the pair's common calendar as T01/T02 use it."),
    ]
    for i, (a, b) in enumerate(lines, start=3):
        ws[f"A{i}"] = a; ws[f"B{i}"] = b; ws[f"B{i}"].alignment = WRAP
        ws[f"A{i}"].font = F_BOLD; ws[f"B{i}"].font = F_BOLD if b.isupper() else F_BASE
    return ws


# ---------------------------------------------------------------------------------------------------- Toy example
TOY = [100, 104, 109, 115, 120, 117, 112, 107, 103, 100, 102, 98, 104, 108.5, 113, 110, 118, 121, 115, 108, 104]
TOY_NOTES = ["Start: waiting for the first swing", "", "", "115 is 15% above the low of 100 (week 0): first swing → week 0 was a TROUGH; now rising",
             "New high (120)", "Small dip: 117 is only 2.5% below 120", "6.7% below the high — not 10% yet", "107 is 10.8% below 120 → week 4 was a PEAK (known only now, 3 weeks later); now falling",
             "New low", "New low (100)", "Bounce of 2% — not 10%", "New low (98)", "6.1% above the low — not 10% yet", "108.5 is 10.7% above 98 → week 11 was a TROUGH; now rising",
             "New high", "Dip of 2.7% — ignored (smaller than 10%)", "New high", "New high (121)", "5% below the high — not 10% yet", "108 is 10.7% below 121 → week 17 was a PEAK; now falling", "New low"]


def toy(wb):
    ws = wb.create_sheet("Toy example")
    ws["A1"] = "Toy example: the zig-zag rule on 21 made-up weeks"; ws["A1"].font = F_TITLE
    ws["A2"] = "Change the blue ratios or the yellow threshold and every column recalculates. Same rule as 'Real data', but on the ratio itself with a fixed threshold (no drift, no logs), to keep it simple."; ws["A2"].font = F_NOTE
    ws["A3"] = "Threshold (a move of at least this much confirms a turning point)"; ws["A3"].font = F_BOLD
    ws["H3"] = 0.10; ws["H3"].font = F_INPUT; ws["H3"].fill = IN_FILL; ws["H3"].number_format = "0%"
    ws["A4"] = "Answer for the values as given: troughs at weeks 0 and 11, peaks at weeks 4 and 17; the 2.7% dip in week 15 is ignored."; ws["A4"].font = F_NOTE
    names = ["Week", "Ratio (input)", "Direction after this week", "Highest so far (start only)", "Week of that high", "Lowest so far (start only)", "Week of that low",
             "Extreme since the last turning point", "Week of the extreme", "Move from the extreme", "Turning point confirmed this week", "Week of that turning point",
             "This week was a … (found later)", "What happened (formula)", "Notes for the values as given"]
    header(ws, 6, names, [7, 10, 12, 12, 9, 12, 9, 13, 10, 11, 13, 12, 13, 60, 60])
    r0 = 7; rN = r0 + len(TOY) - 1; T = "$H$3"
    for i, v in enumerate(TOY):
        r = r0 + i; p = r - 1
        ws[f"A{r}"] = i; ws[f"B{r}"] = v; ws[f"B{r}"].font = F_INPUT; ws[f"O{r}"] = TOY_NOTES[i]; ws[f"O{r}"].font = F_NOTE
        if i == 0:
            ws[f"C{r}"] = "waiting"; ws[f"D{r}"] = f"=B{r}"; ws[f"E{r}"] = f"=A{r}"; ws[f"F{r}"] = f"=B{r}"; ws[f"G{r}"] = f"=A{r}"; ws[f"H{r}"] = f"=B{r}"; ws[f"I{r}"] = f"=A{r}"
            ws[f"J{r}"] = ""; ws[f"K{r}"] = ""; ws[f"L{r}"] = ""
        else:
            ws[f"C{r}"] = (f'=IF(C{p}="waiting",IF(D{r}/F{r}-1>={T},IF(E{r}>G{r},"rising","falling"),"waiting"),'
                           f'IF(C{p}="rising",IF(B{r}>H{p},"rising",IF(B{r}/H{p}-1<=-{T},"falling","rising")),'
                           f'IF(B{r}<H{p},"falling",IF(B{r}/H{p}-1>={T},"rising","falling"))))')
            ws[f"D{r}"] = f'=IF(C{p}="waiting",MAX(D{p},B{r}),D{p})'; ws[f"E{r}"] = f'=IF(AND(C{p}="waiting",B{r}>D{p}),A{r},E{p})'
            ws[f"F{r}"] = f'=IF(C{p}="waiting",MIN(F{p},B{r}),F{p})'; ws[f"G{r}"] = f'=IF(AND(C{p}="waiting",B{r}<F{p}),A{r},G{p})'
            ws[f"K{r}"] = f'=IF(C{p}="waiting",IF(C{r}="waiting","",IF(C{r}="rising","TROUGH","PEAK")),IF(C{p}=C{r},"",IF(C{r}="falling","PEAK","TROUGH")))'
            ws[f"L{r}"] = f'=IF(K{r}="","",IF(C{p}="waiting",IF(K{r}="TROUGH",G{r},E{r}),I{p}))'
            ws[f"H{r}"] = (f'=IF(C{p}="waiting",IF(C{r}="waiting",H{p},IF(C{r}="rising",D{r},F{r})),'
                           f'IF(C{p}="rising",IF(OR(B{r}>H{p},K{r}<>""),B{r},H{p}),IF(OR(B{r}<H{p},K{r}<>""),B{r},H{p})))')
            ws[f"I{r}"] = (f'=IF(C{p}="waiting",IF(C{r}="waiting",I{p},IF(C{r}="rising",E{r},G{r})),'
                           f'IF(C{p}="rising",IF(OR(B{r}>H{p},K{r}<>""),A{r},I{p}),IF(OR(B{r}<H{p},K{r}<>""),A{r},I{p})))')
            ws[f"J{r}"] = f'=IF(C{p}="waiting","",B{r}/H{p}-1)'
        ws[f"M{r}"] = f'=IFERROR(INDEX($K${r0}:$K${rN},MATCH(A{r},$L${r0}:$L${rN},0)),"")'
        ws[f"N{r}"] = ("" if i == 0 else
                       f'=IF(K{r}="","",IF(C{r - 1}="waiting","First swing: the ratio moved "&TEXT({T},"0%")&" from its "&IF(K{r}="TROUGH","low","high")&" → week "&L{r}&" was a "&K{r},'
                       f'IF(K{r}="PEAK","Fell "&TEXT(-J{r},"0.0%")&" from the high of week "&L{r}&" → that high was a PEAK","Rose "&TEXT(J{r},"0.0%")&" from the low of week "&L{r}&" → that low was a TROUGH")))')
    fmt(ws, "B", r0, rN, "0.0"); fmt(ws, "D", r0, rN, "0.0"); fmt(ws, "F", r0, rN, "0.0"); fmt(ws, "H", r0, rN, "0.0"); fmt(ws, "J", r0, rN, "+0.0%;-0.0%;0.0%")
    ws.conditional_formatting.add(f"A{r0}:N{rN}", FormulaRule(formula=[f'$K{r0}="PEAK"'], fill=PEAK_FILL))
    ws.conditional_formatting.add(f"A{r0}:N{rN}", FormulaRule(formula=[f'$K{r0}="TROUGH"'], fill=TROUGH_FILL))
    for c in ("M",):
        ws.conditional_formatting.add(f"{c}{r0}:{c}{rN}", FormulaRule(formula=[f'${c}{r0}="PEAK"'], font=Font(name="Arial", size=10, bold=True, color="C2410C")))
        ws.conditional_formatting.add(f"{c}{r0}:{c}{rN}", FormulaRule(formula=[f'${c}{r0}="TROUGH"'], font=Font(name="Arial", size=10, bold=True, color="047857")))
    ch = LineChart(); ch.title = "Toy ratio by week"; ch.y_axis.title = "ratio"; ch.x_axis.title = "week"; ch.height = 7; ch.width = 16
    ch.add_data(Reference(ws, min_col=2, min_row=6, max_row=rN), titles_from_data=True); ch.set_categories(Reference(ws, min_col=1, min_row=r0, max_row=rN))
    ch.roundedCorners = False; ch.legend = None; ch.title.overlay = False; ch.x_axis.title.overlay = False; ch.y_axis.title.overlay = False; ch.x_axis.delete = False; ch.y_axis.delete = False
    ws.add_chart(ch, f"A{rN + 3}")
    ws.freeze_panes = "C7"; all_font(ws)


# ---------------------------------------------------------------------------------------------------- Real data
def real(wb, wk, ca, cb):
    ws = wb.create_sheet("Real data")
    n = len(wk); r0 = 8; rN = r0 + n - 1; WARM, HALF = "$B$3", "$B$4"
    ws["A1"] = f"Real data: {E.nice(PAIR)}, weekly, {wk.index[0].date()} → {wk.index[-1].date()}"; ws["A1"].font = F_TITLE
    ws["A2"] = ("Source: data/index_ohlcv_all.csv via analysis/approved_data.load_daily() (fingerprint-checked); closes on the last trading day of each ISO week on the pair's "
                "common calendar, exactly as T01/T02 and E01 use them. Blue = input; every other number is a formula."); ws["A2"].font = F_NOTE
    ws["A3"] = "Warm-up (weeks)"; ws["B3"] = E.BURN; ws["A4"] = "Weeks in a half-year (threshold = sd × √ of this)"; ws["B4"] = E.H_WEEKS
    for c in ("B3", "B4"):
        ws[c].font = F_INPUT; ws[c].fill = IN_FILL
    ws["A3"].font = F_BOLD; ws["A4"].font = F_BOLD
    ws["D3"] = "Turning points found:"; ws["D3"].font = F_BOLD; ws["F3"] = f'=COUNTIF(X{r0}:X{rN},"PEAK")+COUNTIF(X{r0}:X{rN},"TROUGH")'
    ws["D4"] = "Orange row = a PEAK is confirmed that week; green = a TROUGH. Column AB marks the actual peak / trough week."; ws["D4"].font = F_NOTE
    names = ["Week #", "Date (Friday close)", "N100 close", "SC100 close", "Ratio N100 ÷ SC100", "Weekly change LN(ratio ÷ last week's)", "Earlier weekly changes (count)",
             "Drift = average of earlier changes", "Sd of earlier changes", "Change minus drift", "Path = running total (drift-removed, log)", "Drift-removed ratio",
             "Threshold h = sd × √26 (log)", "Rise needed for a trough = e^h − 1", "Fall needed for a peak = 1 − e^−h", "Direction after this week",
             "Highest point so far (start only)", "Week of that high", "Lowest point so far (start only)", "Week of that low", "Extreme since the last turning point (path)",
             "Week of the extreme", "Move from the extreme", "Turning point confirmed this week", "Week # of that turning point", "Turning point no.",
             "What happened (formula)", "This week was a … (found later)"]
    header(ws, 7, names, [7, 11, 10, 10, 10, 12, 9, 12, 11, 11, 12, 11, 11, 10, 10, 11, 11, 8, 11, 8, 12, 9, 10, 12, 10, 8, 62, 12])
    for i in range(n):
        r = r0 + i; p = r - 1
        ws[f"A{r}"] = i if i == 0 else f"=A{p}+1"
        ws[f"B{r}"] = wk.index[i].to_pydatetime(); ws[f"C{r}"] = float(ca.iloc[i]); ws[f"D{r}"] = float(cb.iloc[i])
        for c in ("B", "C", "D"):
            ws[f"{c}{r}"].font = F_INPUT
        ws[f"E{r}"] = f"=C{r}/D{r}"
        ws[f"F{r}"] = "" if i == 0 else f"=LN(E{r}/E{p})"
        ws[f"G{r}"] = "" if i == 0 else f"=A{r}-1"
        if i >= 1:
            ws[f"H{r}"] = f'=IF(G{r}<{WARM},"",AVERAGE(F${r0 + 1}:F{p}))'
            ws[f"I{r}"] = f'=IF(G{r}<{WARM},"",STDEV(F${r0 + 1}:F{p}))'
            ws[f"J{r}"] = f'=IF(H{r}="","",F{r}-H{r})'
            ws[f"M{r}"] = f'=IF(I{r}="","",I{r}*SQRT({HALF}))'
            ws[f"N{r}"] = f'=IF(M{r}="","",EXP(M{r})-1)'
            ws[f"O{r}"] = f'=IF(M{r}="","",1-EXP(-M{r}))'
        ws[f"K{r}"] = f'=IF(A{r}<{WARM},"",IF(A{r}={WARM},0,K{p}+J{r}))' if i >= 1 else f'=IF(A{r}={WARM},0,"")'
        ws[f"L{r}"] = f'=IF(K{r}="","",INDEX($E${r0}:$E${rN},{WARM}+1)*EXP(K{r}))'
        # the zig-zag state machine: blank before the warm-up, starts at the warm-up's last week
        st = f'A{r}<{WARM}'
        if i == 0:
            ws[f"P{r}"] = f'=IF(A{r}={WARM},"waiting","")'
            for c, f in (("Q", f"K{r}"), ("R", f"A{r}"), ("S", f"K{r}"), ("T", f"A{r}"), ("U", f"K{r}"), ("V", f"A{r}")):
                ws[f"{c}{r}"] = f'=IF(A{r}={WARM},{f},"")'
            for c in ("W", "X", "Y"):
                ws[f"{c}{r}"] = ""
        else:
            ws[f"P{r}"] = (f'=IF({st},"",IF(A{r}={WARM},"waiting",IF(P{p}="waiting",IF(Q{r}-S{r}>=M{r},IF(R{r}>T{r},"rising","falling"),"waiting"),'
                           f'IF(P{p}="rising",IF(K{r}>U{p},"rising",IF(U{p}-K{r}>=M{r},"falling","rising")),'
                           f'IF(K{r}<U{p},"falling",IF(K{r}-U{p}>=M{r},"rising","falling"))))))')
            ws[f"Q{r}"] = f'=IF({st},"",IF(A{r}={WARM},K{r},IF(P{p}="waiting",MAX(Q{p},K{r}),Q{p})))'
            ws[f"R{r}"] = f'=IF({st},"",IF(A{r}={WARM},A{r},IF(AND(P{p}="waiting",K{r}>Q{p}),A{r},R{p})))'
            ws[f"S{r}"] = f'=IF({st},"",IF(A{r}={WARM},K{r},IF(P{p}="waiting",MIN(S{p},K{r}),S{p})))'
            ws[f"T{r}"] = f'=IF({st},"",IF(A{r}={WARM},A{r},IF(AND(P{p}="waiting",K{r}<S{p}),A{r},T{p})))'
            ws[f"X{r}"] = f'=IF(OR({st},A{r}={WARM}),"",IF(P{p}="waiting",IF(P{r}="waiting","",IF(P{r}="rising","TROUGH","PEAK")),IF(P{p}=P{r},"",IF(P{r}="falling","PEAK","TROUGH"))))'
            ws[f"Y{r}"] = f'=IF(X{r}="","",IF(P{p}="waiting",IF(X{r}="TROUGH",T{r},R{r}),V{p}))'
            ws[f"U{r}"] = (f'=IF({st},"",IF(A{r}={WARM},K{r},IF(P{p}="waiting",IF(P{r}="waiting",U{p},IF(P{r}="rising",Q{r},S{r})),'
                           f'IF(P{p}="rising",IF(OR(K{r}>U{p},X{r}<>""),K{r},U{p}),IF(OR(K{r}<U{p},X{r}<>""),K{r},U{p})))))')
            ws[f"V{r}"] = (f'=IF({st},"",IF(A{r}={WARM},A{r},IF(P{p}="waiting",IF(P{r}="waiting",V{p},IF(P{r}="rising",R{r},T{r})),'
                           f'IF(P{p}="rising",IF(OR(K{r}>U{p},X{r}<>""),A{r},V{p}),IF(OR(K{r}<U{p},X{r}<>""),A{r},V{p})))))')
            ws[f"W{r}"] = f'=IF(OR({st},A{r}={WARM}),"",IF(P{p}="waiting","",EXP(K{r}-U{p})-1))'
            ws[f"AA{r}"] = (f'=IF(X{r}="","",IF(P{p}="waiting","First swing: the path moved one threshold from its "&IF(X{r}="TROUGH","low","high")&" → week "&Y{r}&" was a "&X{r},'
                            f'IF(X{r}="PEAK","Fell "&TEXT(-W{r},"0.0%")&" from the high of week "&Y{r}&" (needed "&TEXT(O{r},"0.0%")&") → that high was a PEAK",'
                            f'"Rose "&TEXT(W{r},"0.0%")&" from the low of week "&Y{r}&" (needed "&TEXT(N{r},"0.0%")&") → that low was a TROUGH")))')
        ws[f"Z{r}"] = f'=IF(X{r}="","",COUNTIF(X${r0}:X{r},"PEAK")+COUNTIF(X${r0}:X{r},"TROUGH"))'
        ws[f"AB{r}"] = f'=IFERROR(INDEX($X${r0}:$X${rN},MATCH(A{r},$Y${r0}:$Y${rN},0)),"")'
    for c, f in (("B", "yyyy-mm-dd"), ("C", "#,##0.00"), ("D", "#,##0.00"), ("E", "0.0000"), ("F", "0.0000"), ("H", "0.00000"), ("I", "0.0000"), ("J", "0.0000"), ("K", "0.0000"),
                 ("L", "0.0000"), ("M", "0.0000"), ("N", "0.0%"), ("O", "0.0%"), ("Q", "0.0000"), ("S", "0.0000"), ("U", "0.0000"), ("W", "+0.0%;-0.0%;0.0%")):
        fmt(ws, c, r0, rN, f)
    ws.conditional_formatting.add(f"A{r0}:AA{rN}", FormulaRule(formula=[f'$X{r0}="PEAK"'], fill=PEAK_FILL))
    ws.conditional_formatting.add(f"A{r0}:AA{rN}", FormulaRule(formula=[f'$X{r0}="TROUGH"'], fill=TROUGH_FILL))
    ws.conditional_formatting.add(f"AB{r0}:AB{rN}", FormulaRule(formula=[f'$AB{r0}="PEAK"'], font=Font(name="Arial", size=10, bold=True, color="C2410C"), fill=PEAK_FILL))
    ws.conditional_formatting.add(f"AB{r0}:AB{rN}", FormulaRule(formula=[f'$AB{r0}="TROUGH"'], font=Font(name="Arial", size=10, bold=True, color="047857"), fill=TROUGH_FILL))
    ch = LineChart(); ch.title = "Ratio and drift-removed ratio (point-in-time)"; ch.y_axis.title = "ratio"; ch.height = 9; ch.width = 26
    ch.add_data(Reference(ws, min_col=5, min_row=7, max_row=rN), titles_from_data=True); ch.add_data(Reference(ws, min_col=12, min_row=7, max_row=rN), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=2, min_row=r0, max_row=rN)); ch.x_axis.number_format = "yyyy"; ch.x_axis.tickLblSkip = 104; ch.x_axis.tickMarkSkip = 52
    ch.roundedCorners = False; ch.legend.position = "b"; ch.legend.overlay = False; ch.title.overlay = False; ch.y_axis.title.overlay = False; ch.x_axis.delete = False; ch.y_axis.delete = False
    ws.add_chart(ch, "AD7")
    ws.freeze_panes = "C8"; all_font(ws)
    return r0, rN


# ---------------------------------------------------------------------------------------------------- Turning points
def turning(wb, r0, rN, ref):
    ws = wb.create_sheet("Turning points")
    S = "'Real data'!"; K = 40
    ws["A1"] = f"Turning points of {E.nice(PAIR)}, as Excel finds them — and as the Python script finds them"; ws["A1"].font = F_TITLE
    ws["A2"] = ("Columns B–K are formulas on 'Real data'. Columns L–M are the Python result (explore/E01_ratio_cycles/cycles.py: point_in_time + zigzag, the same rule) written in "
                "when this workbook was built; column N checks each row."); ws["A2"].font = F_NOTE
    ws["A3"] = "Found by Excel:"; ws["C3"] = "=COUNT(D7:D46)"; ws["E3"] = "Found by Python:"; ws["G3"] = "=COUNTA(L7:L46)"
    ws["I3"] = "Do they agree?"; ws["K3"] = '=IF(AND(C3=G3,COUNTIF(N7:N46,"yes")=G3),"YES — same weeks, same types","NO — see column N")'
    for c in ("A3", "E3", "I3"):
        ws[c].font = F_BOLD
    ws["K3"].font = Font(name="Arial", size=11, bold=True, color="047857")
    names = ["#", "Row in 'Real data' where it is confirmed", "Type", "Week # of the turning point", "Date of the turning point", "Ratio that week", "Drift-removed ratio that week",
             "Confirmed on (date)", "Weeks until confirmed", "Swing length since the previous turning point (weeks)", "Swing size (drift-removed)", "Python: date", "Python: type", "Match?"]
    header(ws, 6, names, [5, 12, 9, 10, 12, 10, 12, 12, 10, 14, 11, 12, 9, 12])
    for k in range(1, K + 1):
        r = 6 + k; p = r - 1
        ws[f"A{r}"] = k
        ws[f"B{r}"] = f'=IFERROR(MATCH(A{r},{S}$Z${r0}:$Z${rN},0),"")'
        ws[f"C{r}"] = f'=IF(B{r}="","",INDEX({S}$X${r0}:$X${rN},B{r}))'
        ws[f"D{r}"] = f'=IF(B{r}="","",INDEX({S}$Y${r0}:$Y${rN},B{r}))'
        ws[f"E{r}"] = f'=IF(D{r}="","",INDEX({S}$B${r0}:$B${rN},D{r}+1))'
        ws[f"F{r}"] = f'=IF(D{r}="","",INDEX({S}$E${r0}:$E${rN},D{r}+1))'
        ws[f"G{r}"] = f'=IF(D{r}="","",INDEX({S}$L${r0}:$L${rN},D{r}+1))'
        ws[f"H{r}"] = f'=IF(B{r}="","",INDEX({S}$B${r0}:$B${rN},B{r}))'
        ws[f"I{r}"] = f'=IF(B{r}="","",INDEX({S}$A${r0}:$A${rN},B{r})-D{r})'
        ws[f"J{r}"] = "" if k == 1 else f'=IF(OR(D{r}="",D{p}=""),"",D{r}-D{p})'
        ws[f"K{r}"] = "" if k == 1 else f'=IF(OR(G{r}="",G{p}=""),"",G{r}/G{p}-1)'
        if k <= len(ref):
            ws[f"L{r}"] = ref[k - 1][0].to_pydatetime(); ws[f"M{r}"] = ref[k - 1][1]
            ws[f"L{r}"].font = F_INPUT; ws[f"M{r}"].font = F_INPUT
        ws[f"N{r}"] = f'=IF(L{r}="",IF(E{r}="","","extra in Excel"),IF(AND(E{r}=L{r},C{r}=M{r}),"yes","NO"))'
    for c, f in (("E", "yyyy-mm-dd"), ("H", "yyyy-mm-dd"), ("L", "yyyy-mm-dd"), ("F", "0.0000"), ("G", "0.0000"), ("K", "+0.0%;-0.0%;0.0%")):
        fmt(ws, c, 7, 6 + K, f)
    ws.conditional_formatting.add("A7:N46", FormulaRule(formula=['$C7="PEAK"'], fill=PEAK_FILL))
    ws.conditional_formatting.add("A7:N46", FormulaRule(formula=['$C7="TROUGH"'], fill=TROUGH_FILL))
    ws.freeze_panes = "A7"; all_font(ws)


def main():
    px = load_daily(); pf = pair_frame(px, PAIR, "plain"); d = pf[pf["decision"]]; wk, ca, cb = d["ratio"], d["ca"], d["cb"]
    # the Python reference: the same point_in_time + zigzag that cycles.py runs
    g = np.diff(np.log(wk.to_numpy())); _, _, _, z, hx = E.point_in_time(g)
    tps = E.zigzag(np.concatenate([[0.0], np.cumsum(z)]).tolist(), hx.tolist())
    ref = [(wk.index[E.BURN + i], "PEAK" if t > 0 else "TROUGH") for i, t in tps]
    wb = Workbook(); how_it_works(wb, len(wk), wk.index[0].date(), wk.index[-1].date()); toy(wb); r0, rN = real(wb, wk, ca, cb); turning(wb, r0, rN, ref)
    os.makedirs(OUT, exist_ok=True); out = os.path.join(OUT, "E01_peaks_troughs_explained.xlsx"); wb.save(out)
    print(f"wrote {out}: {len(wk)} weeks, Python reference {len(ref)} turning points")


if __name__ == "__main__":
    main()
