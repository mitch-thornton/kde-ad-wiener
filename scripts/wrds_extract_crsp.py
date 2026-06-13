#!/usr/bin/env python3
"""wrds_extract_crsp.py -- extract CRSP daily returns from WRDS for the AD-KDE financial study.

RUN THIS ON YOUR OWN WRDS ACCOUNT. Requires the `wrds` package and valid WRDS credentials
(uses ~/.pgpass or prompts on first connect). Writes everything into a `data/` subdirectory under
the current working directory:

    data/crsp_market_daily.csv   date, vwretd, ewretd, sprtrn   (CRSP daily market indices)
    data/crsp_stock_daily.csv    permno, ticker, date, ret      (individual common stocks)
    data/extract_meta.json       schema used, date range, universe, row counts, notes

Then run `analyze_crsp_adkde.py` (no WRDS needed) to produce results/*.json, and upload those.

CRSP exists in two flavors on WRDS: the new CIZ format (crsp.dsf_v2 / crsp.stksecurityinfohist,
date column dlycaldt, return column dlyret, delisting returns already merged in) and the legacy
SIZ format (crsp.dsf / crsp.stocknames / crsp.dsi, columns caldt / ret). This script auto-detects
which is available and adapts; column lists for the index file are also auto-detected.
"""
import os, json, argparse, datetime

# ----------------------------- configuration -------------------------------
START_DATE = "2005-01-01"
END_DATE   = "2024-12-31"
TICKERS = ["AAPL", "MSFT", "XOM", "JPM", "PG", "GE", "KO", "WMT", "JNJ", "INTC"]
OUTDIR = "data"
# ---------------------------------------------------------------------------


def has_table(db, schema, table):
    q = ("select 1 from information_schema.tables "
         "where table_schema=%(s)s and table_name=%(t)s limit 1")
    try:
        return len(db.raw_sql(q, params={"s": schema, "t": table})) > 0
    except Exception:
        return False

def columns(db, schema, table):
    q = ("select column_name from information_schema.columns "
         "where table_schema=%(s)s and table_name=%(t)s")
    try:
        return set(c.lower() for c in db.raw_sql(q, params={"s": schema, "t": table})["column_name"])
    except Exception:
        return set()


def extract_market(db, ciz, start, end):
    """Daily market indices (value-weighted, equal-weighted, S&P 500 returns)."""
    if ciz and has_table(db, "crsp", "wrds_dailyindexret_query"):
        table = "wrds_dailyindexret_query"
    elif ciz and has_table(db, "crsp", "dsi_v2"):
        table = "dsi_v2"
    else:
        table = "dsi"
    cols = columns(db, "crsp", table)
    datecol = "caldt" if "caldt" in cols else ("dlycaldt" if "dlycaldt" in cols else None)
    rets = [c for c in ("vwretd", "ewretd", "sprtrn") if c in cols]
    if datecol is None or not rets:
        raise RuntimeError("could not locate date/return columns in crsp.%s (found: %s)"
                           % (table, sorted(cols)))
    sel = ", ".join(["%s as date" % datecol] + rets)
    mkt = db.raw_sql(
        "select %s from crsp.%s where %s between %%(s)s and %%(e)s order by %s"
        % (sel, table, datecol, datecol),
        params={"s": start, "e": end}, date_cols=["date"],
    )
    return mkt, table, rets


def extract_stocks(db, ciz, tickers, start, end):
    """Individual common-stock daily returns (NYSE/AMEX/Nasdaq, common shares)."""
    if ciz:
        sql = """
            select a.permno, a.dlycaldt as date, a.dlyret as ret, b.ticker
            from crsp.dsf_v2 a
            inner join crsp.stksecurityinfohist b
              on a.permno = b.permno
             and a.dlycaldt between b.secinfostartdt and b.secinfoenddt
            where a.dlycaldt between %(s)s and %(e)s
              and a.dlyret is not null
              and b.sharetype = 'NS' and b.securitytype = 'EQTY'
              and b.securitysubtype = 'COM' and b.usincflg = 'Y'
              and b.issuertype in ('ACOR','CORP')
              and b.primaryexch in ('N','A','Q')
              and b.conditionaltype in ('RW','NW')
              and b.tradingstatusflg = 'A'
              and b.ticker in %(tk)s
            order by a.permno, a.dlycaldt
        """
        table = "crsp.dsf_v2 + crsp.stksecurityinfohist (CIZ)"
    else:
        sql = """
            select a.permno, a.date as date, a.ret, b.ticker
            from crsp.dsf a
            inner join crsp.stocknames b
              on a.permno = b.permno and a.date between b.namedt and b.nameenddt
            where a.date between %(s)s and %(e)s
              and a.ret is not null
              and b.shrcd in (10,11) and b.exchcd in (1,2,3)
              and b.ticker in %(tk)s
            order by a.permno, a.date
        """
        table = "crsp.dsf + crsp.stocknames (SIZ)"
    stocks = db.raw_sql(sql, params={"s": start, "e": end, "tk": tuple(tickers)},
                        date_cols=["date"])
    return stocks, table


def main():
    ap = argparse.ArgumentParser(description="Extract CRSP daily returns from WRDS (CIZ or SIZ).")
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--end", default=END_DATE)
    ap.add_argument("--tickers", default=",".join(TICKERS))
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--wrds-username", default=None)
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    import wrds
    db = wrds.Connection(wrds_username=args.wrds_username) if args.wrds_username else wrds.Connection()

    ciz = has_table(db, "crsp", "dsf_v2")
    print("CRSP format detected: %s" % ("CIZ (dsf_v2)" if ciz else "legacy SIZ (dsf)"))

    print("[1/2] market indices ...")
    mkt, mkt_table, mkt_rets = extract_market(db, ciz, args.start, args.end)
    mkt.to_csv(os.path.join(args.outdir, "crsp_market_daily.csv"), index=False)

    print("[2/2] individual stocks for %d tickers ..." % len(tickers))
    stocks, stk_table = extract_stocks(db, ciz, tickers, args.start, args.end)
    stocks = stocks[["permno", "ticker", "date", "ret"]]
    stocks.to_csv(os.path.join(args.outdir, "crsp_stock_daily.csv"), index=False)
    db.close()

    permnos = sorted(set(int(p) for p in stocks["permno"].tolist())) if len(stocks) else []
    meta = {
        "source": "WRDS / CRSP",
        "format": "CIZ" if ciz else "SIZ",
        "market_table": mkt_table, "market_return_cols": mkt_rets,
        "stock_table": stk_table,
        "extracted": datetime.datetime.now().isoformat(timespec="seconds"),
        "start": args.start, "end": args.end,
        "tickers_requested": tickers, "permnos_resolved": permnos,
        "rows_market": int(len(mkt)), "rows_stock": int(len(stocks)),
        "notes": {
            "ret": "daily holding-period return incl. distributions; CIZ dlyret also includes delisting",
            "indices": "vwretd/ewretd value- and equal-weighted incl. dividends; sprtrn = S&P 500",
            "filters": "common stock on NYSE/AMEX/Nasdaq",
        },
    }
    with open(os.path.join(args.outdir, "extract_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if len(stocks) == 0:
        print("WARNING: no individual-stock rows returned. Check ticker spellings and date range; "
              "CRSP tickers are historical and can differ from current symbols.")
    print("done. market rows: %d ; stock rows: %d ; permnos: %d"
          % (len(mkt), len(stocks), len(permnos)))


if __name__ == "__main__":
    main()
