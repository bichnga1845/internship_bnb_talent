import io
import pandas as pd

import app


def summarize_dataset():
    df = pd.read_csv(r"E:\Internship_BNB_talent\data\synthetic_leads.csv")
    ids = df["Lead ID"].dropna().astype(str).str.strip()
    ids = ids[ids != ""]
    print("DATASET_SUMMARY")
    print(f"total_records={len(df)}")
    print(f"unique_non_empty_lead_ids={ids.nunique()}")
    print("first_10_unique_ids")
    for value in ids.unique()[:10]:
        print(value)
    return df, ids.unique().tolist()


def check_search_flow(df):
    lead_options = app._lead_search_options(df)
    selected = lead_options[:3]
    print("SEARCH_FLOW")
    print(f"lead_options_count={len(lead_options)}")
    for target in selected:
        partial = target[:5]
        matches = app._lead_search_matches(lead_options, partial)
        print(f"target={target}; partial={partial}; matches_contains_target={target in matches}; match_count={len(matches)}")
        coerced = app._coerce_selected_lead_id(lead_options, target)
        print(f"coerced={coerced}")


def check_uploaded_file_flow():
    df = pd.read_csv(r"E:\Internship_BNB_talent\data\synthetic_leads.csv")
    csv_buffer = io.BytesIO(df.head(20).to_csv(index=False).encode("utf-8"))
    csv_buffer.name = "uploaded_leads.csv"
    csv_df, validation = app.load_dataset_from_source(uploaded_file=csv_buffer, use_synthetic=False)
    print("UPLOADED_CSV")
    print(f"rows={len(csv_df)}; valid_rows={validation.get('valid_rows', 0)}; invalid_rows={validation.get('invalid_rows', 0)}")
    csv_ids = app._lead_search_options(csv_df)
    print(f"csv_lead_options_count={len(csv_ids)}")
    for target in csv_ids[:3]:
        print(f"csv_target={target}; matches={app._lead_search_matches(csv_ids, target[:5])[:5]}")

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        df.head(20).to_excel(writer, index=False)
    xlsx_buffer.seek(0)
    xlsx_buffer.name = "uploaded_leads.xlsx"
    xlsx_df, xlsx_validation = app.load_dataset_from_source(uploaded_file=xlsx_buffer, use_synthetic=False)
    print("UPLOADED_XLSX")
    print(f"rows={len(xlsx_df)}; valid_rows={xlsx_validation.get('valid_rows', 0)}; invalid_rows={xlsx_validation.get('invalid_rows', 0)}")
    xlsx_ids = app._lead_search_options(xlsx_df)
    print(f"xlsx_lead_options_count={len(xlsx_ids)}")
    for target in xlsx_ids[:3]:
        print(f"xlsx_target={target}; matches={app._lead_search_matches(xlsx_ids, target[:5])[:5]}")


if __name__ == "__main__":
    df, _ = summarize_dataset()
    check_search_flow(df)
    check_uploaded_file_flow()
