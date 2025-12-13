import pandas as pd
import numpy as np
import random
import math

#EDGE CASE MUTATIONS DICTIONARY

def safe_assign(df, idx, col, val):
    dt = df[col].dtype
    import pandas as pd
    import numpy as np

    if val is None or (isinstance(val, float) and np.isnan(val)):
        if pd.api.types.is_integer_dtype(dt):
            df.at[idx, col] = None
        else:
            df.at[idx, col] = np.nan
        return

    if pd.api.types.is_integer_dtype(dt):
        df.at[idx, col] = int(val)
        return

    if pd.api.types.is_float_dtype(dt):
        df.at[idx, col] = float(val)
        return
    
    if pd.api.types.is_datetime64_any_dtype(dt):
        df.at[idx, col] = pd.to_datetime(val)
        return

    df.at[idx, col] = str(val)

def mutate_large_numbers(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col] * random.randint(10_000, 200_000)
    return df


def mutate_small_numbers(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col] / random.randint(1000, 100000)
    return df


def mutate_negative_numbers(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = np.random.rand(len(df)) < 0.3
            df.loc[mask, col] = -abs(df.loc[mask, col])
    return df


def mutate_extreme_outliers(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            idx = random.randint(0, len(df) - 1)
            value = df[col].mean() * random.randint(1000, 5000)

            if pd.api.types.is_integer_dtype(df[col].dtype):
                value = int(value)
            elif pd.api.types.is_float_dtype(df[col].dtype):
                value = float(value)
            else:
                value = str(value)

            safe_assign(df, idx, col, value)
    return df


def mutate_zero_values(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = np.random.rand(len(df)) < 0.25
            df.loc[mask, col] = 0
    return df


def mutate_large_variance(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x * random.uniform(0.1, 50))
    return df


def mutate_constant_values(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].iloc[0]
            df[col] = val
    return df


def mutate_monotonic_increasing(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = np.sort(df[col].values)
    return df


def mutate_monotonic_decreasing(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = np.sort(df[col].values)[::-1]
    return df


def mutate_missing_numeric(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = np.random.rand(len(df)) < 0.30
            df.loc[mask, col] = None
    return df


def mutate_nan_infinity(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            idx = random.randint(0, len(df) - 1)
            safe_assign(df, idx, col, np.inf)
    return df


def mutate_nan_negative_infinity(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            idx = random.randint(0, len(df) - 1)
            safe_assign(df, idx, col, -np.inf)
    return df


def mutate_nan_values(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            idx = random.randint(0, len(df) - 1)
            safe_assign(df, idx, col, np.nan)
    return df


def mutate_rounding_edge(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(0)
    return df


def mutate_repeating_decimals(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: float(str(x)[:6] + '333'))
    return df


def mutate_extreme_ratios(df):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        a, b = num_cols[:2]
        df[b] = df[b].replace(0, 1e-9)
        df[a] = df[a] / df[b]
    return df


def mutate_float_precision(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("float64") / random.uniform(1e6, 1e12)
    return df


def mutate_skewed_distribution(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = np.random.exponential(scale=2, size=len(df))
    return df

def mutate_empty_strings(df):
    for col in df:
        if df[col].dtype == object:
            mask = np.random.rand(len(df)) < 0.30
            df.loc[mask, col] = ""
    return df


def mutate_whitespace_strings(df):
    for col in df:
        if df[col].dtype == object:
            mask = np.random.rand(len(df)) < 0.20
            df.loc[mask, col] = " " * random.randint(1, 10)
    return df


def mutate_very_long_strings(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x + "A" * random.randint(100, 500))
    return df


def mutate_unicode_strings(df):
    chars = "你好안녕こんにちはمرحبا"
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: (
                    str(x) + random.choice(chars)
                    if isinstance(x, str)
                    else x
                )
            )
    return df


def mutate_special_characters(df):
    chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>/?"
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: (
                    str(x) + random.choice(chars)
                    if isinstance(x, str)
                    else x
                )
            )
    return df


def mutate_mixed_casing(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: ''.join(
                c.upper() if random.random() < 0.5 else c.lower()
                for c in x
            ))
    return df


def mutate_inconsistent_categories(df):
    for col in df:
        if df[col].dtype == object:
            mask = np.random.rand(len(df)) < 0.30
            df.loc[mask, col] = random.choice(["UNKNOWN", "Other", "misc"])
    return df


def mutate_duplicate_strings(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].sample(frac=1).values
    return df


def mutate_html_strings(df):
    tags = ["<b>", "<i>", "<div>", "<span>", "<p>"]
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: random.choice(tags) + x + "</>")
    return df


def mutate_sql_injection_strings(df):
    payloads = ["' OR 1=1 --", "DROP TABLE users;", "'; SELECT * FROM admin; --"]
    for col in df:
        if df[col].dtype == object:
            mask = np.random.rand(len(df)) < 0.15
            df.loc[mask, col] = random.choice(payloads)
    return df


def mutate_multiple_spaces(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.replace(" ", "   "))
    return df


def mutate_newline_strings(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x + "\nnewline")
    return df


def mutate_tab_strings(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x + "\tTAB")
    return df


def mutate_leading_trailing_spaces(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: "  " + x + "  ")
    return df


def mutate_repeated_words(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x + " " + x.split()[0])
    return df


def mutate_random_emoji(df):
    emojis = "😀😎🤖🔥💀✨🐍💯"
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x + random.choice(emojis))
    return df

def mutate_duplicate_rows(df):
    if len(df) > 0:
        dup = df.sample(frac=0.1, replace=True)
        return pd.concat([df, dup], ignore_index=True)
    return df


def mutate_sorted_ascending(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            return df.sort_values(by=col, ascending=True).reset_index(drop=True)
    return df


def mutate_sorted_descending(df):
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            return df.sort_values(by=col, ascending=False).reset_index(drop=True)
    return df


def mutate_shuffled_rows(df):
    return df.sample(frac=1).reset_index(drop=True)


def mutate_swap_columns(df):
    cols = df.columns.tolist()
    if len(cols) > 1:
        i, j = random.sample(range(len(cols)), 2)
        df = df.copy()
        df[cols[i]], df[cols[j]] = df[cols[j]], df[cols[i]]
    return df


def mutate_remove_column(df):
    if len(df.columns) > 1:
        col = random.choice(list(df.columns))
        return df.drop(columns=[col])
    return df


def mutate_add_dummy_column(df):
    df["dummy_column"] = [random.randint(0, 10) for _ in range(len(df))]
    return df


def mutate_extremely_wide_table(df):
    for i in range(20):
        df[f"extra_col_{i}"] = np.random.randint(0, 100, size=len(df))
    return df


def mutate_multicollinearity(df):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 1:
        base = num_cols[0]
        for col in num_cols[1:]:
            df[col] = df[base] * random.uniform(0.9, 1.1)
    return df


def mutate_high_cardinality_strings(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = [f"unique_{i}_{random.randint(1,9999)}" for i in range(len(df))]
    return df

def mutate_invalid_dates(df):
    for col in df:
        if "date" in col.lower():
            df[col] = ["2023-99-99"] * len(df)
    return df


def mutate_far_future_dates(df):
    for col in df:
        if "date" in col.lower():
            df[col] = ["2500-01-01"] * len(df)
    return df


def mutate_far_past_dates(df):
    for col in df:
        if "date" in col.lower():
            df[col] = ["1800-01-01"] * len(df)
    return df


def mutate_unix_timestamps(df):
    for col in df:
        if "date" in col.lower():
            df[col] = [random.randint(1_000_000_000, 2_000_000_000)]
    return df


def mutate_randomized_dates(df):
    for col in df:
        if "date" in col.lower():
            df[col] = pd.date_range("2000-01-01", periods=len(df), freq="D")
    return df


def mutate_null_dates(df):
    for col in df:
        if "date" in col.lower():
            mask = np.random.rand(len(df)) < 0.40
            df.loc[mask, col] = None
    return df


def mutate_string_dates(df):
    for col in df:
        if "date" in col.lower():
            df[col] = ["YEAR2020-DAY12"] * len(df)
    return df


def mutate_timezone_dates(df):
    for col in df:
        if "date" in col.lower():
            df[col] = ["2024-01-01T12:00:00Z"] * len(df)
    return df

def mutate_boolean_flip(df):
    for col in df:
        if df[col].dtype == bool:
            df[col] = ~df[col]
    return df


def mutate_boolean_to_strings(df):
    for col in df:
        if df[col].dtype == bool:
            df[col] = df[col].apply(lambda x: "YES" if x else "NO")
    return df


def mutate_random_boolean(df):
    for col in df:
        if df[col].dtype == bool:
            df[col] = np.random.rand(len(df)) < 0.5
    return df


def mutate_unexpected_enum(df):
    for col in df:
        if df[col].dtype == object:
            mask = np.random.rand(len(df)) < 0.20
            df.loc[mask, col] = "UNEXPECTED_OPTION"
    return df


def mutate_enum_case_variants(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.upper() if random.random() < 0.5 else x.lower())
    return df


def mutate_enum_misspellings(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x + random.choice(["_typo", "_miss"]))
    return df

def mutate_join_key_collisions(df):
    if "id" in df.columns:
        df["id"] = df["id"].apply(lambda x: random.choice([x, random.randint(1, 3)]))
    return df


def mutate_join_key_nulls(df):
    if "id" in df.columns:
        mask = np.random.rand(len(df)) < 0.20
        df.loc[mask, "id"] = None
    return df


def mutate_duplicate_join_keys(df):
    if "id" in df.columns:
        values = df["id"].tolist()
        for i in range(len(values)//3):
            df.at[i, "id"] = values[-1]
    return df


def mutate_mixed_join_types(df):
    if "id" in df.columns:
        mask = np.random.rand(len(df)) < 0.50
        df.loc[mask, "id"] = df.loc[mask, "id"].astype(str)
    return df


SCENARIO_REGISTRY = {

    #Numeric cases

    "large_numbers": {
        "desc": "Numeric columns contain extremely large values.",
        "mutate": mutate_large_numbers,
    },
    "small_numbers": {
        "desc": "Numeric columns contain extremely small fractional values.",
        "mutate": mutate_small_numbers,
    },
    "negative_numbers": {
        "desc": "Inject negative values randomly.",
        "mutate": mutate_negative_numbers,
    },
    "extreme_outliers": {
        "desc": "Single-row extreme numeric outliers.",
        "mutate": mutate_extreme_outliers,
    },
    "zero_values": {
        "desc": "Random zeros in numeric columns.",
        "mutate": mutate_zero_values,
    },
    "large_variance": {
        "desc": "Very wide numeric spread within a column.",
        "mutate": mutate_large_variance,
    },
    "constant_values": {
        "desc": "All numeric values identical (degenerate distribution).",
        "mutate": mutate_constant_values,
    },
    "monotonic_increasing": {
        "desc": "Numeric columns sorted ascending.",
        "mutate": mutate_monotonic_increasing,
    },
    "monotonic_decreasing": {
        "desc": "Numeric columns sorted descending.",
        "mutate": mutate_monotonic_decreasing,
    },
    "missing_numeric": {
        "desc": "Inject missing values into numeric columns.",
        "mutate": mutate_missing_numeric,
    },
    "nan_infinity": {
        "desc": "Inject +infinity into numeric columns.",
        "mutate": mutate_nan_infinity,
    },
    "nan_negative_infinity": {
        "desc": "Inject -infinity into numeric values.",
        "mutate": mutate_nan_negative_infinity,
    },
    "nan_values": {
        "desc": "Inject NaN values into numeric columns.",
        "mutate": mutate_nan_values,
    },
    "rounding_edges": {
        "desc": "Round all numeric values.",
        "mutate": mutate_rounding_edge,
    },
    "repeating_decimals": {
        "desc": "Introduce patterns of repeating decimals.",
        "mutate": mutate_repeating_decimals,
    },
    "extreme_ratios": {
        "desc": "Extreme ratios or division-based values.",
        "mutate": mutate_extreme_ratios,
    },
    "float_precision": {
        "desc": "Large float precision reductions.",
        "mutate": mutate_float_precision,
    },
    "skewed_distribution": {
        "desc": "Numeric columns follow exponential distribution.",
        "mutate": mutate_skewed_distribution,
    },

    #String cases

    "empty_strings": {
        "desc": "Text fields contain empty strings.",
        "mutate": mutate_empty_strings,
    },
    "whitespace_strings": {
        "desc": "Text values replaced with whitespace.",
        "mutate": mutate_whitespace_strings,
    },
    "very_long_strings": {
        "desc": "Extremely long text values (hundreds of chars).",
        "mutate": mutate_very_long_strings,
    },
    "unicode_strings": {
        "desc": "Add unicode characters (CJK, Arabic, etc.).",
        "mutate": mutate_unicode_strings,
    },
    "special_characters": {
        "desc": "Inject special punctuation symbols.",
        "mutate": mutate_special_characters,
    },
    "mixed_casing": {
        "desc": "Random changes between upper/lowercase.",
        "mutate": mutate_mixed_casing,
    },
    "inconsistent_categories": {
        "desc": "Inject inconsistent category labels.",
        "mutate": mutate_inconsistent_categories,
    },
    "duplicate_strings": {
        "desc": "String columns reshuffled producing duplicates.",
        "mutate": mutate_duplicate_strings,
    },
    "html_strings": {
        "desc": "Add HTML tags inside text.",
        "mutate": mutate_html_strings,
    },
    "sql_injection_strings": {
        "desc": "Insert SQL injection-like payloads.",
        "mutate": mutate_sql_injection_strings,
    },
    "multiple_spaces": {
        "desc": "Replace single spaces with multiple spaces.",
        "mutate": mutate_multiple_spaces,
    },
    "newline_strings": {
        "desc": "Add newline characters to text.",
        "mutate": mutate_newline_strings,
    },
    "tab_strings": {
        "desc": "Add tab characters inside text.",
        "mutate": mutate_tab_strings,
    },
    "leading_trailing_spaces": {
        "desc": "Pad strings with whitespace.",
        "mutate": mutate_leading_trailing_spaces,
    },
    "repeated_words": {
        "desc": "Duplicate first word in each text value.",
        "mutate": mutate_repeated_words,
    },
    "emoji_strings": {
        "desc": "Append random emoji to strings.",
        "mutate": mutate_random_emoji,
    },

    #Structural cases

    "duplicate_rows": {
        "desc": "Duplicate a subset of rows.",
        "mutate": mutate_duplicate_rows,
    },
    "sorted_ascending": {
        "desc": "Sort a numeric column ascending.",
        "mutate": mutate_sorted_ascending,
    },
    "sorted_descending": {
        "desc": "Sort a numeric column descending.",
        "mutate": mutate_sorted_descending,
    },
    "shuffled_rows": {
        "desc": "Completely shuffle row order.",
        "mutate": mutate_shuffled_rows,
    },
    "swap_columns": {
        "desc": "Randomly swap two column values.",
        "mutate": mutate_swap_columns,
    },
    "remove_column": {
        "desc": "Remove a random column.",
        "mutate": mutate_remove_column,
    },
    "add_dummy_column": {
        "desc": "Add a useless dummy column.",
        "mutate": mutate_add_dummy_column,
    },
    "wide_table": {
        "desc": "Add 20+ extra numeric columns.",
        "mutate": mutate_extremely_wide_table,
    },
    "multicollinearity": {
        "desc": "Make numeric columns linearly dependent.",
        "mutate": mutate_multicollinearity,
    },
    "high_cardinality_strings": {
        "desc": "Each row gets a unique long string id.",
        "mutate": mutate_high_cardinality_strings,
    },

    #Date/time cases

    "invalid_dates": {
        "desc": "Assign impossible dates like 2023-99-99.",
        "mutate": mutate_invalid_dates,
    },
    "far_future_dates": {
        "desc": "Dates far in the future.",
        "mutate": mutate_far_future_dates,
    },
    "far_past_dates": {
        "desc": "Dates far in the past.",
        "mutate": mutate_far_past_dates,
    },
    "unix_timestamps": {
        "desc": "Replace dates with unix timestamps.",
        "mutate": mutate_unix_timestamps,
    },
    "randomized_dates": {
        "desc": "Sequential randomized dates.",
        "mutate": mutate_randomized_dates,
    },
    "null_dates": {
        "desc": "Randomly null out date values.",
        "mutate": mutate_null_dates,
    },
    "string_dates": {
        "desc": "Replace dates with malformed strings.",
        "mutate": mutate_string_dates,
    },
    "timezone_dates": {
        "desc": "UTC timestamp-formatted dates.",
        "mutate": mutate_timezone_dates,
    },

    #Logical cases

    "boolean_flip": {
        "desc": "Flip boolean values.",
        "mutate": mutate_boolean_flip,
    },
    "boolean_to_strings": {
        "desc": "Convert BOOL values to YES/NO.",
        "mutate": mutate_boolean_to_strings,
    },
    "random_booleans": {
        "desc": "Randomize boolean values.",
        "mutate": mutate_random_boolean,
    },
    "unexpected_enum": {
        "desc": "Insert enum values not seen in data.",
        "mutate": mutate_unexpected_enum,
    },
    "enum_case_variants": {
        "desc": "Randomize casing for enum-like text.",
        "mutate": mutate_enum_case_variants,
    },
    "enum_misspellings": {
        "desc": "Inject slight spelling errors.",
        "mutate": mutate_enum_misspellings,
    },

    #Relational cases

    "join_key_collisions": {
        "desc": "Repeated ID values that cause collisions.",
        "mutate": mutate_join_key_collisions,
    },
    "join_key_nulls": {
        "desc": "Null out join keys.",
        "mutate": mutate_join_key_nulls,
    },
    "duplicate_join_keys": {
        "desc": "Make multiple rows share join-key values.",
        "mutate": mutate_duplicate_join_keys,
    },
    "mixed_join_types": {
        "desc": "Join key values sometimes strings, sometimes ints.",
        "mutate": mutate_mixed_join_types,
    },
}

__all__ = ["SCENARIO_REGISTRY"]