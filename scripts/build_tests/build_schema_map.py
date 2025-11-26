def build_schema_map(schema_entry):
    """
    Convert Spider/BIRD schema JSON into:
        {
            table_name : { column_name : column_type }
        }
    """

    table_names = schema_entry["table_names_original"]
    column_names = schema_entry["column_names_original"]
    column_types = schema_entry["column_types"]

    schema_map = {}

    # Create entry for each table
    for tbl in table_names:
        schema_map[tbl.lower()] = {}   # normalize to lowercase keys

    # Populate columns
    for (table_id, col_name), col_type in zip(column_names, column_types):

        # Skip special column (-1, "*")
        if table_id == -1 or col_name == "*":
            continue

        tbl = table_names[table_id].lower()

        # Use original column name EXACTLY as appears in BIRD JSON
        schema_map[tbl][col_name] = col_type.lower()

    return schema_map
