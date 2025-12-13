def build_schema_map(schema_entry):

    #BUILDS THE SCHEMA MAP FOR PROMPTING AND GENERATION

    table_names = schema_entry["table_names_original"]
    column_names = schema_entry["column_names_original"]
    column_types = schema_entry["column_types"]

    schema_map = {}

    for tbl in table_names:
        schema_map[tbl.lower()] = {}

    for (table_id, col_name), col_type in zip(column_names, column_types):

        if table_id == -1 or col_name == "*":
            continue

        tbl = table_names[table_id].lower()

        schema_map[tbl][col_name] = col_type.lower()

    return schema_map
