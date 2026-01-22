-- COCOON BLOCK START: PLEASE DO NOT MODIFY THIS BLOCK FOR SELF-MAINTENANCE
-- Generated at 2025-12-02 14:38:22.642667+00:00
WITH 
"movies_trimmed" AS (
    -- Trim Leading and Trailing Spaces
    SELECT
        "Id",
        "Name",
        "Release Date",
        "Director",
        "Creator",
        "Actors",
        "Cast",
        "Language",
        "Country",
        "Duration",
        "RatingValue",
        "RatingCount",
        "ReviewCount",
        "Genre",
        "Filming Locations",
        "Description",
        TRIM("Year") AS "Year"
    FROM "movies"
),

"movies_trimmed_null" AS (
    -- NULL Imputation: Impute Null to Disguised Missing Values
    -- Id: ['xxxxx']
    -- Description: ['Add a Plot', 'No Description', 'The plot is unknown at this time.', 'The plot is unknown.', 'Plot is unknown.']
    SELECT 
        CASE
            WHEN "Id" = 'xxxxx' THEN NULL
            ELSE "Id"
        END AS "Id",
        CASE
            WHEN "Description" = 'Add a Plot' THEN NULL
            WHEN "Description" = 'No Description' THEN NULL
            WHEN "Description" = 'The plot is unknown at this time.' THEN NULL
            WHEN "Description" = 'The plot is unknown.' THEN NULL
            WHEN "Description" = 'Plot is unknown.' THEN NULL
            ELSE "Description"
        END AS "Description",
        "Release Date",
        "Genre",
        "Duration",
        "Name",
        "Year",
        "Director",
        "Country",
        "ReviewCount",
        "Cast",
        "Actors",
        "RatingValue",
        "RatingCount",
        "Language",
        "Filming Locations",
        "Creator"
    FROM "movies_trimmed"
),

"movies_trimmed_null_casted" AS (
    -- Column Type Casting: 
    -- Actors: from VARCHAR to ARRAY
    -- Cast: from VARCHAR to ARRAY
    -- Creator: from VARCHAR to ARRAY
    -- Duration: from VARCHAR to INT
    -- Genre: from VARCHAR to ARRAY
    -- Language: from VARCHAR to ARRAY
    -- RatingCount: from VARCHAR to INT
    -- RatingValue: from VARCHAR to DECIMAL
    -- Release Date: from VARCHAR to DATE
    -- Year: from VARCHAR to INT
    SELECT
        "Id",
        "Description",
        "Name",
        "Director",
        "Country",
        "ReviewCount",
        "Filming Locations",
        SPLIT("Actors", ',') 
        AS "Actors",
        string_split("Cast", ',') 
        AS "Cast",
        SPLIT("Creator", ',') 
        AS "Creator",
        CAST(REGEXP_EXTRACT("Duration", '(\d+)') AS INT) 
        AS "Duration",
        SPLIT("Genre", ',') 
        AS "Genre",
        string_split("Language", ',') 
        AS "Language",
        CAST(REPLACE("RatingCount", ',', '') AS INT) 
        AS "RatingCount",
        CAST(REGEXP_EXTRACT("RatingValue", '(\d+(\.\d+)?)') AS DECIMAL) 
        AS "RatingValue",
        CAST(COALESCE(try_strptime(replace("Release Date", ' (USA)', ''), '%d %B %Y'), try_strptime(replace("Release Date", ' (USA)', ''), '%Y')) AS DATE) 
        AS "Release Date",
        CAST(REGEXP_EXTRACT("Year", '(\d{4})') AS INT) 
        AS "Year"
    FROM "movies_trimmed_null"
)

-- COCOON BLOCK END
SELECT *
FROM "movies_trimmed_null_casted"