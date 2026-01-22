-- COCOON BLOCK START: PLEASE DO NOT MODIFY THIS BLOCK FOR SELF-MAINTENANCE
-- Generated at 2025-12-02 14:17:36.559750+00:00
WITH 
"beers_cleaned" AS (
    -- Clean unusual string values: 
    -- ounces: The 'ounces' column has several inconsistencies: 1.  **Inconsistent unit abbreviations**: The unit is represented as 'oz.', 'oz', 'ounce', and 'OZ.'. The most frequent and standard abbreviation is 'oz.'. 2.  **Inclusion of product types**: Some entries include product types like 'Alumi-Tek' or 'Silo Can' appended to the ounce value, which should be removed to isolate the measurement. 3.  **Case inconsistency**: 'oz.' vs 'OZ.' for the abbreviation.To fix these issues, all values will be standardized to the format '[NUMBER] oz.'. This involves: -   Extracting only the numerical part. -   Standardizing the unit abbreviation to 'oz.' (lowercase with a dot). -   Removing any extraneous product type information.
    -- abv: The `abv` column has inconsistent formats for representing Alcohol By Volume. The standard and most frequent format is a decimal string (e.g., '0.05' for 5% ABV). However, a number of values incorrectly append a percentage sign '%' to this decimal representation (e.g., '0.05%', '0.065%'). This is semantically incorrect, as '0.05%' literally means 0.0005, which is an unlikely ABV. The intended value was the decimal itself. The fix is to remove the trailing '%' from these inconsistent entries to standardize the entire column to the decimal format.
    SELECT
        "id",
        "beer_name",
        "style",
        CASE
            WHEN "ounces" = '12.0 oz' THEN '12.0 oz.'
            WHEN "ounces" = '12.0 ounce' THEN '12.0 oz.'
            WHEN "ounces" = '16.0 oz' THEN '16.0 oz.'
            WHEN "ounces" = '16.0 ounce' THEN '16.0 oz.'
            WHEN "ounces" = '12.0 OZ.' THEN '12.0 oz.'
            WHEN "ounces" = '12.0 oz. Alumi-Tek' THEN '12.0 oz.'
            WHEN "ounces" = '16.0 OZ.' THEN '16.0 oz.'
            WHEN "ounces" = '16.0 oz. Alumi-Tek' THEN '16.0 oz.'
            WHEN "ounces" = '12.0 oz. Silo Can' THEN '12.0 oz.'
            WHEN "ounces" = '24.0 ounce' THEN '24.0 oz.'
            WHEN "ounces" = '16.0 oz. Silo Can' THEN '16.0 oz.'
            WHEN "ounces" = '19.2 OZ.' THEN '19.2 oz.'
            WHEN "ounces" = '19.2 oz' THEN '19.2 oz.'
            WHEN "ounces" = '24.0 oz' THEN '24.0 oz.'
            ELSE "ounces"
        END AS "ounces",
        CASE
            WHEN "abv" = '0.045%' THEN '0.045'
            WHEN "abv" = '0.047%' THEN '0.047'
            WHEN "abv" = '0.048%' THEN '0.048'
            WHEN "abv" = '0.049%' THEN '0.049'
            WHEN "abv" = '0.05%' THEN '0.05'
            WHEN "abv" = '0.051%' THEN '0.051'
            WHEN "abv" = '0.052000000000000005%' THEN '0.052000000000000005'
            WHEN "abv" = '0.054000000000000006%' THEN '0.054000000000000006'
            WHEN "abv" = '0.055%' THEN '0.055'
            WHEN "abv" = '0.055999999999999994%' THEN '0.055999999999999994'
            WHEN "abv" = '0.057%' THEN '0.057'
            WHEN "abv" = '0.057999999999999996%' THEN '0.057999999999999996'
            WHEN "abv" = '0.06%' THEN '0.06'
            WHEN "abv" = '0.065%' THEN '0.065'
            WHEN "abv" = '0.068%' THEN '0.068'
            WHEN "abv" = '0.07%' THEN '0.07'
            WHEN "abv" = '0.08%' THEN '0.08'
            WHEN "abv" = '0.038%' THEN '0.038'
            WHEN "abv" = '0.039%' THEN '0.039'
            WHEN "abv" = '0.04%' THEN '0.04'
            WHEN "abv" = '0.040999999999999995%' THEN '0.040999999999999995'
            WHEN "abv" = '0.042%' THEN '0.042'
            WHEN "abv" = '0.043%' THEN '0.043'
            WHEN "abv" = '0.044000000000000004%' THEN '0.044000000000000004'
            WHEN "abv" = '0.046%' THEN '0.046'
            WHEN "abv" = '0.053%' THEN '0.053'
            WHEN "abv" = '0.059000000000000004%' THEN '0.059000000000000004'
            WHEN "abv" = '0.061%' THEN '0.061'
            WHEN "abv" = '0.062%' THEN '0.062'
            WHEN "abv" = '0.063%' THEN '0.063'
            WHEN "abv" = '0.064%' THEN '0.064'
            WHEN "abv" = '0.066%' THEN '0.066'
            WHEN "abv" = '0.067%' THEN '0.067'
            WHEN "abv" = '0.069%' THEN '0.069'
            WHEN "abv" = '0.071%' THEN '0.071'
            WHEN "abv" = '0.07200000000000001%' THEN '0.07200000000000001'
            WHEN "abv" = '0.073%' THEN '0.073'
            WHEN "abv" = '0.07400000000000001%' THEN '0.07400000000000001'
            WHEN "abv" = '0.075%' THEN '0.075'
            WHEN "abv" = '0.08199999999999999%' THEN '0.08199999999999999'
            WHEN "abv" = '0.09%' THEN '0.09'
            WHEN "abv" = '0.095%' THEN '0.095'
            WHEN "abv" = '0.099%' THEN '0.099'
            WHEN "abv" = '0.001%' THEN '0.001'
            WHEN "abv" = '0.027000000000000003%' THEN '0.027'
            WHEN "abv" = '0.035%' THEN '0.035'
            WHEN "abv" = '0.076%' THEN '0.076'
            WHEN "abv" = '0.077%' THEN '0.077'
            WHEN "abv" = '0.078%' THEN '0.078'
            WHEN "abv" = '0.079%' THEN '0.079'
            WHEN "abv" = '0.083%' THEN '0.083'
            WHEN "abv" = '0.085%' THEN '0.085'
            WHEN "abv" = '0.086%' THEN '0.086'
            WHEN "abv" = '0.087%' THEN '0.087'
            WHEN "abv" = '0.08900000000000001%' THEN '0.089'
            WHEN "abv" = '0.091%' THEN '0.091'
            WHEN "abv" = '0.092%' THEN '0.092'
            WHEN "abv" = '0.09300000000000001%' THEN '0.093'
            WHEN "abv" = '0.096%' THEN '0.096'
            WHEN "abv" = '0.09699999999999999%' THEN '0.097'
            WHEN "abv" = '0.10400000000000001%' THEN '0.104'
            WHEN "abv" = '0.125%' THEN '0.125'
            WHEN "abv" = '0.128%' THEN '0.128'
            ELSE "abv"
        END AS "abv",
        "ibu",
        "brewery_id",
        "brewery_name",
        "city",
        "state"
    FROM "beers"
),

"beers_cleaned_casted" AS (
    -- Column Type Casting: 
    -- abv: from VARCHAR to DECIMAL
    -- ibu: from DECIMAL to INT
    -- ounces: from VARCHAR to DECIMAL
    SELECT
        "id",
        "beer_name",
        "style",
        "brewery_id",
        "brewery_name",
        "city",
        "state",
        CAST("abv" AS DECIMAL) 
        AS "abv",
        CAST("ibu" AS INT) 
        AS "ibu",
        CAST(REGEXP_EXTRACT("ounces", '(\d+(\.\d+)?)') AS DECIMAL) 
        AS "ounces"
    FROM "beers_cleaned"
)

-- COCOON BLOCK END
SELECT *
FROM "beers_cleaned_casted"