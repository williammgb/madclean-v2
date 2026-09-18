-- COCOON BLOCK START: PLEASE DO NOT MODIFY THIS BLOCK FOR SELF-MAINTENANCE
-- Generated at 2025-12-02 14:47:40.831052+00:00
WITH 
"rayyan_trimmed" AS (
    -- Trim Leading and Trailing Spaces
    SELECT
        "id",
        "article_language",
        "journal_abbreviation",
        "journal_issn",
        "article_jvolumn",
        "article_jissue",
        "article_jcreated_at",
        "article_pagination",
        "author_list",
        TRIM("article_title") AS "article_title",
        TRIM("journal_title") AS "journal_title"
    FROM "rayyan"
),

"rayyan_trimmed_cleaned" AS (
    -- Clean unusual string values: 
    -- article_language: The `article_language` column has several issues.  First, there is inconsistency in representing the same language, using different casings, full names, and abbreviation standards (e.g., 'eng', 'ENG', 'English', 'en' for English; 'ger', 'German' for German).  Second, some values are non-standard composites, combining multiple languages without a clear separator (e.g., 'ENGPOR', 'engfre').  Third, some values contain extra, irrelevant text or inconsistent spacing (e.g., 'English; ABSTRACT LANGUAGE:English', 'Chinese     Chinese, English').The solution is to standardize all values to their lowercase, 3-letter ISO 639-2 codes. For composite entries, the languages are extracted and represented as a semicolon-separated list of these standard codes.
    -- journal_abbreviation: Values mix full titles ('The American journal...') and inconsistent abbreviations ('Eur J Anaesthesiol' vs. 'Eur. Respir. J.').
    -- journal_issn: Values 'xxxx-xxxx', 'Feb-65', and '0949-2658 (Print) 0949-2658' are unusual as they are placeholders, dates, or contain extra text.
    -- article_pagination: '7-Jan' is a date, not a page number. Ranges like '185-90' use an inconsistent shorthand format.
    -- author_list: '{NULL}' is a placeholder. 'K��lendorf' has a character encoding error. Initials are inconsistent ('T. Sone' vs 'A Green').
    -- article_title: '...�_-aminobutyric...', '<title>...', '...hyperplasia].' contain encoding errors, HTML tags, and unmatched brackets respectively.
    -- journal_title: 'CANCER' and 'Cancer' are redundant. Inconsistent capitalization exists across titles like 'annals of internal medicine' (lowercase).
    SELECT
        "id",
        CASE
            WHEN "article_language" = 'ENG' THEN 'eng'
            WHEN "article_language" = 'ENGPOR' THEN 'eng; por'
            WHEN "article_language" = 'Eng' THEN 'eng'
            WHEN "article_language" = 'English' THEN 'eng'
            WHEN "article_language" = 'English; ABSTRACT LANGUAGE:English' THEN 'eng'
            WHEN "article_language" = 'English     Italian' THEN 'eng; ita'
            WHEN "article_language" = 'French' THEN 'fre'
            WHEN "article_language" = 'German' THEN 'ger'
            WHEN "article_language" = 'Korean' THEN 'kor'
            WHEN "article_language" = 'Chinese     Chinese, English' THEN 'chi; eng'
            WHEN "article_language" = 'Chinese     English, Chinese' THEN 'chi; eng'
            WHEN "article_language" = 'en' THEN 'eng'
            WHEN "article_language" = 'eng     por' THEN 'eng; por'
            WHEN "article_language" = 'engfre' THEN 'eng; fre'
            WHEN "article_language" = 'engpor' THEN 'eng; por'
            WHEN "article_language" = 'engspa' THEN 'eng; spa'
            WHEN "article_language" = 'pt' THEN 'por'
            ELSE "article_language"
        END AS "article_language",
        "journal_abbreviation",
        "journal_issn",
        "article_jvolumn",
        "article_jissue",
        "article_jcreated_at",
        "article_pagination",
        "author_list",
        "article_title",
        "journal_title"
    FROM "rayyan_trimmed"
),

"rayyan_trimmed_cleaned_null" AS (
    -- NULL Imputation: Impute Null to Disguised Missing Values
    -- journal_issn: ['xxxx-xxxx']
    -- article_pagination: ['-']
    -- author_list: ['{NULL}']
    SELECT 
        CASE
            WHEN "journal_issn" = 'xxxx-xxxx' THEN NULL
            ELSE "journal_issn"
        END AS "journal_issn",
        CASE
            WHEN "article_pagination" = '-' THEN NULL
            ELSE "article_pagination"
        END AS "article_pagination",
        CASE
            WHEN "author_list" = '{NULL}' THEN NULL
            ELSE "author_list"
        END AS "author_list",
        "article_title",
        "article_language",
        "id",
        "journal_title",
        "article_jcreated_at",
        "article_jvolumn",
        "journal_abbreviation",
        "article_jissue"
    FROM "rayyan_trimmed_cleaned"
),

"rayyan_trimmed_cleaned_null_casted" AS (
    -- Column Type Casting: 
    -- article_jcreated_at: from VARCHAR to DATE
    -- article_jissue: from DECIMAL to INT
    -- article_jvolumn: from DECIMAL to INT
    -- author_list: from VARCHAR to ARRAY
    SELECT
        "journal_issn",
        "article_pagination",
        "article_title",
        "article_language",
        "id",
        "journal_title",
        "journal_abbreviation",
        CAST(strptime("article_jcreated_at", '%-m/%-d/%y') AS DATE) 
        AS "article_jcreated_at",
        CAST("article_jissue" AS INT) 
        AS "article_jissue",
        CAST("article_jvolumn" AS INT) 
        AS "article_jvolumn",
        "author_list" 
        AS "author_list"
    FROM "rayyan_trimmed_cleaned_null"
)

-- COCOON BLOCK END
SELECT *
FROM "rayyan_trimmed_cleaned_null_casted"