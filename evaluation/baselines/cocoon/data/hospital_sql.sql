-- COCOON BLOCK START: PLEASE DO NOT MODIFY THIS BLOCK FOR SELF-MAINTENANCE
-- Generated at 2025-12-03 15:47:40.496782+00:00
WITH 
"hospital_cleaned" AS (
    -- Clean unusual string values: 
    -- Address1: The Address1 column suffers from three main issues: 1.  **Inconsistent Abbreviations:** Common street suffixes and directional words are abbreviated inconsistently. For example, 'st' is used alongside 'street', 'rd' with 'road', 'dr' with 'drive', 'ave' with 'avenue', 'blvd' with 'boulevard', 'med' with 'medical', and 'n'/'s' with 'north'/'south'. 2.  **Typographical Errors:** Several values contain clear typos, such as 'fixst stxeet noxth' instead of 'first street north', 'mxmorial' for 'memorial', 'sxuth' for 'south', and 'gilxreath' for 'gilbreath'. 3.  **Incorrect Separators:** A few entries use 'x' as a word separator instead of a space, like in '124xsxmemorialxdr'.The solution involves standardizing all abbreviations to their full, unabbreviated forms for uniformity, correcting all spelling mistakes, and replacing incorrect separators with spaces. This ensures the addresses are consistent and easier to parse.
    -- State: The problem is that the 'State' column contains values 'xl' and 'ax', which are not valid two-letter US state abbreviations. These appear to be typos of the valid and frequent values also present in the column, 'al' (Alabama) and 'ak' (Alaska). 'xl' likely corresponds to 'al', sharing the letter 'l'. 'ax' likely corresponds to 'ak', sharing the initial 'a'. The correction maps these invalid codes to their most plausible valid counterparts.
    -- HospitalType: The `HospitalType` column contains numerous misspellings and typos for the value 'acute care hospitals'. These variations appear to be data entry errors, frequently involving the letter 'x' being substituted for other characters or spaces (e.g., 'acuxe' for 'acute', 'hospixals' for 'hospitals', 'acutexcarexhospitals' for 'acute care hospitals'). The solution is to standardize all these variations by mapping them to the most frequent and correct form, 'acute care hospitals'.
    -- HospitalOwner: The problem is that the HospitalOwner column contains numerous typographical errors. A common pattern is the substitution of the letter 'x' for other characters (e.g., 'prxprietary' instead of 'proprietary', 'gxvernment' instead of 'government'). In some cases, 'x' is also used as a separator instead of spaces or hyphens. These typos create many inconsistent variations of the standard ownership categories. The correct values are the high-frequency, well-formed categories like 'proprietary', 'government - federal', and 'voluntary non-profit - private'. The solution is to map all the misspelled variations to their corresponding correct forms.
    -- EmergencyService: The column `EmergencyService` appears to be a binary field indicating 'yes' or 'no'. However, it contains several typographical errors. The values 'yxs', 'yex', and 'xes' are all slight variations of 'yes', likely due to keyboard mistypes where keys adjacent to 's' were pressed. Similarly, 'xo' is a likely typo for 'no'. The correction maps these unusual values to their intended standard forms, 'yes' and 'no', to ensure data consistency.
    -- Condition: The 'Condition' column contains several misspellings of standard medical conditions. The primary issue is the frequent, seemingly random substitution of the letter 'x' for other characters in words (e.g., 'heart attaxk' for 'heart attack', 'xneumonia' for 'pneumonia'). This creates multiple inconsistent representations for the same condition. The correct values are the most frequent and correctly spelled versions: 'surgical infection prevention', 'heart attack', 'pneumonia', 'heart failure', and 'children s asthma care'. The task is to consolidate all misspelled variants into these standard forms.
    -- MeasureName: The `MeasureName` column contains several types of data quality issues. 1.  **Formatting Errors**: Some values contain double spaces instead of single spaces (e.g., "treatment  at", "the  right"). 2.  **Typographical Errors**: Many values have systematic typos where correct letters are replaced with the letter 'x' (e.g., "surgerx" for "surgery", "patxents" for "patients"). 3.  **Delimiter Errors**: Some entries use the letter 'x' as a word separator instead of a space (e.g., "allxheartxsurgeryxpatients...").The solution is to standardize these values by mapping them to their correct, most frequent counterparts. This involves replacing double spaces with single spaces, correcting the 'x' typos, and replacing 'x' separators with spaces.
    -- Score: The 'Score' column contains several unusual values that are not valid numerical percentages. The value 'empty' is a string placeholder used to represent missing data. Additionally, there are several corrupted or masked values containing the character 'x' (e.g., '1xx%', 'x5%', '9x%', 'xx%'). These values are ambiguous and cannot be interpreted as a specific score. The fix is to standardize the representation of missing or invalid data by mapping both the 'empty' placeholder and all corrupted values containing 'x' to an empty string. All other valid percentage strings are kept as they are.
    -- Sample: 'empty' and '0 patients' are inconsistent representations for the same concept of zero.
    SELECT
        "ProviderNumber",
        "HospitalName",
        CASE
            WHEN "Address1" = '101 sivley rd' THEN '101 sivley road'
            WHEN "Address1" = '301 east 18th st' THEN '301 east 18th street'
            WHEN "Address1" = '200 med center drive' THEN '200 medical center drive'
            WHEN "Address1" = '400 northwood dr' THEN '400 northwood drive'
            WHEN "Address1" = '4800 48th st' THEN '4800 48th street'
            WHEN "Address1" = '1720 university blvd' THEN '1720 university boulevard'
            WHEN "Address1" = '400 n edwards street' THEN '400 north edwards street'
            WHEN "Address1" = '126 hospital ave' THEN '126 hospital avenue'
            WHEN "Address1" = '702 n main st' THEN '702 north main street'
            WHEN "Address1" = '515 miranda st' THEN '515 miranda street'
            WHEN "Address1" = '124 s memorial dr' THEN '124 south memorial drive'
            WHEN "Address1" = '315 w hickory st' THEN '315 west hickory street'
            WHEN "Address1" = '810 st vincents drive' THEN '810 saint vincents drive'
            WHEN "Address1" = '1000 fixst stxeet noxth' THEN '1000 first street north'
            WHEN "Address1" = '1201 7th street se' THEN '1201 7th street southeast'
            WHEN "Address1" = '124 s mxmorial dr' THEN '124 south memorial drive'
            WHEN "Address1" = '124xsxmemorialxdr' THEN '124 south memorial drive'
            WHEN "Address1" = '1256 military street sxuth' THEN '1256 military street south'
            WHEN "Address1" = '126xhospitalxave' THEN '126 hospital avenue'
            WHEN "Address1" = '150 gilxreath drive' THEN '150 gilbreath drive'
            WHEN "Address1" = '1530xuxsxhighwayx43' THEN '1530 us highway 43'
            WHEN "Address1" = '1720 univxrsity blvd' THEN '1720 university blvd'
            WHEN "Address1" = '1912xalabamaxhighwayx157' THEN '1912 alabama highway 157'
            WHEN "Address1" = '1x6 hospital ave' THEN '106 hospital ave'
            WHEN "Address1" = '201 pine sxreex norxhwesx' THEN '201 pine street northwest'
            WHEN "Address1" = '209 xorth maix street' THEN '209 north main street'
            WHEN "Address1" = '2505xuxsxhighwayx431xnorth' THEN '2505 us highway 431 north'
            WHEN "Address1" = '301 east 1xth st' THEN '301 east 10th street'
            WHEN "Address1" = '301xeastx18thxst' THEN '301 east 18th street'
            WHEN "Address1" = '315 w hickxry st' THEN '315 w hickory street'
            WHEN "Address1" = '315xwxhickoryxst' THEN '315 w hickory street'
            WHEN "Address1" = '33700 highway x3' THEN '33700 highway 43'
            WHEN "Address1" = '33700 hxghway 43' THEN '33700 highway 43'
            WHEN "Address1" = '400xnxedwardsxstreet' THEN '400 n edwards street'
            WHEN "Address1" = '4370xwestxmainxstreet' THEN '4370 west main street'
            WHEN "Address1" = '515 miranda sx' THEN '515 miranda street'
            WHEN "Address1" = '515 miranxa st' THEN '515 miranda street'
            WHEN "Address1" = '515 mirxndx st' THEN '515 miranda street'
            WHEN "Address1" = '600 souxh xhird sxreex' THEN '600 south third street'
            WHEN "Address1" = '702 x maix st' THEN '702 n main street'
            WHEN "Address1" = '702xnxmainxst' THEN '702 n main street'
            WHEN "Address1" = '8000 alabama xigxway 69' THEN '8000 alabama highway 69'
            WHEN "Address1" = '810 st vxncents drxve' THEN '810 st vincents drive'
            WHEN "Address1" = 'poxboxx287' THEN 'po box 287'
            WHEN "Address1" = 'x0x0 lay dam road' THEN '1010 lay dam road'
            ELSE "Address1"
        END AS "Address1",
        "Address2",
        "City",
        CASE
            WHEN "State" = 'xl' THEN 'al'
            WHEN "State" = 'ax' THEN 'ak'
            ELSE "State"
        END AS "State",
        "ZipCode",
        "CountyName",
        "PhoneNumber",
        CASE
            WHEN "HospitalType" = 'acuxe care hospixals' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acutexcarexhospitals' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acute care hosxitals' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acute care hospitaxs' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acute caxe hospitals' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acutx carx hospitals' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acute care hoxpitalx' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acute care hxspitals' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acute cxre hospitxls' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acute xare hospitals' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'acxte care hospitals' THEN 'acute care hospitals'
            WHEN "HospitalType" = 'axute care hospitals' THEN 'acute care hospitals'
            ELSE "HospitalType"
        END AS "HospitalType",
        CASE
            WHEN "HospitalOwner" = 'proxrietary' THEN 'proprietary'
            WHEN "HospitalOwner" = 'prxprietary' THEN 'proprietary'
            WHEN "HospitalOwner" = 'gxvernment - hxspital district xr authxrity' THEN 'government - hospital district or authority'
            WHEN "HospitalOwner" = 'pxopxietaxy' THEN 'proprietary'
            WHEN "HospitalOwner" = 'voluntaxy non-pxofit - pxivate' THEN 'voluntary non-profit - private'
            WHEN "HospitalOwner" = 'volunxary non-profix - privaxe' THEN 'voluntary non-profit - private'
            WHEN "HospitalOwner" = 'government - hospital xistrict or authority' THEN 'government - hospital district or authority'
            WHEN "HospitalOwner" = 'government - hospxtal dxstrxct or authorxty' THEN 'government - hospital district or authority'
            WHEN "HospitalOwner" = 'governmenx - hospixal disxricx or auxhorixy' THEN 'government - hospital district or authority'
            WHEN "HospitalOwner" = 'governxent - federal' THEN 'government - federal'
            WHEN "HospitalOwner" = 'goverxmext - hospital district or authority' THEN 'government - hospital district or authority'
            WHEN "HospitalOwner" = 'voluntary non-profit - otxer' THEN 'voluntary non-profit - other'
            WHEN "HospitalOwner" = 'voluntary nonxprofit x private' THEN 'voluntary non-profit - private'
            WHEN "HospitalOwner" = 'voluntaryxnon-profitx-xchurch' THEN 'voluntary non-profit - church'
            WHEN "HospitalOwner" = 'voluntaryxnon-profitx-xprivate' THEN 'voluntary non-profit - private'
            WHEN "HospitalOwner" = 'voluxtary xox-profit - private' THEN 'voluntary non-profit - private'
            WHEN "HospitalOwner" = 'volxntary non-profit - chxrch' THEN 'voluntary non-profit - church'
            WHEN "HospitalOwner" = 'voxuntary non-profit - church' THEN 'voluntary non-profit - church'
            WHEN "HospitalOwner" = 'xroprietary' THEN 'proprietary'
            ELSE "HospitalOwner"
        END AS "HospitalOwner",
        CASE
            WHEN "EmergencyService" = 'yxs' THEN 'yes'
            WHEN "EmergencyService" = 'yex' THEN 'yes'
            WHEN "EmergencyService" = 'xes' THEN 'yes'
            WHEN "EmergencyService" = 'xo' THEN 'no'
            ELSE "EmergencyService"
        END AS "EmergencyService",
        CASE
            WHEN "Condition" = 'heart attaxk' THEN 'heart attack'
            WHEN "Condition" = 'hexrt attxck' THEN 'heart attack'
            WHEN "Condition" = 'hearx axxack' THEN 'heart attack'
            WHEN "Condition" = 'hxart failurx' THEN 'heart failure'
            WHEN "Condition" = 'surgical ixfectiox prevextiox' THEN 'surgical infection prevention'
            WHEN "Condition" = 'suxgical infection pxevention' THEN 'surgical infection prevention'
            WHEN "Condition" = 'xneumonia' THEN 'pneumonia'
            WHEN "Condition" = 'heart faixure' THEN 'heart failure'
            WHEN "Condition" = 'heart faxlure' THEN 'heart failure'
            WHEN "Condition" = 'heartxattack' THEN 'heart attack'
            WHEN "Condition" = 'heaxt attack' THEN 'heart attack'
            WHEN "Condition" = 'heaxt failuxe' THEN 'heart failure'
            WHEN "Condition" = 'hexrt fxilure' THEN 'heart failure'
            WHEN "Condition" = 'pneumonix' THEN 'pneumonia'
            WHEN "Condition" = 'pneumonxa' THEN 'pneumonia'
            WHEN "Condition" = 'pnexmonia' THEN 'pneumonia'
            WHEN "Condition" = 'pnxumonia' THEN 'pneumonia'
            WHEN "Condition" = 'pxeumoxia' THEN 'pneumonia'
            WHEN "Condition" = 'surgical infection xrevention' THEN 'surgical infection prevention'
            WHEN "Condition" = 'surgical infxction prxvxntion' THEN 'surgical infection prevention'
            WHEN "Condition" = 'surgical xnfection prevention' THEN 'surgical infection prevention'
            WHEN "Condition" = 'surgxcal infectxon preventxon' THEN 'surgical infection prevention'
            WHEN "Condition" = 'xeart failure' THEN 'heart failure'
            ELSE "Condition"
        END AS "Condition",
        "MeasureCode",
        CASE
            WHEN "MeasureName" = 'patients who got treatment  at the right time (within 24 hours before or after their surgery) to help prevent blood clots after certain types of surgery' THEN 'patients who got treatment at the right time (within 24 hours before or after their surgery) to help prevent blood clots after certain types of surgery'
            WHEN "MeasureName" = 'surgery patients who were given the  right kind  of antibiotic to help prevent infection' THEN 'surgery patients who were given the right kind of antibiotic to help prevent infection'
            WHEN "MeasureName" = 'all heart surgerx patients whose blood sugar (blood glucose) is kept under good control in the daxs right after surgerx' THEN 'all heart surgery patients whose blood sugar (blood glucose) is kept under good control in the days right after surgery'
            WHEN "MeasureName" = 'all heaxt suxgexy patients whose blood sugax (blood glucose) is kept undex good contxol in the days xight aftex suxgexy' THEN 'all heart surgery patients whose blood sugar (blood glucose) is kept under good control in the days right after surgery'
            WHEN "MeasureName" = 'allxheartxsurgeryxpatientsxwhosexbloodxsugarx(bloodxglucose)xisxkeptxunderxgoodxcontrolxinxthexdaysxrightxafterxsurgery' THEN 'all heart surgery patients whose blood sugar (blood glucose) is kept under good control in the days right after surgery'
            WHEN "MeasureName" = 'childrenxwhoxreceivedxrelieverxmedicationxwhilexhospitalizedxforxasthma' THEN 'children who received reliever medication while hospitalized for asthma'
            WHEN "MeasureName" = 'heart attack patients given aspirin at arrivax' THEN 'heart attack patients given aspirin at arrival'
            WHEN "MeasureName" = 'heart attack patxents gxven aspxrxn at arrxval' THEN 'heart attack patients given aspirin at arrival'
            WHEN "MeasureName" = 'heart attaxk patients given aspirin at disxharge' THEN 'heart attack patients given aspirin at discharge'
            WHEN "MeasureName" = 'heart failure patients given ace inhibitor or arb for left xentricular systolic dysfunction (lxsd)' THEN 'heart failure patients given ace inhibitor or arb for left ventricular systolic dysfunction (lvsd)'
            WHEN "MeasureName" = 'heart faxlure patxents gxven ace inhxbxtor or arb for left ventrxcular systolxc dysfunctxon (lvsd)' THEN 'heart failure patients given ace inhibitor or arb for left ventricular systolic dysfunction (lvsd)'
            WHEN "MeasureName" = 'heart faxlure patxents gxven smokxng cessatxon advxce/counselxng' THEN 'heart failure patients given smoking cessation advice/counseling'
            WHEN "MeasureName" = 'heart xailure patients given smoking cessation advice/counseling' THEN 'heart failure patients given smoking cessation advice/counseling'
            WHEN "MeasureName" = 'heartxattackxpatientsxgivenxaspirinxatxarrival' THEN 'heart attack patients given aspirin at arrival'
            WHEN "MeasureName" = 'heartxattackxpatientsxgivenxpcixwithinx90xminutesxofxarrival' THEN 'heart attack patients given pci within 90 minutes of arrival'
            WHEN "MeasureName" = 'hearx failure paxienxs given smoking cessaxion advice/counseling' THEN 'heart failure patients given smoking cessation advice/counseling'
            WHEN "MeasureName" = 'heaxt attack patients given aspixin at dischaxge' THEN 'heart attack patients given aspirin at discharge'
            WHEN "MeasureName" = 'heaxt attack patients given beta blockex at dischaxge' THEN 'heart attack patients given beta blocker at discharge'
            WHEN "MeasureName" = 'heaxt failuxe patients given an evaluation of left ventxiculax systolic (lvs) function' THEN 'heart failure patients given an evaluation of left ventricular systolic (lvs) function'
            WHEN "MeasureName" = 'heaxt failuxe patients given dischaxge instxuctions' THEN 'heart failure patients given discharge instructions'
            WHEN "MeasureName" = 'hexrt attxck pxtients given aspirin xt arrivxl' THEN 'heart attack patients given aspirin at arrival'
            WHEN "MeasureName" = 'hxart attack patixnts givxn ace inhibitor or arb for lxft vxntricular systolic dysfunction (lvsd)' THEN 'heart attack patients given ace inhibitor or arb for left ventricular systolic dysfunction (lvsd)'
            WHEN "MeasureName" = 'pneumonia patients assessed and given influenza vaxxination' THEN 'pneumonia patients assessed and given influenza vaccination'
            WHEN "MeasureName" = 'pneumoniaxpatientsxassessedxandxgivenxinfluenzaxvaccination' THEN 'pneumonia patients assessed and given influenza vaccination'
            WHEN "MeasureName" = 'pneumonix pxtients assessed xnd given influenzx vxccinxtion' THEN 'pneumonia patients assessed and given influenza vaccination'
            WHEN "MeasureName" = 'pneumonix pxtients given initixl antibiotic(s) within 6 hours after arrivxl' THEN 'pneumonia patients given initial antibiotic(s) within 6 hours after arrival'
            WHEN "MeasureName" = 'pneumonxa patxents whose inxtxal emergency room blood culture was performed prxor to the admxnxstratxon of the fxrst hospxtal dose of antxbxotxcs' THEN 'pneumonia patients whose initial emergency room blood culture was performed prior to the administration of the first hospital dose of antibiotics'
            WHEN "MeasureName" = 'pnxumonia patixnts whosx initial emxrgxncy room blood culturx was pxrformxd prior to thx administration of thx first hospital dosx of antibiotics' THEN 'pneumonia patients whose initial emergency room blood culture was performed prior to the administration of the first hospital dose of antibiotics'
            WHEN "MeasureName" = 'pxeumoxia patiexts assessed axd givex ixfluexza vaccixatiox' THEN 'pneumonia patients assessed and given influenza vaccination'
            WHEN "MeasureName" = 'pxeumoxia patiexts givex ixitial axtibiotic(s) withix 6 hours after arrival' THEN 'pneumonia patients given initial antibiotic(s) within 6 hours after arrival'
            WHEN "MeasureName" = 'pxeumoxia patiexts givex the most appropriate ixitial axtibiotic(s)' THEN 'pneumonia patients given the most appropriate initial antibiotic(s)'
            WHEN "MeasureName" = 'surgery patients who were taking heart drugs caxxed beta bxockers before coming to the hospitax who were kept on the beta bxockers during the period just before and after their surgery' THEN 'surgery patients who were taking heart drugs called beta blockers before coming to the hospital who were kept on the beta blockers during the period just before and after their surgery'
            WHEN "MeasureName" = 'surgery patiexts who were takixg heart drugs called beta blockers before comixg to the hospital who were kept ox the beta blockers durixg the period just before axd after their surgery' THEN 'surgery patients who were taking heart drugs called beta blockers before coming to the hospital who were kept on the beta blockers during the period just before and after their surgery'
            WHEN "MeasureName" = 'surgery paxienxs needing hair removed from xhe surgical area before surgery who had hair removed using a safer mexhod (elecxric clippers or hair removal cream c nox a razor)' THEN 'surgery patients needing hair removed from the surgical area before surgery who had hair removed using a safer method (electric clippers or hair removal cream - not a razor)'
            WHEN "MeasureName" = 'surgery pxtients who were txking hexrt drugs cxlled betx blockers before coming to the hospitxl who were kept on the betx blockers during the period just before xnd xfter their surgery' THEN 'surgery patients who were taking heart drugs called beta blockers before coming to the hospital who were kept on the beta blockers during the period just before and after their surgery'
            WHEN "MeasureName" = 'surgery pxtients whose doctors ordered trextments to prevent blood clots xfter certxin types of surgeries' THEN 'surgery patients whose doctors ordered treatments to prevent blood clots after certain types of surgeries'
            WHEN "MeasureName" = 'surgeryxpatientsxneedingxhairxremovedxfromxthexsurgicalxareaxbeforexsurgery& xwhoxhadxhairxremovedxusingxaxsaferxmethodx(electricxclippersxorxhairxremovalxcreamxï¿½cxnotxaxrazor)' THEN 'surgery patients needing hair removed from the surgical area before surgery who had hair removed using a safer method (electric clippers or hair removal cream - not a razor)'
            WHEN "MeasureName" = 'xeart attack patients given aspirin at arrival' THEN 'heart attack patients given aspirin at arrival'
            ELSE "MeasureName"
        END AS "MeasureName",
        CASE
            WHEN "Score" = '1xx%' THEN NULL
            WHEN "Score" = '9x%' THEN NULL
            WHEN "Score" = 'empty' THEN NULL
            WHEN "Score" = 'x00%' THEN NULL
            WHEN "Score" = 'x5%' THEN NULL
            WHEN "Score" = 'x7%' THEN NULL
            WHEN "Score" = 'xx%' THEN NULL
            WHEN "Score" = '80x' THEN '80%'
            WHEN "Score" = '89x' THEN '89%'
            WHEN "Score" = '93x' THEN '93%'
            WHEN "Score" = '95x' THEN '95%'
            WHEN "Score" = 'x3%' THEN NULL
            WHEN "Score" = 'x4%' THEN NULL
            WHEN "Score" = 'x6%' THEN NULL
            ELSE "Score"
        END AS "Score",
        "Sample",
        "Stateavg"
    FROM "hospital"
),

"hospital_cleaned_null" AS (
    -- NULL Imputation: Impute Null to Disguised Missing Values
    -- Address2: ['empty']
    -- Sample: ['empty']
    SELECT 
        CASE
            WHEN "Address2" = 'empty' THEN NULL
            ELSE "Address2"
        END AS "Address2",
        CASE
            WHEN "Sample" = 'empty' THEN NULL
            ELSE "Sample"
        END AS "Sample",
        "CountyName",
        "State",
        "Address1",
        "HospitalType",
        "EmergencyService",
        "Stateavg",
        "ProviderNumber",
        "PhoneNumber",
        "MeasureCode",
        "HospitalName",
        "Condition",
        "HospitalOwner",
        "Score",
        "MeasureName",
        "ZipCode",
        "City"
    FROM "hospital_cleaned"
),

"hospital_cleaned_null_casted" AS (
    -- Column Type Casting: 
    -- EmergencyService: from VARCHAR to BOOLEAN
    -- Sample: from VARCHAR to INT
    -- Score: from VARCHAR to DECIMAL
    SELECT
        "Address2",
        "CountyName",
        "State",
        "Address1",
        "HospitalType",
        "Stateavg",
        "ProviderNumber",
        "PhoneNumber",
        "MeasureCode",
        "HospitalName",
        "Condition",
        "HospitalOwner",
        "MeasureName",
        "ZipCode",
        "City",
        CAST("EmergencyService" AS BOOLEAN) 
        AS "EmergencyService",
        TRY_CAST(REGEXP_EXTRACT("Sample", '(\d+)') AS INT) 
        AS "Sample",
        CAST(REGEXP_EXTRACT("Score", '(\d+)') AS DECIMAL) 
        AS "Score"
    FROM "hospital_cleaned_null"
)

-- COCOON BLOCK END
SELECT *
FROM "hospital_cleaned_null_casted"