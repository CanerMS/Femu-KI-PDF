"""
Configuration Module
Centralized configuration for the PDF classifier
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent # Project root directory
DATA_DIR = BASE_DIR / "data" # Data directory
RESULTS_DIR = BASE_DIR / "results" # Results directory
LOGS_DIR = BASE_DIR / "logs" # Logs directory

# Data directories
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs" # Directory for raw PDFs
USEFUL_PDFS_DIR = DATA_DIR / "useful_pdfs" # Directory for useful PDFs
RAW_TXTS_DIR = DATA_DIR / "raw_texts" # Directory for raw texts
USEFUL_TXTS_DIR = DATA_DIR / "useful_texts" # Directory for useful texts
EXTRACTED_TEXTS_DIR = DATA_DIR / "extracted_texts" # Directory for extracted texts
LABELS_PATH = DATA_DIR / "labels.csv" # Path for labels CSV
PREPROCESSED_TEXTS_DIR = DATA_DIR / "preprocessed_texts" # Directory for preprocessed texts

# Model parameters
MAX_FEATURES = 2000 # Maximum number of features for vectorizer
NGRAM_RANGE = (1, 2) # N-gram range for text vectorization
TEST_SIZE = 0.25 # Proportion of data for test set
RANDOM_STATE = 42 # Random state for reproducibility
SMOTE_THRESHOLD =5  #  Imbalance ratio threshold, above which SMOTE will be applied

# Model hyperparameters
N_ESTIMATORS = 100 # Number of trees in Random Forest
MAX_DEPTH = 10 # Maximum depth of each tree
MIN_SAMPLES_SPLIT = 5 # Minimum samples required to split a node

analysis_results_dir = RESULTS_DIR / "feature_analysis" # Directory for feature analysis results

# Custom stop words
CUSTOM_STOP_WORDS = [ # List of custom stop words to exclude from text analysis
    # ════════════════════════════════════════════
    # GENERIC RESEARCH TERMS 
    # ════════════════════════════════════════════
    'study', 'studies', 'studied', 'result', 'results',
    'method', 'methods', 'data', 'analysis', 'conclusion',
    'background', 'objective', 'finding', 'findings',
    'significant', 'significantly', 'showed', 'shown',
    'used', 'using', 'based', 'compared', 'total',
    'mean', 'average', 'respectively', 'associated',
    'observed', 'measured', 'performed', 'obtained',
    'means',  'values', 'indicate', 'indicated',
    
    # ════════════════════════════════════════════
    # DEMOGRAPHICS 
    # ════════════════════════════════════════════
    'age', 'gender', 'male', 'female', 'group', 'groups',
    'control', 'controls', 'case', 'cases', 'patient', 'patients',
    
    # ════════════════════════════════════════════
    # MEDICAL MEASUREMENTS 
    # ════════════════════════════════════════════
    'score', 'scores', 'scan', 'scans', 'image', 'images',
    'test', 'tests', 'value', 'values', 'level', 'levels',
    
    # ════════════════════════════════════════════
    # MEDICAL PROCEDURES 
    # ════════════════════════════════════════════
    'diagnosis', 'diagnostic', 'procedure', 'procedures',
    'treatment', 'treatments', 'therapy', 'therapies',
    'examination', 'examined', 'assess', 'assessed', 'assessment',
    'monitoring', 'monitored', 'evaluation', 'evaluated',
    'surgery', 'surgical', 'operation', 'operated',
    'injection', 'injected', 'administration', 'administered',
    
    # ════════════════════════════════════════════
    # IMAGING & DETECTION 
    # ════════════════════════════════════════════
    'detection', 'detecting', 'detected', 'detector',
    'optical', 'visible', 'visualization', 'imaging',
    'superior', 'inferior', 'anterior', 'posterior',
    'sample', 'samples', 'sampling', 'specimen',
    'contrast',  
    'double',   
    
    # ════════════════════════════════════════════
    # STATISTICS & NUMBERS 
    # ════════════════════════════════════════════
    'number', 'numbers', 'percentage', 'percent',
    'ratio', 'ratios', 'rate', 'rates',
    'frequency', 'frequencies', 'prevalence',
    'incidence', 'occurrence', 'distribution',
    'range', 'ranges', 'variation', 'variance',
    'slightly',  # ← YENİ: belirsiz terim
    
    # ════════════════════════════════════════════
    # TIME & LOCATION 
    # ════════════════════════════════════════════
    'time', 'times', 'period', 'periods', 'duration',
    'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    'baseline', 'follow', 'followup', 'postoperative', 'preoperative',
    'hospital', 'clinic', 'center', 'department',
    
    # ════════════════════════════════════════════
    # RESEARCH PROCESS 
    # ════════════════════════════════════════════
    'research', 'investigate', 'investigated', 'investigation',
    'aim', 'aims', 'purpose', 'hypothesis',
    'design', 'designed', 'protocol', 'criteria',
    'retrospective', 'prospective', 'randomized', 'controlled',
    'cohort', 'trial', 'trials', 'experiment', 'experimental',
    'approved',  

    # ════════════════════════════════════════════
    # META WORDS 
    # ════════════════════════════════════════════
    'targeting',     # ← Generic term
    'algorithm', 'algorithms',
    'artificial', 'intelligence',
    'features', 'feature',
    'united', 'states',  # Geographic noise
    'longitudinal',
    
    # ════════════════════════════════════════════
    # JOURNAL/FIGURE NOISE TERMS
    # ════════════════════════════════════════════
    'fig', 'figs',   # Already has, but add variations
    'crossref', 'pubmed',  # Journal metadata
    'researcharticle', 'vol',
    'opticsexpress',
    'cid',  # Citation ID
    'ps',  'pdf',   # File format noise
    
    # ════════════════════════════════════════════
    # COMPARISON & OUTCOMES 
    # ════════════════════════════════════════════
    'comparison', 'compare', 'difference', 'differences',
    'similar', 'similarly', 'higher', 'lower',
    'increase', 'increased', 'decrease', 'decreased',
    'improvement', 'improved', 'outcome', 'outcomes',
    'effect', 'effects', 'efficacy', 'effective', 'effectiveness',
    
    # ════════════════════════════════════════════
    # QUALITY & VALIDITY 
    # ════════════════════════════════════════════
    'accuracy', 'accurate', 'precision', 'precise',
    'sensitivity', 'specificity', 'reliability', 'valid', 'validity',
    'quality', 'standard', 'standards', 'normal',
    
    # ════════════════════════════════════════════
    # JOURNAL METADATA 
    # ════════════════════════════════════════════
    'abstract', 'introduction', 'materials', 'discussion',
    'reference', 'references', 'figure', 'figures', 'table', 'tables',
    'published', 'publication', 'article', 'journal',
    'doi', 'manuscript', 'supplementary', 'appendix',
    
    # ════════════════════════════════════════════
    # GENERIC DESCRIPTORS 
    # ════════════════════════════════════════════
    'various', 'several', 'multiple', 'different',
    'important', 'potential', 'possible', 'likely',
    'general', 'specific', 'particular', 'common',
    'primary', 'secondary', 'initial', 'final',
    'major', 'minor', 'main', 'overall', 'processing',      # Generic processing term
    'shorter',         # Generic descriptor
    'disorder', 'disorders',  # Generic medical
    'enabling',        # Generic capability term
    'associations',    # Generic statistics
    'technology', 'technologies',  # Generic tech
    'la',              # Abbreviation noise (Los Angeles?)
    'respiratory',     # Generic medical term
    'processing', 'severe', 'stronger', 'radiology',
    'spectra', 'spectral', 'reconstruction', 'view',

    # ════════════════════════════════════════════
    # Countries 
    # ════════════════════════════════════════════
    'china', 'chinese', 'india', 'indian',
    'japan', 'japanese', 'germany', 'german',
    'france', 'french', 'brazil', 'brazilian',
    'canada', 'canadian', 'australia', 'australian',
    'uk', 'united kingdom', 'britain', 'british',
    'russia', 'russian', 'spain', 'spanish',
    'italy', 'italian', 'korea', 'korean',
    'south korea', 'north korea',
    'turkey', 'turkish',
    'mexico', 'mexican',
    'saudi', 'arabia', 'saudi arabia',
    'uae', 'united arab emirates',
    'singapore', 'singaporean',
    'sweden', 'swedish',
    'norway', 'norwegian',
    'denmark', 'danish',
    'finland', 'finnish',
    'netherlands', 'dutch',
    'belgium', 'belgian',
    'switzerland', 'swiss',
    'austria', 'austrian',
    'poland', 'polish',
    'greece', 'greek',
    'portugal', 'portuguese',
    'ireland', 'irish',
    'new zealand', 'zealandic',
    'egypt', 'egyptian',
    'south africa', 'african',
    'argentina', 'argentinian',
    'chile', 'chilean',
    'colombia', 'colombian',
    'venezuela', 'venezuelan',
    'peru', 'peruvian',
    'pakistan', 'pakistani',
    'bangladesh', 'bangladeshi',    
    'sri lanka', 'sri lankan',
    'nepal', 'nepali',
    'bhutan', 'bhutanese',
    'malaysia', 'malaysian',
    'indonesia', 'indonesian',
    'philippines', 'filipino',
    'vietnam', 'vietnamese',
    'thailand', 'thai',
    'myanmar', 'burmese',
    'cambodia', 'cambodian',
    'laos', 'laotian',
    'mongolia', 'mongolian',
    'iran', 'iranian',
    'iraq', 'iraqi',
    'afghanistan', 'afghan',
    'syria', 'syrian',
    'lebanon', 'lebanese',
    'jordan', 'jordanian',
    'yemen', 'yemeni',
    'oman', 'omani',
    'kuwait', 'kuwaiti',
    'qatar', 'qatari',
    'bahrain', 'bahraini',
    'haiti', 'haitian',
    'cuba', 'cuban',
    'jamaica', 'jamaican',
    'trinidad', 'tobago', 'trinidad and tobago',
    'dominican', 'republic', 'dominican republic',
    'guatemala', 'guatemalan',
    'honduras', 'honduran',
    'el salvador', 'salvadoran',
    'nicaragua', 'nicaraguan',
    'costa rica', 'costa rican',
    'panama', 'panamanian',
    'uruguay', 'uruguayan',
    'paraguay', 'paraguayan',
    'bolivia', 'bolivian',
    'ecuador', 'ecuadorian',
    'puerto rico', 'puerto rican',
    'sudan', 'sudanese',
    'ethiopia', 'ethiopian',
    'somalia', 'somali',
    'kenya', 'kenyan',
    'tanzania', 'tanzanian',
    'uganda', 'ugandan',
    'ghana', 'ghanian',
    'nigeria', 'nigerian',
    'cameroon', 'cameroonian',
    'ivory coast', 'cote d\'ivoire', 'ivoirian',
    'algeria', 'algerian',
    'morocco', 'moroccan',
    'tunisia', 'tunisian',
    'libya', 'libyan',
    'angola', 'angolan',
    'zambia', 'zambian',
    'zimbabwe', 'zimbabwean',
    'malawi', 'malawian',
    'mozambique', 'mozambican',
    'botswana', 'botswanan',
    'namibia', 'namibian',
    'lesotho', 'lesothan',
    'swaziland', 'eswatini', 'swazi',
    'madagascar', 'malagasy',
    'reunion', 'mauritius', 'mauritian',
    'seychelles', 'seychellois',

    # ════════════════════════════════════════════
    # Cities
    # ════════════════════════════════════════════
    'london', 'paris', 'berlin', 'madrid', 'rome',
    'vienna', 'prague', 'budapest', 'warsaw', 'aachen', 'ac',
    'munich', 'hamburg', 'frankfurt', 'cologne', 'duesseldorf',
    'stuttgart', 'dresden', 'leipzig', 'bremen', 'hannover',
    'brussels', 'amsterdam', 'rotterdam', 'the hague', 'dublin',
    'copenhagen', 'stockholm', 'oslo', 'helsinki', 'zagreb',
    'belgrade', 'sarajevo', 'ljubljana', 'skopje', 'podgorica',
    'tirana', 'pristina', 'vilnius', 'riga', 'tallinn',
    'athens', 'nicosia', 'valletta',
    'reykjavik',
    'istanbul', 'ankara', 'izmir',
    'tokyo', 'osaka', 'kyoto',
    'beijing', 'shanghai', 'guangzhou',
    'shenzhen', 'chengdu', 'wuhan',
    'mumbai', 'delhi', 'bangalore',
    'chennai', 'kolkata',
    'seoul', 'busan', 'incheon',
    'jakarta', 'surabaya', 'bandung',
    'manila', 'cebu',
    'ho chi minh city', 'hanoi',
    'bangkok', 'phuket', 'chiang mai', 
    'yangon', 'mandalay',
    'cairo', 'alexandria',

    # ════════════════════════════════════════════
    # Months
    # ════════════════════════════════════════════

    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',

    # ════════════════════════════════════════════
    # Days
    # ════════════════════════════════════════════
    
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'

    # ════════════════════════════════════════════
    # Add more as needed
    # ════════════════════════════════════════════

]

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, LOGS_DIR, RAW_PDFS_DIR,     # List of directories to create
                  USEFUL_PDFS_DIR, EXTRACTED_TEXTS_DIR, analysis_results_dir]: # Iterate through each directory path
    dir_path.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist