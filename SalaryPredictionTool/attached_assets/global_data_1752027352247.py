"""
Global data for countries, states, cities, job titles (2030 projections), and industries
Based on authentic data sources and 2030 employment projections
"""

# Global Countries with major states/provinces and cities
GLOBAL_LOCATIONS = {
    "United States": {
        "states": {
            "California": ["Los Angeles", "San Francisco", "San Diego", "Sacramento", "San Jose", "Fresno", "Long Beach", "Oakland", "Bakersfield", "Anaheim"],
            "Texas": ["Houston", "Dallas", "San Antonio", "Austin", "Fort Worth", "El Paso", "Arlington", "Corpus Christi", "Plano", "Lubbock"],
            "New York": ["New York City", "Buffalo", "Rochester", "Yonkers", "Syracuse", "Albany", "New Rochelle", "Mount Vernon", "Schenectady", "Utica"],
            "Florida": ["Jacksonville", "Miami", "Tampa", "Orlando", "St. Petersburg", "Hialeah", "Tallahassee", "Fort Lauderdale", "Port St. Lucie", "Cape Coral"],
            "Illinois": ["Chicago", "Aurora", "Peoria", "Rockford", "Joliet", "Naperville", "Springfield", "Elgin", "Waukegan", "Cicero"],
            "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Reading", "Scranton", "Bethlehem", "Lancaster", "Harrisburg", "Altoona"],
            "Ohio": ["Columbus", "Cleveland", "Cincinnati", "Toledo", "Akron", "Dayton", "Parma", "Canton", "Youngstown", "Lorain"],
            "Georgia": ["Atlanta", "Augusta", "Columbus", "Savannah", "Athens", "Sandy Springs", "Roswell", "Macon", "Johns Creek", "Albany"],
            "North Carolina": ["Charlotte", "Raleigh", "Greensboro", "Durham", "Winston-Salem", "Fayetteville", "Cary", "Wilmington", "High Point", "Concord"],
            "Michigan": ["Detroit", "Grand Rapids", "Warren", "Sterling Heights", "Lansing", "Ann Arbor", "Flint", "Dearborn", "Livonia", "Westland"]
        }
    },
    "Canada": {
        "provinces": {
            "Ontario": ["Toronto", "Ottawa", "Hamilton", "London", "Kitchener", "Windsor", "Oshawa", "Barrie", "Guelph", "Kingston"],
            "Quebec": ["Montreal", "Quebec City", "Laval", "Gatineau", "Longueuil", "Sherbrooke", "Saguenay", "Trois-Rivières", "Terrebonne", "Saint-Jean-sur-Richelieu"],
            "British Columbia": ["Vancouver", "Victoria", "Burnaby", "Richmond", "Abbotsford", "Coquitlam", "Surrey", "Langley", "Saanich", "Delta"],
            "Alberta": ["Calgary", "Edmonton", "Red Deer", "Lethbridge", "St. Albert", "Medicine Hat", "Grande Prairie", "Airdrie", "Spruce Grove", "Leduc"],
            "Manitoba": ["Winnipeg", "Brandon", "Steinbach", "Thompson", "Portage la Prairie", "Winkler", "Selkirk", "Morden", "Dauphin", "The Pas"],
            "Saskatchewan": ["Saskatoon", "Regina", "Prince Albert", "Moose Jaw", "Swift Current", "Yorkton", "North Battleford", "Estevan", "Weyburn", "Lloydminster"]
        }
    },
    "United Kingdom": {
        "countries": {
            "England": ["London", "Birmingham", "Manchester", "Leeds", "Liverpool", "Sheffield", "Bristol", "Newcastle", "Nottingham", "Leicester"],
            "Scotland": ["Edinburgh", "Glasgow", "Aberdeen", "Dundee", "Stirling", "Perth", "Inverness", "Paisley", "East Kilbride", "Hamilton"],
            "Wales": ["Cardiff", "Swansea", "Newport", "Wrexham", "Barry", "Caerphilly", "Bridgend", "Neath", "Port Talbot", "Cwmbran"],
            "Northern Ireland": ["Belfast", "Derry", "Lisburn", "Newtownabbey", "Bangor", "Craigavon", "Castlereagh", "Ballymena", "Newtownards", "Carrickfergus"]
        }
    },
    "Germany": {
        "states": {
            "Bavaria": ["Munich", "Nuremberg", "Augsburg", "Regensburg", "Würzburg", "Ingolstadt", "Fürth", "Erlangen", "Bamberg", "Bayreuth"],
            "North Rhine-Westphalia": ["Cologne", "Düsseldorf", "Dortmund", "Essen", "Duisburg", "Bochum", "Wuppertal", "Bielefeld", "Bonn", "Münster"],
            "Baden-Württemberg": ["Stuttgart", "Mannheim", "Karlsruhe", "Freiburg", "Heidelberg", "Heilbronn", "Ulm", "Pforzheim", "Reutlingen", "Esslingen"],
            "Lower Saxony": ["Hanover", "Braunschweig", "Oldenburg", "Osnabrück", "Wolfsburg", "Göttingen", "Salzgitter", "Hildesheim", "Delmenhorst", "Wilhelmshaven"],
            "Hesse": ["Frankfurt", "Wiesbaden", "Kassel", "Darmstadt", "Offenbach", "Hanau", "Marburg", "Gießen", "Fulda", "Rüsselsheim"],
            "Berlin": ["Berlin"]
        }
    },
    "France": {
        "regions": {
            "Île-de-France": ["Paris", "Boulogne-Billancourt", "Saint-Denis", "Argenteuil", "Montreuil", "Créteil", "Nanterre", "Courbevoie", "Versailles", "Rueil-Malmaison"],
            "Auvergne-Rhône-Alpes": ["Lyon", "Grenoble", "Saint-Étienne", "Villeurbanne", "Clermont-Ferrand", "Valence", "Chambéry", "Annecy", "Bourg-en-Bresse", "Roanne"],
            "Nouvelle-Aquitaine": ["Bordeaux", "Limoges", "Poitiers", "Pau", "La Rochelle", "Mérignac", "Pessac", "Bayonne", "Angoulême", "Niort"],
            "Occitanie": ["Toulouse", "Montpellier", "Nîmes", "Perpignan", "Béziers", "Narbonne", "Carcassonne", "Albi", "Tarbes", "Castres"],
            "Hauts-de-France": ["Lille", "Amiens", "Tourcoing", "Roubaix", "Dunkerque", "Calais", "Villeneuve-d'Ascq", "Saint-Quentin", "Beauvais", "Compiègne"],
            "Grand Est": ["Strasbourg", "Reims", "Metz", "Nancy", "Mulhouse", "Troyes", "Châlons-en-Champagne", "Colmar", "Épinal", "Thionville"]
        }
    },
    "Japan": {
        "prefectures": {
            "Tokyo": ["Tokyo", "Hachioji", "Tachikawa", "Musashino", "Mitaka", "Ome", "Fuchu", "Akishima", "Chofu", "Machida"],
            "Osaka": ["Osaka", "Sakai", "Higashiosaka", "Hirakata", "Toyonaka", "Suita", "Takatsuki", "Yao", "Ibaraki", "Neyagawa"],
            "Kanagawa": ["Yokohama", "Kawasaki", "Sagamihara", "Fujisawa", "Chigasaki", "Hiratsuka", "Odawara", "Yamato", "Zushi", "Kamakura"],
            "Aichi": ["Nagoya", "Toyota", "Okazaki", "Ichinomiya", "Seto", "Kasugai", "Inuyama", "Konan", "Komaki", "Inazawa"],
            "Hokkaido": ["Sapporo", "Asahikawa", "Hakodate", "Kushiro", "Obihiro", "Kitami", "Iwamizawa", "Abashiri", "Rumoi", "Nemuro"],
            "Kyoto": ["Kyoto", "Uji", "Kameoka", "Joyo", "Muko", "Nagaokakyo", "Yawata", "Kyotanabe", "Kyotango", "Nantan"]
        }
    },
    "China": {
        "provinces": {
            "Guangdong": ["Guangzhou", "Shenzhen", "Dongguan", "Foshan", "Zhongshan", "Zhuhai", "Jiangmen", "Huizhou", "Zhaoqing", "Shantou"],
            "Jiangsu": ["Nanjing", "Suzhou", "Wuxi", "Xuzhou", "Changzhou", "Nantong", "Lianyungang", "Huai'an", "Yancheng", "Yangzhou"],
            "Shandong": ["Jinan", "Qingdao", "Zibo", "Zaozhuang", "Dongying", "Yantai", "Weifang", "Jining", "Tai'an", "Weihai"],
            "Zhejiang": ["Hangzhou", "Ningbo", "Wenzhou", "Jiaxing", "Huzhou", "Shaoxing", "Jinhua", "Quzhou", "Zhoushan", "Taizhou"],
            "Henan": ["Zhengzhou", "Kaifeng", "Luoyang", "Pingdingshan", "Anyang", "Hebi", "Xinxiang", "Jiaozuo", "Puyang", "Xuchang"],
            "Sichuan": ["Chengdu", "Zigong", "Panzhihua", "Luzhou", "Deyang", "Mianyang", "Guangyuan", "Suining", "Neijiang", "Leshan"]
        }
    },
    "India": {
        "states": {
            "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Solapur", "Kolhapur", "Amravati", "Sangli", "Malegaon"],
            "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum", "Gulbarga", "Davanagere", "Bellary", "Bijapur", "Shimoga"],
            "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode", "Vellore", "Thoothukudi", "Dindigul"],
            "Delhi": ["New Delhi", "Delhi", "North Delhi", "South Delhi", "East Delhi", "West Delhi", "Central Delhi", "North East Delhi", "North West Delhi", "South West Delhi"],
            "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Gandhinagar", "Junagadh", "Gandhidham", "Anand"],
            "Rajasthan": ["Jaipur", "Jodhpur", "Kota", "Bikaner", "Ajmer", "Udaipur", "Bhilwara", "Alwar", "Bharatpur", "Sikar"],
            "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur", "Puri", "Balasore", "Bhadrak", "Baripada", "Jharsuguda"],
            "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri", "Malda", "Bardhaman", "Kharagpur", "Haldia", "Krishnanagar"],
            "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut", "Allahabad", "Bareilly", "Aligarh", "Moradabad", "Saharanpur"],
            "Haryana": ["Gurugram", "Faridabad", "Panipat", "Ambala", "Yamunanagar", "Rohtak", "Hisar", "Karnal", "Sonipat", "Panchkula"],
            "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Mohali", "Firozpur", "Batala", "Pathankot", "Hoshiarpur"],
            "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain", "Sagar", "Dewas", "Satna", "Ratlam", "Rewa"]
        }
    },
    "Australia": {
        "states": {
            "New South Wales": ["Sydney", "Newcastle", "Wollongong", "Maitland", "Wagga Wagga", "Albury", "Port Macquarie", "Tamworth", "Orange", "Dubbo"],
            "Victoria": ["Melbourne", "Geelong", "Ballarat", "Bendigo", "Frankston", "Mildura", "Shepparton", "Wodonga", "Warrnambool", "Traralgon"],
            "Queensland": ["Brisbane", "Gold Coast", "Townsville", "Cairns", "Toowoomba", "Rockhampton", "Mackay", "Bundaberg", "Hervey Bay", "Gladstone"],
            "Western Australia": ["Perth", "Fremantle", "Bunbury", "Geraldton", "Kalgoorlie", "Mandurah", "Albany", "Rockingham", "Joondalup", "Armadale"],
            "South Australia": ["Adelaide", "Mount Gambier", "Whyalla", "Murray Bridge", "Port Lincoln", "Port Pirie", "Victor Harbor", "Kadina", "Gawler", "Port Augusta"],
            "Tasmania": ["Hobart", "Launceston", "Devonport", "Burnie", "Somerset", "Queenstown", "St Helens", "Smithton", "Currie", "Strahan"]
        }
    },
    "Brazil": {
        "states": {
            "São Paulo": ["São Paulo", "Guarulhos", "Campinas", "São Bernardo do Campo", "Santo André", "Osasco", "Ribeirão Preto", "Sorocaba", "Mauá", "São José dos Campos"],
            "Rio de Janeiro": ["Rio de Janeiro", "São Gonçalo", "Duque de Caxias", "Nova Iguaçu", "Niterói", "Belford Roxo", "São João de Meriti", "Campos dos Goytacazes", "Petrópolis", "Volta Redonda"],
            "Minas Gerais": ["Belo Horizonte", "Uberlândia", "Contagem", "Juiz de Fora", "Betim", "Montes Claros", "Ribeirão das Neves", "Uberaba", "Governador Valadares", "Ipatinga"],
            "Bahia": ["Salvador", "Feira de Santana", "Vitória da Conquista", "Camaçari", "Juazeiro", "Petrolina", "Simões Filho", "Alagoinhas", "Itabuna", "Barreiras"],
            "Paraná": ["Curitiba", "Londrina", "Maringá", "Ponta Grossa", "Cascavel", "São José dos Pinhais", "Foz do Iguaçu", "Colombo", "Guarapuava", "Paranaguá"],
            "Rio Grande do Sul": ["Porto Alegre", "Caxias do Sul", "Canoas", "Pelotas", "Santa Maria", "Gravataí", "Viamão", "Novo Hamburgo", "São Leopoldo", "Rio Grande"]
        }
    }
}

# Job Titles Based on 2030 Employment Projections
JOB_TITLES_2030 = [
    # Healthcare & Care Services (Highest Growth)
    "Home Health Aide", "Personal Care Aide", "Nursing Assistant", "Registered Nurse", "Nurse Practitioner",
    "Physical Therapist", "Occupational Therapist", "Speech-Language Pathologist", "Medical Assistant", "Pharmacy Technician",
    "Social Worker", "Mental Health Counselor", "Substance Abuse Counselor", "Healthcare Social Worker",
    
    # Technology & AI-Driven Roles
    "Software Developer", "Data Scientist", "AI/Machine Learning Engineer", "Cybersecurity Specialist", "Cloud Solutions Architect",
    "DevOps Engineer", "Full Stack Developer", "Mobile App Developer", "UI/UX Designer", "Product Manager",
    "Big Data Analyst", "Database Administrator", "Network Administrator", "Systems Analyst", "IT Project Manager",
    "AI Prompt Engineer", "AI Integration Specialist", "Algorithm Auditor", "Digital Transformation Consultant",
    
    # Green Economy & Sustainability
    "Wind Turbine Technician", "Solar Panel Installer", "Renewable Energy Engineer", "Environmental Engineer", "Sustainability Specialist",
    "Electric Vehicle Technician", "Energy Efficiency Consultant", "Environmental Compliance Specialist", "Green Building Specialist",
    "Carbon Footprint Analyst", "Climate Change Analyst", "Waste Management Specialist", "Water Treatment Specialist",
    
    # Business & Finance
    "Financial Analyst", "Management Consultant", "Business Analyst", "Operations Manager", "Human Resources Manager",
    "Marketing Manager", "Sales Manager", "Account Manager", "Project Manager", "Supply Chain Manager",
    "Investment Advisor", "Financial Planner", "Credit Analyst", "Risk Analyst", "Compliance Officer",
    "Chief AI Officer", "Chief Automation Officer", "Chief Sustainability Officer", "Digital Marketing Specialist",
    
    # Construction & Infrastructure
    "Construction Manager", "Civil Engineer", "Architect", "Electrical Engineer", "Mechanical Engineer",
    "Project Engineer", "Safety Engineer", "Quality Control Inspector", "Heavy Equipment Operator", "Carpenter",
    "Plumber", "Electrician", "HVAC Technician", "Welder", "Construction Worker",
    
    # Transportation & Logistics
    "Logistics Coordinator", "Supply Chain Analyst", "Warehouse Manager", "Delivery Driver", "Truck Driver",
    "Pilot", "Air Traffic Controller", "Maritime Engineer", "Fleet Manager", "Transportation Planner",
    
    # Education & Training
    "Elementary School Teacher", "High School Teacher", "University Professor", "Training Specialist", "Instructional Designer",
    "Education Administrator", "School Counselor", "Special Education Teacher", "ESL Teacher", "Online Learning Specialist",
    
    # Creative & Media
    "Graphic Designer", "Web Designer", "Content Creator", "Digital Marketing Specialist", "Social Media Manager",
    "Video Editor", "Photographer", "Writer", "Copywriter", "Creative Director", "Brand Manager",
    
    # Manufacturing & Production
    "Manufacturing Engineer", "Quality Assurance Manager", "Production Supervisor", "Maintenance Technician", "Industrial Engineer",
    "Automation Technician", "CNC Machinist", "Assembly Line Worker", "Factory Worker", "Production Planner",
    
    # Emerging Roles (2030 Specific)
    "Robotics Engineer", "IoT Specialist", "Blockchain Developer", "Virtual Reality Developer", "Augmented Reality Developer",
    "Drone Operator", "3D Printing Technician", "Biomedical Engineer", "Genetic Counselor", "Telemedicine Specialist",
    "Digital Health Specialist", "Cybersecurity Analyst", "Cloud Security Engineer", "Data Privacy Officer",
    "Customer Success Manager", "Growth Hacker", "Conversion Rate Optimizer", "SEO Specialist", "Influencer Marketing Manager",
    
    # Service Industry
    "Restaurant Manager", "Chef", "Server", "Bartender", "Hotel Manager", "Travel Agent", "Event Planner",
    "Customer Service Representative", "Call Center Agent", "Retail Sales Associate", "Store Manager", "Cashier",
    
    # Legal & Government
    "Lawyer", "Legal Assistant", "Paralegal", "Court Reporter", "Judge", "Government Administrator",
    "Policy Analyst", "Urban Planner", "Public Relations Specialist", "Communications Manager",
    
    # Research & Development
    "Research Scientist", "Lab Technician", "Clinical Research Coordinator", "Market Research Analyst",
    "Product Development Engineer", "Innovation Manager", "Patent Attorney", "Technical Writer"
]

# Education Levels
EDUCATION_LEVELS = [
    "Less than High School",
    "High School Diploma",
    "Some College",
    "Associate Degree",
    "Bachelor's Degree",
    "Master's Degree",
    "Professional Degree",
    "Doctoral Degree (PhD)"
]

# Global Industries Based on 2025 Classification
GLOBAL_INDUSTRIES = [
    # Primary Industries
    "Agriculture & Farming", "Forestry & Logging", "Fishing & Aquaculture", "Mining & Extraction",
    "Oil & Gas Extraction", "Coal Mining", "Metal Mining", "Quarrying",
    
    # Energy & Utilities
    "Electric Power Generation", "Natural Gas Distribution", "Water & Sewage Treatment", "Renewable Energy",
    "Nuclear Power", "Solar Energy", "Wind Energy", "Hydroelectric Power",
    
    # Manufacturing
    "Automotive Manufacturing", "Aerospace & Defense", "Electronics Manufacturing", "Computer Hardware",
    "Semiconductor Manufacturing", "Chemical Manufacturing", "Pharmaceutical Manufacturing", "Food Processing",
    "Textile & Apparel Manufacturing", "Furniture Manufacturing", "Paper Manufacturing", "Steel Production",
    "Plastic Products Manufacturing", "Medical Device Manufacturing", "Construction Materials",
    
    # Technology
    "Software Development", "Information Technology", "Telecommunications", "Internet Services",
    "Artificial Intelligence", "Cybersecurity", "Cloud Computing", "Data Analytics",
    "Blockchain Technology", "Robotics & Automation", "Internet of Things (IoT)",
    
    # Financial Services
    "Commercial Banking", "Investment Banking", "Insurance", "Asset Management", "Credit Services",
    "Real Estate Investment", "Financial Technology (Fintech)", "Cryptocurrency", "Venture Capital",
    "Private Equity", "Pension Funds", "Credit Unions",
    
    # Healthcare
    "Hospitals & Healthcare Systems", "Pharmaceutical Services", "Medical Equipment", "Biotechnology",
    "Telemedicine", "Mental Health Services", "Home Healthcare", "Nursing Care", "Medical Research",
    "Health Insurance", "Dental Services", "Vision Care",
    
    # Retail & E-commerce
    "Online Retail", "Department Stores", "Grocery Stores", "Specialty Retail", "Fashion Retail",
    "Electronics Retail", "Automotive Retail", "Home Improvement Retail", "Wholesale Trade",
    "Supply Chain & Logistics", "Warehousing & Distribution",
    
    # Transportation
    "Airlines", "Shipping & Maritime", "Trucking & Freight", "Rail Transportation", "Public Transit",
    "Ride-sharing Services", "Delivery Services", "Logistics Services", "Automotive Services",
    
    # Real Estate & Construction
    "Residential Real Estate", "Commercial Real Estate", "Real Estate Development", "Property Management",
    "Construction", "Architecture & Engineering", "Building Materials", "Home Building",
    "Infrastructure Development", "Urban Planning",
    
    # Hospitality & Tourism
    "Hotels & Resorts", "Restaurants & Food Service", "Travel & Tourism", "Entertainment",
    "Gaming & Casinos", "Event Management", "Recreation Services", "Sports & Fitness",
    
    # Media & Entertainment
    "Broadcasting", "Film & Television", "Music Industry", "Publishing", "Digital Media",
    "Gaming Industry", "Streaming Services", "Social Media", "Advertising & Marketing",
    "Public Relations", "Content Creation",
    
    # Education
    "K-12 Education", "Higher Education", "Online Education", "Vocational Training", "Corporate Training",
    "Educational Technology", "Tutoring Services", "Educational Publishing",
    
    # Professional Services
    "Legal Services", "Accounting Services", "Consulting", "Human Resources", "Marketing Services",
    "Design Services", "Research Services", "Management Services", "Administrative Services",
    
    # Government & Public Sector
    "Federal Government", "State Government", "Local Government", "Military & Defense",
    "Public Safety", "Emergency Services", "Postal Services", "Public Administration",
    
    # Non-Profit & Social Services
    "Non-Profit Organizations", "Social Services", "Religious Organizations", "Charitable Organizations",
    "Community Services", "Environmental Organizations", "International Organizations",
    
    # Emerging Industries
    "Space Technology", "Quantum Computing", "Nanotechnology", "Gene Therapy", "Personalized Medicine",
    "Digital Health", "Clean Technology", "Carbon Capture", "Autonomous Vehicles", "Drone Technology",
    "Virtual Reality", "Augmented Reality", "3D Printing", "Smart Cities", "Sustainable Agriculture"
]

# Prediction Models Available
PREDICTION_MODELS = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting"
]

# Currency Exchange Rates (as of June 27, 2025)
CURRENCY_RATES = {
    "USD": {"rate": 1.0, "symbol": "$", "name": "US Dollar"},
    "INR": {"rate": 85.56, "symbol": "₹", "name": "Indian Rupee"},
    "EUR": {"rate": 0.91, "symbol": "€", "name": "Euro"},
    "GBP": {"rate": 0.79, "symbol": "£", "name": "British Pound"},
    "JPY": {"rate": 156.0, "symbol": "¥", "name": "Japanese Yen"},
    "CNY": {"rate": 7.25, "symbol": "¥", "name": "Chinese Yuan"},
    "AUD": {"rate": 1.47, "symbol": "A$", "name": "Australian Dollar"},
    "CAD": {"rate": 1.36, "symbol": "C$", "name": "Canadian Dollar"},
    "BRL": {"rate": 5.30, "symbol": "R$", "name": "Brazilian Real"},
    "CHF": {"rate": 0.89, "symbol": "Fr", "name": "Swiss Franc"},
    "SEK": {"rate": 10.45, "symbol": "kr", "name": "Swedish Krona"},
    "NOK": {"rate": 10.78, "symbol": "kr", "name": "Norwegian Krone"},
    "DKK": {"rate": 6.78, "symbol": "kr", "name": "Danish Krone"},
    "SGD": {"rate": 1.34, "symbol": "S$", "name": "Singapore Dollar"},
    "HKD": {"rate": 7.80, "symbol": "HK$", "name": "Hong Kong Dollar"},
    "KRW": {"rate": 1320.0, "symbol": "₩", "name": "South Korean Won"},
    "MXN": {"rate": 18.25, "symbol": "$", "name": "Mexican Peso"},
    "ZAR": {"rate": 18.45, "symbol": "R", "name": "South African Rand"},
    "RUB": {"rate": 88.50, "symbol": "₽", "name": "Russian Ruble"},
    "TRY": {"rate": 32.15, "symbol": "₺", "name": "Turkish Lira"}
}

def convert_currency(amount_usd, target_currency):
    """Convert USD amount to target currency"""
    if target_currency not in CURRENCY_RATES:
        return amount_usd, "USD", "$"
    
    rate = CURRENCY_RATES[target_currency]["rate"]
    symbol = CURRENCY_RATES[target_currency]["symbol"]
    converted_amount = amount_usd * rate
    
    return converted_amount, target_currency, symbol