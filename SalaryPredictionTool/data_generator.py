import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=10000):
    """
    Generate synthetic employee salary data for training and testing
    """
    np.random.seed(42)
    random.seed(42)
    
    # Define data categories
    job_titles = [
        # Technology & Engineering
        "Software Engineer", "Senior Software Engineer", "Principal Software Engineer", "Software Architect",
        "Data Scientist", "Senior Data Scientist", "Principal Data Scientist", "Chief Data Scientist",
        "Data Engineer", "Senior Data Engineer", "Data Architect", "Machine Learning Engineer",
        "DevOps Engineer", "Senior DevOps Engineer", "DevOps Architect", "Site Reliability Engineer",
        "Frontend Developer", "Backend Developer", "Full Stack Developer", "Mobile Developer",
        "Cloud Engineer", "Cloud Architect", "Platform Engineer", "Security Engineer",
        "AI/ML Engineer", "Deep Learning Engineer", "Computer Vision Engineer", "NLP Engineer",
        "Blockchain Developer", "Smart Contract Developer", "Web3 Developer", "Cryptocurrency Analyst",
        "Cybersecurity Analyst", "Information Security Manager", "Penetration Tester", "Security Consultant",
        "Database Administrator", "Database Engineer", "Data Warehouse Engineer", "ETL Developer",
        "System Administrator", "Network Engineer", "Infrastructure Engineer", "IT Manager",
        "Technical Support Engineer", "Customer Support Engineer", "Field Service Engineer",
        "Quality Assurance Engineer", "QA Automation Engineer", "Test Engineer", "QA Manager",
        "Technical Writer", "Documentation Manager", "Technical Content Creator", "Developer Advocate",
        "Solutions Architect", "Enterprise Architect", "Integration Architect", "API Developer",
        
        # Management & Leadership
        "Product Manager", "Senior Product Manager", "Director of Product", "Chief Product Officer",
        "Engineering Manager", "Senior Engineering Manager", "Director of Engineering", "VP of Engineering",
        "Project Manager", "Senior Project Manager", "Program Manager", "Portfolio Manager",
        "Scrum Master", "Agile Coach", "Project Coordinator", "Program Coordinator",
        "Team Lead", "Tech Lead", "Principal Engineer", "Staff Engineer",
        "Operations Manager", "Senior Operations Manager", "Operations Director", "COO",
        "General Manager", "Department Manager", "Regional Manager", "Branch Manager",
        "Business Development Manager", "Partnership Manager", "Strategic Account Manager",
        
        # Business & Finance
        "Financial Analyst", "Senior Financial Analyst", "Finance Manager", "Financial Controller",
        "Business Analyst", "Senior Business Analyst", "Business Intelligence Manager", "Data Analyst",
        "Investment Analyst", "Portfolio Manager", "Risk Analyst", "Credit Analyst",
        "Management Consultant", "Strategy Consultant", "Business Consultant", "Operations Consultant",
        "Accountant", "Senior Accountant", "Staff Accountant", "Accounting Manager",
        "Auditor", "Internal Auditor", "External Auditor", "Compliance Officer",
        "Treasury Analyst", "Budget Analyst", "Financial Planner", "Investment Advisor",
        "Insurance Underwriter", "Claims Adjuster", "Actuary", "Risk Manager",
        
        # Sales & Marketing
        "Sales Representative", "Senior Sales Representative", "Sales Manager", "Sales Director",
        "Account Executive", "Senior Account Executive", "Key Account Manager", "Territory Manager",
        "Business Development Representative", "Sales Development Representative", "Inside Sales Rep",
        "Marketing Manager", "Senior Marketing Manager", "Marketing Director", "CMO",
        "Digital Marketing Manager", "Content Marketing Manager", "Email Marketing Manager",
        "Social Media Manager", "SEO Specialist", "SEM Specialist", "Growth Hacker",
        "Brand Manager", "Product Marketing Manager", "Marketing Coordinator", "PR Manager",
        "Customer Success Manager", "Customer Experience Manager", "Account Manager",
        "Sales Engineer", "Technical Sales Representative", "Pre-Sales Engineer",
        
        # Human Resources
        "HR Manager", "Senior HR Manager", "HR Director", "Chief Human Resources Officer",
        "HR Generalist", "HR Business Partner", "HR Coordinator", "HR Assistant",
        "Recruiter", "Senior Recruiter", "Technical Recruiter", "Talent Acquisition Manager",
        "Compensation Analyst", "Benefits Administrator", "Training Manager", "L&D Manager",
        "Employee Relations Manager", "Organizational Development Manager", "HR Analytics Manager",
        "Diversity & Inclusion Manager", "Workplace Culture Manager", "People Operations Manager",
        
        # Design & Creative
        "UX Designer", "Senior UX Designer", "UX Researcher", "Design Director",
        "UI Designer", "Visual Designer", "Product Designer", "Interaction Designer",
        "Graphic Designer", "Web Designer", "Brand Designer", "Creative Director",
        "Art Director", "Senior Art Director", "Design Manager", "Creative Manager",
        "Animator", "Motion Graphics Designer", "Video Editor", "Multimedia Designer",
        "Industrial Designer", "Fashion Designer", "Interior Designer", "Architect",
        
        # Healthcare & Life Sciences
        "Physician", "Surgeon", "Specialist", "Resident", "Fellow", "Attending Physician",
        "Nurse", "Registered Nurse", "Nurse Practitioner", "Nursing Manager",
        "Pharmacist", "Clinical Pharmacist", "Pharmacy Manager", "Pharmaceutical Researcher",
        "Medical Researcher", "Clinical Research Coordinator", "Biostatistician", "Epidemiologist",
        "Medical Device Engineer", "Biomedical Engineer", "Clinical Engineer", "Regulatory Affairs Manager",
        "Medical Writer", "Clinical Data Manager", "Quality Assurance Manager", "Compliance Manager",
        
        # Education & Training
        "Teacher", "Professor", "Associate Professor", "Assistant Professor", "Lecturer",
        "Principal", "Vice Principal", "Dean", "Academic Administrator", "Registrar",
        "Curriculum Developer", "Instructional Designer", "Education Coordinator", "Training Manager",
        "School Counselor", "Academic Advisor", "Student Affairs Manager", "Research Coordinator",
        "Librarian", "Research Librarian", "Digital Librarian", "Information Specialist",
        
        # Legal & Compliance
        "Lawyer", "Attorney", "Legal Counsel", "General Counsel", "Corporate Lawyer",
        "Paralegal", "Legal Assistant", "Contract Manager", "Legal Analyst",
        "Compliance Officer", "Regulatory Affairs Manager", "Risk Manager", "Audit Manager",
        "Patent Attorney", "Intellectual Property Manager", "Legal Operations Manager",
        
        # Media & Communications
        "Journalist", "Reporter", "News Anchor", "Editor", "Content Creator",
        "Communications Manager", "Public Relations Manager", "Social Media Manager",
        "Content Writer", "Copywriter", "Technical Writer", "Grant Writer",
        "Video Producer", "Audio Engineer", "Broadcast Engineer", "Media Coordinator",
        "Marketing Communications Manager", "Internal Communications Manager",
        
        # Manufacturing & Operations
        "Manufacturing Engineer", "Process Engineer", "Quality Engineer", "Production Manager",
        "Operations Manager", "Plant Manager", "Facility Manager", "Maintenance Manager",
        "Supply Chain Manager", "Logistics Manager", "Procurement Manager", "Inventory Manager",
        "Warehouse Manager", "Distribution Manager", "Transportation Manager", "Shipping Coordinator",
        "Production Supervisor", "Quality Control Inspector", "Safety Manager", "Environmental Manager",
        
        # Customer Service & Support
        "Customer Service Representative", "Customer Success Manager", "Support Manager",
        "Technical Support Specialist", "Help Desk Technician", "Customer Experience Manager",
        "Call Center Manager", "Customer Relations Manager", "Account Manager", "Client Services Manager",
        
        # Government & Public Service
        "Policy Analyst", "Government Relations Manager", "Public Administrator", "City Manager",
        "Program Manager", "Grant Manager", "Social Worker", "Case Manager",
        "Urban Planner", "Public Health Analyst", "Environmental Scientist", "Research Analyst",
        "Budget Analyst", "Legislative Assistant", "Communications Director", "Public Affairs Manager",
        
        # Real Estate & Construction
        "Real Estate Agent", "Real Estate Broker", "Property Manager", "Real Estate Developer",
        "Construction Manager", "Project Manager", "Site Supervisor", "Construction Engineer",
        "Architect", "Civil Engineer", "Structural Engineer", "Mechanical Engineer",
        "Electrical Engineer", "HVAC Engineer", "Plumbing Engineer", "Building Inspector",
        "Surveyor", "Land Use Planner", "Environmental Consultant", "Construction Estimator",
        
        # Retail & Hospitality
        "Store Manager", "Assistant Manager", "Department Manager", "Regional Manager",
        "Buyer", "Merchandise Manager", "Inventory Manager", "Loss Prevention Manager",
        "Hotel Manager", "Restaurant Manager", "Food Service Manager", "Event Manager",
        "Tourism Manager", "Recreation Manager", "Hospitality Coordinator", "Guest Services Manager",
        "Chef", "Sous Chef", "Kitchen Manager", "Food & Beverage Manager",
        
        # Transportation & Logistics
        "Logistics Coordinator", "Supply Chain Analyst", "Transportation Manager", "Fleet Manager",
        "Pilot", "Air Traffic Controller", "Airport Manager", "Aviation Manager",
        "Ship Captain", "Marine Engineer", "Port Manager", "Customs Agent",
        "Truck Driver", "Delivery Driver", "Courier", "Dispatcher",
        "Railway Engineer", "Train Conductor", "Transit Manager", "Traffic Engineer",
        
        # Energy & Utilities
        "Power Plant Operator", "Electrical Engineer", "Renewable Energy Engineer", "Energy Analyst",
        "Utility Manager", "Grid Operator", "Energy Trader", "Petroleum Engineer",
        "Mining Engineer", "Geologist", "Environmental Engineer", "Safety Engineer",
        "Nuclear Engineer", "Chemical Engineer", "Process Engineer", "Maintenance Engineer",
        
        # Agriculture & Food
        "Agricultural Engineer", "Farm Manager", "Crop Scientist", "Food Scientist",
        "Veterinarian", "Animal Nutritionist", "Agricultural Inspector", "Food Safety Manager",
        "Agronomist", "Soil Scientist", "Plant Breeder", "Agricultural Economist",
        "Food Production Manager", "Quality Control Manager", "Supply Chain Manager",
        
        # Sports & Entertainment
        "Sports Manager", "Athletic Director", "Coach", "Personal Trainer",
        "Sports Analyst", "Sports Marketing Manager", "Event Coordinator", "Stadium Manager",
        "Entertainment Manager", "Talent Agent", "Producer", "Director",
        "Music Producer", "Sound Engineer", "Video Game Developer", "Game Designer",
        "Esports Manager", "Content Creator", "Streamer", "Podcast Producer"
    ]
    
    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
    
    locations = [
        # United States - Major Cities
        'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ',
        'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA', 'Dallas, TX', 'San Jose, CA',
        'Austin, TX', 'Jacksonville, FL', 'Fort Worth, TX', 'Columbus, OH', 'Charlotte, NC',
        'San Francisco, CA', 'Indianapolis, IN', 'Seattle, WA', 'Denver, CO', 'Washington, DC',
        'Boston, MA', 'El Paso, TX', 'Nashville, TN', 'Detroit, MI', 'Oklahoma City, OK',
        'Portland, OR', 'Las Vegas, NV', 'Memphis, TN', 'Louisville, KY', 'Baltimore, MD',
        'Milwaukee, WI', 'Albuquerque, NM', 'Tucson, AZ', 'Fresno, CA', 'Mesa, AZ',
        'Sacramento, CA', 'Atlanta, GA', 'Kansas City, MO', 'Colorado Springs, CO', 'Miami, FL',
        'Raleigh, NC', 'Omaha, NE', 'Long Beach, CA', 'Virginia Beach, VA', 'Oakland, CA',
        'Minneapolis, MN', 'Tulsa, OK', 'Arlington, TX', 'Tampa, FL', 'New Orleans, LA',
        'Wichita, KS', 'Cleveland, OH', 'Bakersfield, CA', 'Aurora, CO', 'Anaheim, CA',
        'Honolulu, HI', 'Santa Ana, CA', 'Corpus Christi, TX', 'Riverside, CA', 'Lexington, KY',
        'Stockton, CA', 'Henderson, NV', 'Saint Paul, MN', 'St. Louis, MO', 'Cincinnati, OH',
        'Pittsburgh, PA', 'Greensboro, NC', 'Anchorage, AK', 'Plano, TX', 'Lincoln, NE',
        'Orlando, FL', 'Irvine, CA', 'Newark, NJ', 'Durham, NC', 'Chula Vista, CA',
        'Toledo, OH', 'Fort Wayne, IN', 'St. Petersburg, FL', 'Laredo, TX', 'Jersey City, NJ',
        'Chandler, AZ', 'Madison, WI', 'Lubbock, TX', 'Norfolk, VA', 'Baton Rouge, LA',
        'Burnsville, MN', 'North Las Vegas, NV', 'Irving, TX', 'Chesapeake, VA', 'Gilbert, AZ',
        'Reno, NV', 'Hialeah, FL', 'Garland, TX', 'Glendale, AZ', 'Scottsdale, AZ',
        'Boise, ID', 'Fremont, CA', 'Richmond, VA', 'San Bernardino, CA', 'Birmingham, AL',
        'Spokane, WA', 'Rochester, NY', 'Des Moines, IA', 'Modesto, CA', 'Fayetteville, NC',
        'Tacoma, WA', 'Oxnard, CA', 'Fontana, CA', 'Columbus, GA', 'Montgomery, AL',
        'Moreno Valley, CA', 'Shreveport, LA', 'Aurora, IL', 'Yonkers, NY', 'Akron, OH',
        'Huntington Beach, CA', 'Little Rock, AR', 'Augusta, GA', 'Amarillo, TX', 'Glendale, CA',
        'Mobile, AL', 'Grand Rapids, MI', 'Salt Lake City, UT', 'Tallahassee, FL', 'Huntsville, AL',
        'Grand Prairie, TX', 'Knoxville, TN', 'Worcester, MA', 'Newport News, VA', 'Brownsville, TX',
        'Overland Park, KS', 'Santa Clarita, CA', 'Providence, RI', 'Garden Grove, CA', 'Chattanooga, TN',
        'Oceanside, CA', 'Jackson, MS', 'Fort Lauderdale, FL', 'Santa Rosa, CA', 'Rancho Cucamonga, CA',
        'Port St. Lucie, FL', 'Tempe, AZ', 'Ontario, CA', 'Vancouver, WA', 'Cape Coral, FL',
        'Sioux Falls, SD', 'Springfield, MO', 'Peoria, AZ', 'Pembroke Pines, FL', 'Elk Grove, CA',
        'Salem, OR', 'Lancaster, CA', 'Corona, CA', 'Eugene, OR', 'Palmdale, CA',
        'Salinas, CA', 'Springfield, MA', 'Pasadena, CA', 'Fort Collins, CO', 'Hayward, CA',
        'Pomona, CA', 'Cary, NC', 'Rockford, IL', 'Alexandria, VA', 'Escondido, CA',
        'Lakewood, CO', 'Torrance, CA', 'Bridgeport, CT', 'Paterson, NJ', 'Sunrise, FL',
        'Mesquite, TX', 'Sterling Heights, MI', 'Thousand Oaks, CA', 'Roseville, CA', 'Cedar Rapids, IA',
        'Coral Springs, FL', 'Stamford, CT', 'Concord, CA', 'Hartford, CT', 'Kent, WA',
        'Lafayette, LA', 'Midland, TX', 'Victorville, CA', 'Topeka, KS', 'Carrollton, TX',
        'Surprise, AZ', 'Daly City, CA', 'Visalia, CA', 'Olathe, KS', 'Thornton, CO',
        'Waterbury, CT', 'Norwalk, CA', 'Abilene, TX', 'Odessa, TX', 'Waco, TX',
        'Charleston, SC', 'Manchester, NH', 'Allentown, PA', 'McAllen, TX', 'Killeen, TX',
        'College Station, TX', 'Carson, CA', 'Pearland, TX', 'Rialto, CA', 'Dayton, OH',
        'Miami Gardens, FL', 'Temecula, CA', 'Columbia, SC', 'West Valley City, UT', 'Santa Maria, CA',
        'El Monte, CA', 'Murfreesboro, TN', 'Miami Beach, FL', 'Clarksville, TN', 'Westminster, CO',
        'Miramar, FL', 'Santa Monica, CA', 'Norman, OK', 'Fargo, ND', 'Pueblo, CO',
        'Round Rock, TX', 'Inglewood, CA', 'Broken Arrow, OK', 'Lawton, OK', 'Evansville, IN',
        'Beaumont, TX', 'Elgin, IL', 'Peoria, IL', 'Downey, CA', 'Lansing, MI',
        'Wichita Falls, TX', 'Pompano Beach, FL', 'Costa Mesa, CA', 'Miami Lakes, FL', 'Boca Raton, FL',
        'Lewisville, TX', 'South Bend, IN', 'Billings, MT', 'Burbank, CA', 'Vallejo, CA',
        'Lowell, MA', 'El Cajon, CA', 'Arvada, CO', 'Pueblo, CO', 'Centennial, CO',
        'Richardson, TX', 'Clearwater, FL', 'West Palm Beach, FL', 'McKinney, TX', 'Cambridge, MA',
        'Westminster, CA', 'Fairfield, CA', 'Carlsbad, CA', 'Concord, NC', 'Antioch, CA',
        'Temecula, CA', 'Berkeley, CA', 'Frisco, TX', 'Richmond, CA', 'Murrieta, CA',
        
        # Canada - Major Cities
        'Toronto, ON', 'Montreal, QC', 'Vancouver, BC', 'Calgary, AB', 'Edmonton, AB',
        'Ottawa, ON', 'Winnipeg, MB', 'Quebec City, QC', 'Hamilton, ON', 'Kitchener, ON',
        'London, ON', 'Victoria, BC', 'Halifax, NS', 'Oshawa, ON', 'Windsor, ON',
        'Saskatoon, SK', 'St. Catharines, ON', 'Regina, SK', 'Sherbrooke, QC', 'Barrie, ON',
        'Kelowna, BC', 'Abbotsford, BC', 'Sudbury, ON', 'Kingston, ON', 'Saguenay, QC',
        'Trois-Rivières, QC', 'Guelph, ON', 'Cambridge, ON', 'Whitby, ON', 'Coquitlam, BC',
        'Saanich, BC', 'Burlington, ON', 'Richmond, BC', 'Oakville, ON', 'Burnaby, BC',
        'Red Deer, AB', 'Brantford, ON', 'Lethbridge, AB', 'Kamloops, BC', 'Milton, ON',
        'Moncton, NB', 'Nanaimo, BC', 'Thunder Bay, ON', 'Dieppe, NB', 'Waterloo, ON',
        'Delta, BC', 'Chatham, ON', 'Saint John, NB', 'Fredericton, NB', 'Chilliwack, BC',
        
        # United Kingdom - Major Cities
        'London, England', 'Birmingham, England', 'Manchester, England', 'Leeds, England',
        'Liverpool, England', 'Sheffield, England', 'Bristol, England', 'Newcastle, England',
        'Nottingham, England', 'Leicester, England', 'Coventry, England', 'Bradford, England',
        'Stoke-on-Trent, England', 'Wolverhampton, England', 'Plymouth, England', 'Derby, England',
        'Southampton, England', 'Swansea, Wales', 'Cardiff, Wales', 'Newport, Wales',
        'Edinburgh, Scotland', 'Glasgow, Scotland', 'Aberdeen, Scotland', 'Dundee, Scotland',
        'Stirling, Scotland', 'Perth, Scotland', 'Inverness, Scotland', 'Paisley, Scotland',
        'Belfast, Northern Ireland', 'Derry, Northern Ireland', 'Lisburn, Northern Ireland',
        'Newtownabbey, Northern Ireland', 'Bangor, Northern Ireland', 'Craigavon, Northern Ireland',
        'Brighton, England', 'Bournemouth, England', 'Swindon, England', 'Huddersfield, England',
        'Poole, England', 'Oxford, England', 'Middlesbrough, England', 'Blackpool, England',
        'Bolton, England', 'Ipswich, England', 'York, England', 'Peterborough, England',
        'Stockport, England', 'Norwich, England', 'Rotherham, England', 'Cambridge, England',
        'Watford, England', 'Slough, England', 'Exeter, England', 'Crawley, England',
        'Basildon, England', 'Northampton, England', 'Cheltenham, England', 'Luton, England',
        'Southend-on-Sea, England', 'Reading, England', 'Oldham, England', 'Gloucester, England',
        'Blackburn, England', 'Milton Keynes, England', 'Sunderland, England', 'Grimsby, England',
        'Saint Helens, England', 'Telford, England', 'Maidstone, England', 'Hastings, England',
        'High Wycombe, England', 'Basingstoke, England', 'Warrington, England', 'Burton upon Trent, England',
        'Colchester, England', 'Eastbourne, England', 'Scunthorpe, England', 'Gloucester, England',
        'Salford, England', 'Chesterfield, England', 'Chelmsford, England', 'Carlisle, England',
        'Worcester, England', 'Nuneaton, England', 'Lowestoft, England', 'Stevenage, England',
        'Stockton-on-Tees, England', 'Hartlepool, England', 'Barnsley, England', 'Doncaster, England',
        'Gateshead, England', 'South Shields, England', 'Tynemouth, England', 'Wallsend, England',
        'Wigan, England', 'Preston, England', 'Blackburn, England', 'Burnley, England',
        
        # Germany - Major Cities
        'Berlin, Germany', 'Hamburg, Germany', 'Munich, Germany', 'Cologne, Germany',
        'Frankfurt, Germany', 'Stuttgart, Germany', 'Düsseldorf, Germany', 'Dortmund, Germany',
        'Essen, Germany', 'Leipzig, Germany', 'Bremen, Germany', 'Dresden, Germany',
        'Hanover, Germany', 'Nuremberg, Germany', 'Duisburg, Germany', 'Bochum, Germany',
        'Wuppertal, Germany', 'Bielefeld, Germany', 'Bonn, Germany', 'Münster, Germany',
        'Mannheim, Germany', 'Karlsruhe, Germany', 'Augsburg, Germany', 'Wiesbaden, Germany',
        'Gelsenkirchen, Germany', 'Mönchengladbach, Germany', 'Braunschweig, Germany', 'Chemnitz, Germany',
        'Kiel, Germany', 'Aachen, Germany', 'Halle, Germany', 'Magdeburg, Germany',
        'Freiburg, Germany', 'Krefeld, Germany', 'Lübeck, Germany', 'Oberhausen, Germany',
        'Erfurt, Germany', 'Mainz, Germany', 'Rostock, Germany', 'Kassel, Germany',
        'Hagen, Germany', 'Hamm, Germany', 'Saarbrücken, Germany', 'Mülheim, Germany',
        'Potsdam, Germany', 'Ludwigshafen, Germany', 'Oldenburg, Germany', 'Leverkusen, Germany',
        'Osnabrück, Germany', 'Solingen, Germany', 'Heidelberg, Germany', 'Herne, Germany',
        'Neuss, Germany', 'Darmstadt, Germany', 'Paderborn, Germany', 'Regensburg, Germany',
        'Ingolstadt, Germany', 'Würzburg, Germany', 'Fürth, Germany', 'Wolfsburg, Germany',
        'Offenbach, Germany', 'Ulm, Germany', 'Heilbronn, Germany', 'Pforzheim, Germany',
        'Göttingen, Germany', 'Bottrop, Germany', 'Trier, Germany', 'Recklinghausen, Germany',
        'Reutlingen, Germany', 'Bremerhaven, Germany', 'Koblenz, Germany', 'Bergisch Gladbach, Germany',
        'Jena, Germany', 'Remscheid, Germany', 'Erlangen, Germany', 'Moers, Germany',
        'Siegen, Germany', 'Hildesheim, Germany', 'Salzgitter, Germany', 'Cottbus, Germany',
        
        # France - Major Cities
        'Paris, France', 'Marseille, France', 'Lyon, France', 'Toulouse, France',
        'Nice, France', 'Nantes, France', 'Strasbourg, France', 'Montpellier, France',
        'Bordeaux, France', 'Lille, France', 'Rennes, France', 'Reims, France',
        'Le Havre, France', 'Saint-Étienne, France', 'Toulon, France', 'Angers, France',
        'Grenoble, France', 'Dijon, France', 'Nîmes, France', 'Aix-en-Provence, France',
        'Saint-Quentin-en-Yvelines, France', 'Brest, France', 'Le Mans, France', 'Amiens, France',
        'Tours, France', 'Limoges, France', 'Clermont-Ferrand, France', 'Villeurbanne, France',
        'Besançon, France', 'Orléans, France', 'Mulhouse, France', 'Metz, France',
        'Rouen, France', 'Caen, France', 'Nancy, France', 'Argenteuil, France',
        'Montreuil, France', 'Roubaix, France', 'Tourcoing, France', 'Nanterre, France',
        'Avignon, France', 'Créteil, France', 'Dunkirk, France', 'Poitiers, France',
        'Asnières-sur-Seine, France', 'Versailles, France', 'Courbevoie, France', 'Vitry-sur-Seine, France',
        'Pau, France', 'La Rochelle, France', 'Calais, France', 'Cannes, France',
        'Boulogne-Billancourt, France', 'Bourges, France', 'Perpignan, France', 'Bayonne, France',
        'Vannes, France', 'Lorient, France', 'Troyes, France', 'Valence, France',
        'Montauban, France', 'Niort, France', 'Chambéry, France', 'Angoulême, France',
        'Béziers, France', 'Beauvais, France', 'Cholet, France', 'Mérignac, France',
        'Saint-Nazaire, France', 'Colmar, France', 'Issy-les-Moulineaux, France', 'Pessac, France',
        'Levallois-Perret, France', 'Cergy, France', 'Ivry-sur-Seine, France', 'Antony, France',
        
        # Japan - Major Cities
        'Tokyo, Japan', 'Osaka, Japan', 'Nagoya, Japan', 'Sapporo, Japan',
        'Fukuoka, Japan', 'Kobe, Japan', 'Kyoto, Japan', 'Kawasaki, Japan',
        'Saitama, Japan', 'Hiroshima, Japan', 'Yohohama, Japan', 'Sendai, Japan',
        'Kitakyushu, Japan', 'Chiba, Japan', 'Setagaya, Japan', 'Nerima, Japan',
        'Ota, Japan', 'Adachi, Japan', 'Katsushika, Japan', 'Suginami, Japan',
        'Koto, Japan', 'Edogawa, Japan', 'Nakano, Japan', 'Shibuya, Japan',
        'Shinjuku, Japan', 'Toshima, Japan', 'Sumida, Japan', 'Kita, Japan',
        'Arakawa, Japan', 'Itabashi, Japan', 'Meguro, Japan', 'Shinagawa, Japan',
        'Hachioji, Japan', 'Tachikawa, Japan', 'Musashino, Japan', 'Mitaka, Japan',
        'Ome, Japan', 'Fuchu, Japan', 'Akishima, Japan', 'Chofu, Japan',
        'Machida, Japan', 'Koganei, Japan', 'Kodaira, Japan', 'Hino, Japan',
        'Higashimurayama, Japan', 'Kokubunji, Japan', 'Kunitachi, Japan', 'Fussa, Japan',
        'Komae, Japan', 'Higashiyamato, Japan', 'Kiyose, Japan', 'Higashikurume, Japan',
        'Musashimurayama, Japan', 'Tama, Japan', 'Inagi, Japan', 'Hamura, Japan',
        'Akiruno, Japan', 'Nishitokyo, Japan', 'Mizuho, Japan', 'Okutama, Japan',
        'Hinode, Japan', 'Hinohara, Japan', 'Oshima, Japan', 'Toshima, Japan',
        'Niijima, Japan', 'Shikinejima, Japan', 'Kozushima, Japan', 'Miyakejima, Japan',
        'Mikurajima, Japan', 'Hachijojima, Japan', 'Aogashima, Japan', 'Ogasawara, Japan',
        
        # China - Major Cities
        'Beijing, China', 'Shanghai, China', 'Guangzhou, China', 'Shenzhen, China',
        'Tianjin, China', 'Wuhan, China', 'Dongguan, China', 'Chengdu, China',
        'Nanjing, China', 'Chongqing, China', 'Shenyang, China', 'Hangzhou, China',
        'Xian, China', 'Harbin, China', 'Suzhou, China', 'Qingdao, China',
        'Dalian, China', 'Zhengzhou, China', 'Shantou, China', 'Jinan, China',
        'Changchun, China', 'Kunming, China', 'Changsha, China', 'Taiyuan, China',
        'Xiamen, China', 'Shijiazhuang, China', 'Wenzhou, China', 'Aomen, China',
        'Ningbo, China', 'Wuxi, China', 'Xianggang, China', 'Zibo, China',
        'Yantai, China', 'Fushun, China', 'Anshan, China', 'Qiqihar, China',
        'Daqing, China', 'Baotou, China', 'Hohhot, China', 'Luoyang, China',
        'Yichang, China', 'Xiangfan, China', 'Hengyang, China', 'Zhuzhou, China',
        'Guilin, China', 'Nanning, China', 'Liuzhou, China', 'Haikou, China',
        'Sanya, China', 'Guiyang, China', 'Zunyi, China', 'Lanzhou, China',
        'Xining, China', 'Yinchuan, China', 'Urumqi, China', 'Lhasa, China',
        
        # India - Major Cities
        'Mumbai, India', 'Delhi, India', 'Bangalore, India', 'Hyderabad, India',
        'Chennai, India', 'Kolkata, India', 'Pune, India', 'Ahmedabad, India',
        'Surat, India', 'Jaipur, India', 'Lucknow, India', 'Kanpur, India',
        'Nagpur, India', 'Indore, India', 'Thane, India', 'Bhopal, India',
        'Visakhapatnam, India', 'Patna, India', 'Vadodara, India', 'Ghaziabad, India',
        'Ludhiana, India', 'Agra, India', 'Nashik, India', 'Faridabad, India',
        'Meerut, India', 'Rajkot, India', 'Kalyan-Dombivli, India', 'Vasai-Virar, India',
        'Varanasi, India', 'Srinagar, India', 'Dhanbad, India', 'Jodhpur, India',
        'Amritsar, India', 'Raipur, India', 'Allahabad, India', 'Coimbatore, India',
        'Jabalpur, India', 'Gwalior, India', 'Vijayawada, India', 'Madurai, India',
        'Guwahati, India', 'Chandigarh, India', 'Hubli-Dharwad, India', 'Amroha, India',
        'Moradabad, India', 'Gurgaon, India', 'Aligarh, India', 'Solapur, India',
        'Ranchi, India', 'Jalandhar, India', 'Tiruchirappalli, India', 'Bhubaneswar, India',
        'Salem, India', 'Warangal, India', 'Mira-Bhayandar, India', 'Thiruvananthapuram, India',
        'Bhiwandi, India', 'Saharanpur, India', 'Gorakhpur, India', 'Guntur, India',
        'Bikaner, India', 'Amravati, India', 'Noida, India', 'Jamshedpur, India',
        'Bhilai, India', 'Cuttack, India', 'Firozabad, India', 'Kochi, India',
        'Bhavnagar, India', 'Dehradun, India', 'Durgapur, India', 'Asansol, India',
        
        # Australia - Major Cities
        'Sydney, Australia', 'Melbourne, Australia', 'Brisbane, Australia', 'Perth, Australia',
        'Adelaide, Australia', 'Gold Coast, Australia', 'Newcastle, Australia', 'Canberra, Australia',
        'Sunshine Coast, Australia', 'Wollongong, Australia', 'Hobart, Australia', 'Geelong, Australia',
        'Townsville, Australia', 'Cairns, Australia', 'Darwin, Australia', 'Toowoomba, Australia',
        'Ballarat, Australia', 'Bendigo, Australia', 'Albury, Australia', 'Launceston, Australia',
        'Mackay, Australia', 'Rockhampton, Australia', 'Bunbury, Australia', 'Bundaberg, Australia',
        'Coffs Harbour, Australia', 'Wagga Wagga, Australia', 'Hervey Bay, Australia', 'Mildura, Australia',
        'Shepparton, Australia', 'Port Macquarie, Australia', 'Gladstone, Australia', 'Tamworth, Australia',
        'Traralgon, Australia', 'Orange, Australia', 'Bowral, Australia', 'Geraldton, Australia',
        'Nowra, Australia', 'Warrnambool, Australia', 'Kalgoorlie, Australia', 'Albany, Australia',
        'Blue Mountains, Australia', 'Lismore, Australia', 'Goulburn, Australia', 'Sunbury, Australia',
        'Frankston, Australia', 'Pakenham, Australia', 'Cranbourne, Australia', 'Dandenong, Australia',
        'Casey, Australia', 'Monash, Australia', 'Whitehorse, Australia', 'Knox, Australia',
        'Maroondah, Australia', 'Boroondara, Australia', 'Glen Eira, Australia', 'Stonnington, Australia',
        'Port Phillip, Australia', 'Yarra, Australia', 'Melbourne, Australia', 'Maribyrnong, Australia',
        'Moonee Valley, Australia', 'Moreland, Australia', 'Darebin, Australia', 'Banyule, Australia',
        'Manningham, Australia', 'Nillumbik, Australia', 'Whittlesea, Australia', 'Hume, Australia',
        'Mitchell, Australia', 'Macedon Ranges, Australia', 'Mount Alexander, Australia', 'Greater Bendigo, Australia',
        
        # Brazil - Major Cities
        'São Paulo, Brazil', 'Rio de Janeiro, Brazil', 'Salvador, Brazil', 'Brasília, Brazil',
        'Fortaleza, Brazil', 'Belo Horizonte, Brazil', 'Manaus, Brazil', 'Curitiba, Brazil',
        'Recife, Brazil', 'Goiânia, Brazil', 'Belém, Brazil', 'Porto Alegre, Brazil',
        'Guarulhos, Brazil', 'Campinas, Brazil', 'Nova Iguaçu, Brazil', 'Maceió, Brazil',
        'São Luís, Brazil', 'Duque de Caxias, Brazil', 'Teresina, Brazil', 'Natal, Brazil',
        'Campo Grande, Brazil', 'Osasco, Brazil', 'Santo André, Brazil', 'João Pessoa, Brazil',
        'Jaboatão dos Guararapes, Brazil', 'Contagem, Brazil', 'São Bernardo do Campo, Brazil', 'Uberlândia, Brazil',
        'Sorocaba, Brazil', 'Aracaju, Brazil', 'Feira de Santana, Brazil', 'Cuiabá, Brazil',
        'Joinville, Brazil', 'Aparecida de Goiânia, Brazil', 'Londrina, Brazil', 'Juiz de Fora, Brazil',
        'Ananindeua, Brazil', 'Niterói, Brazil', 'Belford Roxo, Brazil', 'Campos dos Goytacazes, Brazil',
        'Caxias do Sul, Brazil', 'Vila Velha, Brazil', 'São João de Meriti, Brazil', 'Florianópolis, Brazil',
        'Santos, Brazil', 'Ribeirão Preto, Brazil', 'Diadema, Brazil', 'Carapicuíba, Brazil',
        'Jundiaí, Brazil', 'Piracicaba, Brazil', 'Bauru, Brazil', 'São Vicente, Brazil',
        'Pelotas, Brazil', 'Franca, Brazil', 'Cariacica, Brazil', 'Betim, Brazil',
        'Olinda, Brazil', 'Paulista, Brazil', 'Maringá, Brazil', 'Cascavel, Brazil',
        'Foz do Iguaçu, Brazil', 'Uberaba, Brazil', 'Ribeirão das Neves, Brazil', 'Blumenau, Brazil',
        'Anápolis, Brazil', 'Taubaté, Brazil', 'Petrópolis, Brazil', 'Canoas, Brazil',
        'Volta Redonda, Brazil', 'Caucaia, Brazil', 'Santarém, Brazil', 'Viamão, Brazil',
        'Novo Hamburgo, Brazil', 'Sete Lagoas, Brazil', 'Barueri, Brazil', 'Embu das Artes, Brazil',
        'Colombo, Brazil', 'Cotia, Brazil', 'Jacareí, Brazil', 'Suzano, Brazil',
        'Praia Grande, Brazil', 'Taboão da Serra, Brazil', 'Sumaré, Brazil', 'São José dos Campos, Brazil',
        'Hortolândia, Brazil', 'Camaçari, Brazil', 'Itaquaquecetuba, Brazil', 'Limeira, Brazil',
        'Americana, Brazil', 'Alvorada, Brazil', 'Araraquara, Brazil', 'Marília, Brazil',
        'Presidente Prudente, Brazil', 'Rio Claro, Brazil', 'São Carlos, Brazil', 'Indaiatuba, Brazil',
        'Governador Valadares, Brazil', 'Várzea Grande, Brazil', 'Itapevi, Brazil', 'Gravataí, Brazil',
        
        # European Cities
        'Rome, Italy', 'Milan, Italy', 'Naples, Italy', 'Turin, Italy', 'Palermo, Italy',
        'Genoa, Italy', 'Bologna, Italy', 'Florence, Italy', 'Bari, Italy', 'Catania, Italy',
        'Venice, Italy', 'Verona, Italy', 'Messina, Italy', 'Padua, Italy', 'Trieste, Italy',
        'Madrid, Spain', 'Barcelona, Spain', 'Valencia, Spain', 'Seville, Spain', 'Zaragoza, Spain',
        'Málaga, Spain', 'Murcia, Spain', 'Palma, Spain', 'Las Palmas, Spain', 'Bilbao, Spain',
        'Alicante, Spain', 'Córdoba, Spain', 'Valladolid, Spain', 'Vigo, Spain', 'Gijón, Spain',
        'Amsterdam, Netherlands', 'Rotterdam, Netherlands', 'The Hague, Netherlands', 'Utrecht, Netherlands',
        'Eindhoven, Netherlands', 'Tilburg, Netherlands', 'Groningen, Netherlands', 'Almere, Netherlands',
        'Brussels, Belgium', 'Antwerp, Belgium', 'Ghent, Belgium', 'Charleroi, Belgium', 'Liège, Belgium',
        'Bruges, Belgium', 'Namur, Belgium', 'Leuven, Belgium', 'Mons, Belgium', 'Aalst, Belgium',
        'Zurich, Switzerland', 'Geneva, Switzerland', 'Basel, Switzerland', 'Lausanne, Switzerland',
        'Bern, Switzerland', 'Winterthur, Switzerland', 'Lucerne, Switzerland', 'St. Gallen, Switzerland',
        'Vienna, Austria', 'Graz, Austria', 'Linz, Austria', 'Salzburg, Austria', 'Innsbruck, Austria',
        'Klagenfurt, Austria', 'Villach, Austria', 'Wels, Austria', 'St. Pölten, Austria', 'Dornbirn, Austria',
        'Stockholm, Sweden', 'Göteborg, Sweden', 'Malmö, Sweden', 'Uppsala, Sweden', 'Västerås, Sweden',
        'Örebro, Sweden', 'Linköping, Sweden', 'Helsingborg, Sweden', 'Jönköping, Sweden', 'Norrköping, Sweden',
        'Oslo, Norway', 'Bergen, Norway', 'Trondheim, Norway', 'Stavanger, Norway', 'Drammen, Norway',
        'Fredrikstad, Norway', 'Kristiansand, Norway', 'Sandnes, Norway', 'Tromsø, Norway', 'Sarpsborg, Norway',
        'Copenhagen, Denmark', 'Aarhus, Denmark', 'Odense, Denmark', 'Aalborg, Denmark', 'Esbjerg, Denmark',
        'Randers, Denmark', 'Kolding, Denmark', 'Horsens, Denmark', 'Vejle, Denmark', 'Roskilde, Denmark',
        'Helsinki, Finland', 'Espoo, Finland', 'Tampere, Finland', 'Vantaa, Finland', 'Oulu, Finland',
        'Turku, Finland', 'Jyväskylä, Finland', 'Lahti, Finland', 'Kuopio, Finland', 'Pori, Finland',
        'Warsaw, Poland', 'Krakow, Poland', 'Łódź, Poland', 'Wrocław, Poland', 'Poznań, Poland',
        'Gdańsk, Poland', 'Szczecin, Poland', 'Bydgoszcz, Poland', 'Lublin, Poland', 'Białystok, Poland',
        'Prague, Czech Republic', 'Brno, Czech Republic', 'Ostrava, Czech Republic', 'Plzeň, Czech Republic',
        'Liberec, Czech Republic', 'Olomouc, Czech Republic', 'Ústí nad Labem, Czech Republic', 'České Budějovice, Czech Republic',
        'Budapest, Hungary', 'Debrecen, Hungary', 'Szeged, Hungary', 'Miskolc, Hungary', 'Pécs, Hungary',
        'Győr, Hungary', 'Nyíregyháza, Hungary', 'Kecskemét, Hungary', 'Székesfehérvár, Hungary', 'Szombathely, Hungary',
        'Bucharest, Romania', 'Cluj-Napoca, Romania', 'Timișoara, Romania', 'Iași, Romania', 'Constanța, Romania',
        'Craiova, Romania', 'Brașov, Romania', 'Galați, Romania', 'Ploiești, Romania', 'Oradea, Romania',
        'Sofia, Bulgaria', 'Plovdiv, Bulgaria', 'Varna, Bulgaria', 'Burgas, Bulgaria', 'Ruse, Bulgaria',
        'Stara Zagora, Bulgaria', 'Pleven, Bulgaria', 'Sliven, Bulgaria', 'Dobrich, Bulgaria', 'Shumen, Bulgaria',
        'Athens, Greece', 'Thessaloniki, Greece', 'Patras, Greece', 'Heraklion, Greece', 'Larissa, Greece',
        'Volos, Greece', 'Rhodes, Greece', 'Ioannina, Greece', 'Chania, Greece', 'Chalcis, Greece',
        'Lisbon, Portugal', 'Porto, Portugal', 'Amadora, Portugal', 'Braga, Portugal', 'Setúbal, Portugal',
        'Coimbra, Portugal', 'Funchal, Portugal', 'Almada, Portugal', 'Agualva-Cacém, Portugal', 'Queluz, Portugal',
        'Dublin, Ireland', 'Cork, Ireland', 'Limerick, Ireland', 'Galway, Ireland', 'Waterford, Ireland',
        'Drogheda, Ireland', 'Dundalk, Ireland', 'Swords, Ireland', 'Bray, Ireland', 'Navan, Ireland',
        
        # Asian Cities
        'Seoul, South Korea', 'Busan, South Korea', 'Incheon, South Korea', 'Daegu, South Korea',
        'Daejeon, South Korea', 'Gwangju, South Korea', 'Suwon, South Korea', 'Ulsan, South Korea',
        'Changwon, South Korea', 'Goyang, South Korea', 'Yongin, South Korea', 'Seongnam, South Korea',
        'Bucheon, South Korea', 'Ansan, South Korea', 'Cheongju, South Korea', 'Jeonju, South Korea',
        'Anyang, South Korea', 'Cheonan, South Korea', 'Pohang, South Korea', 'Uijeongbu, South Korea',
        'Jakarta, Indonesia', 'Surabaya, Indonesia', 'Bandung, Indonesia', 'Bekasi, Indonesia',
        'Medan, Indonesia', 'Tangerang, Indonesia', 'Depok, Indonesia', 'Semarang, Indonesia',
        'Palembang, Indonesia', 'Makassar, Indonesia', 'South Tangerang, Indonesia', 'Batam, Indonesia',
        'Bogor, Indonesia', 'Pekanbaru, Indonesia', 'Bandar Lampung, Indonesia', 'Malang, Indonesia',
        'Padang, Indonesia', 'Yogyakarta, Indonesia', 'Denpasar, Indonesia', 'Samarinda, Indonesia',
        'Bangkok, Thailand', 'Nonthaburi, Thailand', 'Pak Kret, Thailand', 'Hat Yai, Thailand',
        'Chiang Mai, Thailand', 'Khon Kaen, Thailand', 'Udon Thani, Thailand', 'Surat Thani, Thailand',
        'Nakhon Ratchasima, Thailand', 'Rayong, Thailand', 'Chonburi, Thailand', 'Lampang, Thailand',
        'Ubon Ratchathani, Thailand', 'Roi Et, Thailand', 'Kanchanaburi, Thailand', 'Trang, Thailand',
        'Hanoi, Vietnam', 'Ho Chi Minh City, Vietnam', 'Haiphong, Vietnam', 'Da Nang, Vietnam',
        'Bien Hoa, Vietnam', 'Hue, Vietnam', 'Nha Trang, Vietnam', 'Can Tho, Vietnam',
        'Rach Gia, Vietnam', 'Qui Nhon, Vietnam', 'Vung Tau, Vietnam', 'Nam Dinh, Vietnam',
        'Phan Thiet, Vietnam', 'Long Xuyen, Vietnam', 'Ha Long, Vietnam', 'Thai Nguyen, Vietnam',
        'Kuala Lumpur, Malaysia', 'George Town, Malaysia', 'Ipoh, Malaysia', 'Shah Alam, Malaysia',
        'Petaling Jaya, Malaysia', 'Klang, Malaysia', 'Johor Bahru, Malaysia', 'Subang Jaya, Malaysia',
        'Seremban, Malaysia', 'Kota Kinabalu, Malaysia', 'Kuching, Malaysia', 'Ampang, Malaysia',
        'Malacca City, Malaysia', 'Alor Setar, Malaysia', 'Tawau, Malaysia', 'Miri, Malaysia',
        'Singapore, Singapore', 'Jurong West, Singapore', 'Woodlands, Singapore', 'Tampines, Singapore',
        'Sengkang, Singapore', 'Hougang, Singapore', 'Yishun, Singapore', 'Bedok, Singapore',
        'Ang Mo Kio, Singapore', 'Toa Payoh, Singapore', 'Choa Chu Kang, Singapore', 'Punggol, Singapore',
        'Pasir Ris, Singapore', 'Clementi, Singapore', 'Jurong East, Singapore', 'Bukit Batok, Singapore',
        'Manila, Philippines', 'Quezon City, Philippines', 'Caloocan, Philippines', 'Las Piñas, Philippines',
        'Makati, Philippines', 'Pasig, Philippines', 'Taguig, Philippines', 'Marikina, Philippines',
        'Muntinlupa, Philippines', 'Parañaque, Philippines', 'Valenzuela, Philippines', 'Malabon, Philippines',
        'Mandaluyong, Philippines', 'San Juan, Philippines', 'Pasay, Philippines', 'Navotas, Philippines',
        'Pateros, Philippines', 'Davao City, Philippines', 'Cebu City, Philippines', 'Zamboanga City, Philippines',
        'Antipolo, Philippines', 'Pasig, Philippines', 'Cagayan de Oro, Philippines', 'Paranaque, Philippines',
        'Dasmarinas, Philippines', 'General Santos, Philippines', 'Bacoor, Philippines', 'Iloilo City, Philippines',
        'Dhaka, Bangladesh', 'Chittagong, Bangladesh', 'Sylhet, Bangladesh', 'Khulna, Bangladesh',
        'Rajshahi, Bangladesh', 'Comilla, Bangladesh', 'Rangpur, Bangladesh', 'Barisal, Bangladesh',
        'Mymensingh, Bangladesh', 'Bogra, Bangladesh', 'Jessore, Bangladesh', 'Narayanganj, Bangladesh',
        'Faridpur, Bangladesh', 'Kishoreganj, Bangladesh', 'Tangail, Bangladesh', 'Pabna, Bangladesh',
        'Karachi, Pakistan', 'Lahore, Pakistan', 'Faisalabad, Pakistan', 'Rawalpindi, Pakistan',
        'Multan, Pakistan', 'Gujranwala, Pakistan', 'Hyderabad, Pakistan', 'Peshawar, Pakistan',
        'Islamabad, Pakistan', 'Quetta, Pakistan', 'Bahawalpur, Pakistan', 'Sargodha, Pakistan',
        'Sialkot, Pakistan', 'Sukkur, Pakistan', 'Larkana, Pakistan', 'Shekhupura, Pakistan',
        'Kathmandu, Nepal', 'Pokhara, Nepal', 'Lalitpur, Nepal', 'Bharatpur, Nepal',
        'Biratnagar, Nepal', 'Birgunj, Nepal', 'Dharan, Nepal', 'Butwal, Nepal',
        'Hetauda, Nepal', 'Janakpur, Nepal', 'Nepalgunj, Nepal', 'Dhangadhi, Nepal',
        'Colombo, Sri Lanka', 'Dehiwala-Mount Lavinia, Sri Lanka', 'Moratuwa, Sri Lanka', 'Negombo, Sri Lanka',
        'Kandy, Sri Lanka', 'Sri Jayawardenepura Kotte, Sri Lanka', 'Galle, Sri Lanka', 'Trincomalee, Sri Lanka',
        'Batticaloa, Sri Lanka', 'Jaffna, Sri Lanka', 'Anuradhapura, Sri Lanka', 'Ratnapura, Sri Lanka',
        
        # Middle Eastern Cities
        'Dubai, UAE', 'Abu Dhabi, UAE', 'Sharjah, UAE', 'Al Ain, UAE', 'Ajman, UAE',
        'Ras Al Khaimah, UAE', 'Fujairah, UAE', 'Umm Al Quwain, UAE', 'Khor Fakkan, UAE', 'Dibba Al-Fujairah, UAE',
        'Riyadh, Saudi Arabia', 'Jeddah, Saudi Arabia', 'Mecca, Saudi Arabia', 'Medina, Saudi Arabia',
        'Dammam, Saudi Arabia', 'Khobar, Saudi Arabia', 'Tabuk, Saudi Arabia', 'Buraidah, Saudi Arabia',
        'Khamis Mushait, Saudi Arabia', 'Hail, Saudi Arabia', 'Hofuf, Saudi Arabia', 'Mubarak Al-Kabeer, Saudi Arabia',
        'Doha, Qatar', 'Al Rayyan, Qatar', 'Umm Salal, Qatar', 'Al Wakrah, Qatar',
        'Al Khor, Qatar', 'Dukhan, Qatar', 'Lusail, Qatar', 'Mesaieed, Qatar',
        'Kuwait City, Kuwait', 'Hawally, Kuwait', 'Salmiya, Kuwait', 'Sabah Al-Salem, Kuwait',
        'Abdullah Al-Mubarak, Kuwait', 'Mishref, Kuwait', 'Bayan, Kuwait', 'Mangaf, Kuwait',
        'Manama, Bahrain', 'Riffa, Bahrain', 'Muharraq, Bahrain', 'Hamad Town, Bahrain',
        'A\'ali, Bahrain', 'Isa Town, Bahrain', 'Sitra, Bahrain', 'Budaiya, Bahrain',
        'Muscat, Oman', 'Seeb, Oman', 'Salalah, Oman', 'Bawshar, Oman',
        'Sohar, Oman', 'Sur, Oman', 'Ibri, Oman', 'Saham, Oman',
        'Tehran, Iran', 'Mashhad, Iran', 'Isfahan, Iran', 'Karaj, Iran',
        'Shiraz, Iran', 'Tabriz, Iran', 'Qom, Iran', 'Ahvaz, Iran',
        'Kermanshah, Iran', 'Urmia, Iran', 'Rasht, Iran', 'Zahedan, Iran',
        'Ankara, Turkey', 'Istanbul, Turkey', 'Izmir, Turkey', 'Bursa, Turkey',
        'Adana, Turkey', 'Gaziantep, Turkey', 'Konya, Turkey', 'Antalya, Turkey',
        'Kayseri, Turkey', 'Mersin, Turkey', 'Eskisehir, Turkey', 'Diyarbakir, Turkey',
        'Tel Aviv, Israel', 'Jerusalem, Israel', 'Haifa, Israel', 'Rishon LeZion, Israel',
        'Petah Tikva, Israel', 'Ashdod, Israel', 'Netanya, Israel', 'Be\'er Sheva, Israel',
        'Bnei Brak, Israel', 'Holon, Israel', 'Ramat Gan, Israel', 'Ashkelon, Israel',
        'Amman, Jordan', 'Zarqa, Jordan', 'Irbid, Jordan', 'Russeifa, Jordan',
        'Wadi as-Sir, Jordan', 'Ajloun, Jordan', 'Madaba, Jordan', 'Aqaba, Jordan',
        'Beirut, Lebanon', 'Tripoli, Lebanon', 'Sidon, Lebanon', 'Tyre, Lebanon',
        'Nabatieh, Lebanon', 'Jounieh, Lebanon', 'Zahle, Lebanon', 'Baalbek, Lebanon',
        'Damascus, Syria', 'Aleppo, Syria', 'Homs, Syria', 'Latakia, Syria',
        'Hama, Syria', 'Deir ez-Zor, Syria', 'Raqqa, Syria', 'Daraa, Syria',
        'Baghdad, Iraq', 'Basra, Iraq', 'Mosul, Iraq', 'Erbil, Iraq',
        'Sulaymaniyah, Iraq', 'Najaf, Iraq', 'Karbala, Iraq', 'Kirkuk, Iraq',
        
        # African Cities
        'Cairo, Egypt', 'Alexandria, Egypt', 'Giza, Egypt', 'Shubra El Kheima, Egypt',
        'Port Said, Egypt', 'Suez, Egypt', 'Luxor, Egypt', 'Mansoura, Egypt',
        'El Mahalla El Kubra, Egypt', 'Tanta, Egypt', 'Asyut, Egypt', 'Ismailia, Egypt',
        'Fayyum, Egypt', 'Zagazig, Egypt', 'Aswan, Egypt', 'Damietta, Egypt',
        'Lagos, Nigeria', 'Kano, Nigeria', 'Ibadan, Nigeria', 'Abuja, Nigeria',
        'Port Harcourt, Nigeria', 'Benin City, Nigeria', 'Maiduguri, Nigeria', 'Zaria, Nigeria',
        'Aba, Nigeria', 'Jos, Nigeria', 'Ilorin, Nigeria', 'Oyo, Nigeria',
        'Enugu, Nigeria', 'Abeokuta, Nigeria', 'Kaduna, Nigeria', 'Ogbomoso, Nigeria',
        'Johannesburg, South Africa', 'Cape Town, South Africa', 'Durban, South Africa', 'Pretoria, South Africa',
        'Soweto, South Africa', 'Benoni, South Africa', 'Tembisa, South Africa', 'East London, South Africa',
        'Vereeniging, South Africa', 'Bloemfontein, South Africa', 'Boksburg, South Africa', 'Welkom, South Africa',
        'Newcastle, South Africa', 'Krugersdorp, South Africa', 'Diepsloot, South Africa', 'Botshabelo, South Africa',
        'Casablanca, Morocco', 'Rabat, Morocco', 'Fez, Morocco', 'Marrakech, Morocco',
        'Agadir, Morocco', 'Tangier, Morocco', 'Meknes, Morocco', 'Oujda, Morocco',
        'Kenitra, Morocco', 'Tetouan, Morocco', 'Safi, Morocco', 'Mohammedia, Morocco',
        'Algiers, Algeria', 'Oran, Algeria', 'Constantine, Algeria', 'Batna, Algeria',
        'Djelfa, Algeria', 'Sétif, Algeria', 'Annaba, Algeria', 'Sidi Bel Abbès, Algeria',
        'Biskra, Algeria', 'Tébessa, Algeria', 'El Khroub, Algeria', 'Tiaret, Algeria',
        'Tunis, Tunisia', 'Sfax, Tunisia', 'Sousse, Tunisia', 'Ettadhamen, Tunisia',
        'Kairouan, Tunisia', 'Bizerte, Tunisia', 'Gabès, Tunisia', 'Ariana, Tunisia',
        'Gafsa, Tunisia', 'Monastir, Tunisia', 'Ben Arous, Tunisia', 'Kasserine, Tunisia',
        'Tripoli, Libya', 'Benghazi, Libya', 'Misrata, Libya', 'Tarhuna, Libya',
        'Al Bayda, Libya', 'Ajdabiya, Libya', 'Tobruk, Libya', 'Sabha, Libya',
        'Zawiya, Libya', 'Zliten, Libya', 'Derna, Libya', 'Sirte, Libya',
        'Khartoum, Sudan', 'Omdurman, Sudan', 'Khartoum North, Sudan', 'Kassala, Sudan',
        'Obeid, Sudan', 'Nyala, Sudan', 'Port Sudan, Sudan', 'Gedaref, Sudan',
        'Wad Medani, Sudan', 'El Fasher, Sudan', 'Kosti, Sudan', 'Sennar, Sudan',
        'Addis Ababa, Ethiopia', 'Dire Dawa, Ethiopia', 'Adama, Ethiopia', 'Gondar, Ethiopia',
        'Mek\'ele, Ethiopia', 'Awasa, Ethiopia', 'Bahir Dar, Ethiopia', 'Dessie, Ethiopia',
        'Jimma, Ethiopia', 'Jijiga, Ethiopia', 'Shashamane, Ethiopia', 'Nekemte, Ethiopia',
        'Nairobi, Kenya', 'Mombasa, Kenya', 'Nakuru, Kenya', 'Eldoret, Kenya',
        'Kisumu, Kenya', 'Thika, Kenya', 'Malindi, Kenya', 'Kitale, Kenya',
        'Machakos, Kenya', 'Meru, Kenya', 'Kericho, Kenya', 'Nyeri, Kenya',
        'Kampala, Uganda', 'Gulu, Uganda', 'Lira, Uganda', 'Mbarara, Uganda',
        'Jinja, Uganda', 'Busia, Uganda', 'Mbale, Uganda', 'Mukono, Uganda',
        'Kasese, Uganda', 'Masaka, Uganda', 'Entebbe, Uganda', 'Arua, Uganda',
        'Dar es Salaam, Tanzania', 'Mwanza, Tanzania', 'Arusha, Tanzania', 'Dodoma, Tanzania',
        'Mbeya, Tanzania', 'Morogoro, Tanzania', 'Tanga, Tanzania', 'Kahama, Tanzania',
        'Tabora, Tanzania', 'Zanzibar City, Tanzania', 'Kigoma, Tanzania', 'Mtwara, Tanzania',
        'Lusaka, Zambia', 'Ndola, Zambia', 'Kitwe, Zambia', 'Kabwe, Zambia',
        'Chingola, Zambia', 'Mufulira, Zambia', 'Livingstone, Zambia', 'Luanshya, Zambia',
        'Kasama, Zambia', 'Chipata, Zambia', 'Mazabuka, Zambia', 'Chililabombwe, Zambia',
        'Harare, Zimbabwe', 'Bulawayo, Zimbabwe', 'Chitungwiza, Zimbabwe', 'Mutare, Zimbabwe',
        'Gweru, Zimbabwe', 'Epworth, Zimbabwe', 'Kwekwe, Zimbabwe', 'Kadoma, Zimbabwe',
        'Masvingo, Zimbabwe', 'Chinhoyi, Zimbabwe', 'Marondera, Zimbabwe', 'Ruwa, Zimbabwe',
        'Maputo, Mozambique', 'Matola, Mozambique', 'Beira, Mozambique', 'Nampula, Mozambique',
        'Chimoio, Mozambique', 'Nacala, Mozambique', 'Quelimane, Mozambique', 'Tete, Mozambique',
        'Xai-Xai, Mozambique', 'Lichinga, Mozambique', 'Pemba, Mozambique', 'Inhambane, Mozambique',
        'Gaborone, Botswana', 'Francistown, Botswana', 'Molepolole, Botswana', 'Maun, Botswana',
        'Selebi-Phikwe, Botswana', 'Serowe, Botswana', 'Kanye, Botswana', 'Mochudi, Botswana',
        'Mahalapye, Botswana', 'Palapye, Botswana', 'Tlokweng, Botswana', 'Gabane, Botswana',
        'Windhoek, Namibia', 'Rundu, Namibia', 'Walvis Bay, Namibia', 'Oshakati, Namibia',
        'Swakopmund, Namibia', 'Katima Mulilo, Namibia', 'Grootfontein, Namibia', 'Rehoboth, Namibia',
        'Otjiwarongo, Namibia', 'Okahandja, Namibia', 'Ondangwa, Namibia', 'Ongwediva, Namibia',
        'Maseru, Lesotho', 'Teyateyaneng, Lesotho', 'Leribe, Lesotho', 'Mafeteng, Lesotho',
        'Hlotse, Lesotho', 'Mohale\'s Hoek, Lesotho', 'Quthing, Lesotho', 'Qacha\'s Nek, Lesotho',
        'Butha-Buthe, Lesotho', 'Mokhotlong, Lesotho', 'Thaba-Tseka, Lesotho', 'Peka, Lesotho',
        'Mbabane, Swaziland', 'Manzini, Swaziland', 'Lobamba, Swaziland', 'Nhlangano, Swaziland',
        'Piggs Peak, Swaziland', 'Siteki, Swaziland', 'Hluti, Swaziland', 'Mankayane, Swaziland',
        'Kigali, Rwanda', 'Butare, Rwanda', 'Gitarama, Rwanda', 'Musanze, Rwanda',
        'Gisenyi, Rwanda', 'Byumba, Rwanda', 'Cyangugu, Rwanda', 'Kibungo, Rwanda',
        'Bujumbura, Burundi', 'Gitega, Burundi', 'Muyinga, Burundi', 'Ngozi, Burundi',
        'Ruyigi, Burundi', 'Kayanza, Burundi', 'Bururi, Burundi', 'Muramvya, Burundi',
        'Djibouti City, Djibouti', 'Ali Sabieh, Djibouti', 'Dikhil, Djibouti', 'Tadjourah, Djibouti',
        'Obock, Djibouti', 'Arta, Djibouti', 'Holhol, Djibouti', 'Yoboki, Djibouti',
        'Asmara, Eritrea', 'Assab, Eritrea', 'Massawa, Eritrea', 'Mendefera, Eritrea',
        'Barentu, Eritrea', 'Keren, Eritrea', 'Ak\'ordat, Eritrea', 'Adi Keyh, Eritrea',
        'Mogadishu, Somalia', 'Hargeisa, Somalia', 'Bosaso, Somalia', 'Kismayo, Somalia',
        'Merca, Somalia', 'Galcaio, Somalia', 'Berbera, Somalia', 'Baidoa, Somalia',
        'Abidjan, Côte d\'Ivoire', 'Yamoussoukro, Côte d\'Ivoire', 'Bouaké, Côte d\'Ivoire', 'Daloa, Côte d\'Ivoire',
        'San-Pédro, Côte d\'Ivoire', 'Korhogo, Côte d\'Ivoire', 'Man, Côte d\'Ivoire', 'Gagnoa, Côte d\'Ivoire',
        'Accra, Ghana', 'Kumasi, Ghana', 'Tamale, Ghana', 'Takoradi, Ghana',
        'Tema, Ghana', 'Cape Coast, Ghana', 'Obuasi, Ghana', 'Tarkwa, Ghana',
        'Sunyani, Ghana', 'Koforidua, Ghana', 'Ho, Ghana', 'Techiman, Ghana',
        'Ouagadougou, Burkina Faso', 'Bobo-Dioulasso, Burkina Faso', 'Koudougou, Burkina Faso', 'Ouahigouya, Burkina Faso',
        'Banfora, Burkina Faso', 'Dédougou, Burkina Faso', 'Kaya, Burkina Faso', 'Tenkodogo, Burkina Faso',
        'Bamako, Mali', 'Sikasso, Mali', 'Mopti, Mali', 'Koutiala, Mali',
        'Ségou, Mali', 'Kayes, Mali', 'Gao, Mali', 'Tombouctou, Mali',
        'Dakar, Senegal', 'Touba, Senegal', 'Thiès, Senegal', 'Kaolack, Senegal',
        'Saint-Louis, Senegal', 'Ziguinchor, Senegal', 'Diourbel, Senegal', 'Tambacounda, Senegal',
        'Conakry, Guinea', 'Nzérékoré, Guinea', 'Kankan, Guinea', 'Kindia, Guinea',
        'Labe, Guinea', 'Mamou, Guinea', 'Boke, Guinea', 'Faranah, Guinea',
        'Freetown, Sierra Leone', 'Bo, Sierra Leone', 'Kenema, Sierra Leone', 'Koidu, Sierra Leone',
        'Makeni, Sierra Leone', 'Lunsar, Sierra Leone', 'Port Loko, Sierra Leone', 'Waterloo, Sierra Leone',
        'Monrovia, Liberia', 'Gbarnga, Liberia', 'Kakata, Liberia', 'Bensonville, Liberia',
        'Harper, Liberia', 'Voinjama, Liberia', 'Zwedru, Liberia', 'Robertsport, Liberia',
        'Lomé, Togo', 'Sokodé, Togo', 'Kara, Togo', 'Palimé, Togo',
        'Atakpamé, Togo', 'Bassar, Togo', 'Tsévié, Togo', 'Aného, Togo',
        'Cotonou, Benin', 'Porto-Novo, Benin', 'Parakou, Benin', 'Djougou, Benin',
        'Bohicon, Benin', 'Kandi, Benin', 'Ouidah, Benin', 'Abomey, Benin',
        'Niamey, Niger', 'Zinder, Niger', 'Maradi, Niger', 'Agadez, Niger',
        'Tahoua, Niger', 'Dosso, Niger', 'Tillabéri, Niger', 'Diffa, Niger',
        'N\'Djamena, Chad', 'Moundou, Chad', 'Sarh, Chad', 'Abéché, Chad',
        'Kélo, Chad', 'Koumra, Chad', 'Pala, Chad', 'Am Timan, Chad',
        'Bangui, Central African Republic', 'Bimbo, Central African Republic', 'Berbérati, Central African Republic',
        'Carnot, Central African Republic', 'Bambari, Central African Republic', 'Bouar, Central African Republic',
        'Bossangoa, Central African Republic', 'Bria, Central African Republic',
        'Yaoundé, Cameroon', 'Douala, Cameroon', 'Garoua, Cameroon', 'Bamenda, Cameroon',
        'Bafoussam, Cameroon', 'Maroua, Cameroon', 'Ngaoundéré, Cameroon', 'Bertoua, Cameroon',
        'Edéa, Cameroon', 'Loum, Cameroon', 'Kumba, Cameroon', 'Nkongsamba, Cameroon',
        'Malabo, Equatorial Guinea', 'Bata, Equatorial Guinea', 'Ebebiyin, Equatorial Guinea', 'Aconibe, Equatorial Guinea',
        'Añisoc, Equatorial Guinea', 'Luba, Equatorial Guinea', 'Evinayong, Equatorial Guinea', 'Mongomo, Equatorial Guinea',
        'Libreville, Gabon', 'Port-Gentil, Gabon', 'Franceville, Gabon', 'Oyem, Gabon',
        'Moanda, Gabon', 'Mouila, Gabon', 'Lambaréné, Gabon', 'Tchibanga, Gabon',
        'Brazzaville, Republic of the Congo', 'Pointe-Noire, Republic of the Congo', 'Dolisie, Republic of the Congo',
        'Nkayi, Republic of the Congo', 'Mossendjo, Republic of the Congo', 'Kinkala, Republic of the Congo',
        'Impfondo, Republic of the Congo', 'Ouesso, Republic of the Congo',
        'Kinshasa, Democratic Republic of the Congo', 'Lubumbashi, Democratic Republic of the Congo',
        'Mbuji-Mayi, Democratic Republic of the Congo', 'Kisangani, Democratic Republic of the Congo',
        'Masina, Democratic Republic of the Congo', 'Kananga, Democratic Republic of the Congo',
        'Likasi, Democratic Republic of the Congo', 'Kolwezi, Democratic Republic of the Congo',
        'Tshikapa, Democratic Republic of the Congo', 'Beni, Democratic Republic of the Congo',
        'Bukavu, Democratic Republic of the Congo', 'Mwene-Ditu, Democratic Republic of the Congo',
        'Kikwit, Democratic Republic of the Congo', 'Mbandaka, Democratic Republic of the Congo',
        'Matadi, Democratic Republic of the Congo', 'Uvira, Democratic Republic of the Congo',
        'Luanda, Angola', 'Huambo, Angola', 'Lobito, Angola', 'Benguela, Angola',
        'Kuito, Angola', 'Lubango, Angola', 'Malanje, Angola', 'Namibe, Angola',
        'Soyo, Angola', 'Cabinda, Angola', 'Uíge, Angola', 'Saurimo, Angola',
        
        # Latin American Cities
        'Mexico City, Mexico', 'Guadalajara, Mexico', 'Monterrey, Mexico', 'Puebla, Mexico',
        'Tijuana, Mexico', 'León, Mexico', 'Juárez, Mexico', 'Torreón, Mexico',
        'Querétaro, Mexico', 'San Luis Potosí, Mexico', 'Mérida, Mexico', 'Mexicali, Mexico',
        'Aguascalientes, Mexico', 'Acapulco, Mexico', 'Cuernavaca, Mexico', 'Saltillo, Mexico',
        'Villahermosa, Mexico', 'Culiacán, Mexico', 'Cancún, Mexico', 'Veracruz, Mexico',
        'Buenos Aires, Argentina', 'Córdoba, Argentina', 'Rosario, Argentina', 'Mendoza, Argentina',
        'San Miguel de Tucumán, Argentina', 'La Plata, Argentina', 'Mar del Plata, Argentina',
        'Salta, Argentina', 'Santa Fe, Argentina', 'San Juan, Argentina', 'Resistencia, Argentina',
        'Santiago del Estero, Argentina', 'Corrientes, Argentina', 'Posadas, Argentina',
        'Neuquén, Argentina', 'Bahía Blanca, Argentina', 'Paraná, Argentina', 'Formosa, Argentina',
        'San Luis, Argentina', 'Catamarca, Argentina', 'La Rioja, Argentina', 'Río Gallegos, Argentina',
        'Ushuaia, Argentina', 'Santa Rosa, Argentina', 'San Salvador de Jujuy, Argentina', 'Rawson, Argentina',
        'Santiago, Chile', 'Valparaíso, Chile', 'Concepción, Chile', 'La Serena, Chile',
        'Antofagasta, Chile', 'Temuco, Chile', 'Rancagua, Chile', 'Talca, Chile',
        'Arica, Chile', 'Chillán, Chile', 'Iquique, Chile', 'Los Ángeles, Chile',
        'Puerto Montt, Chile', 'Calama, Chile', 'Copiapó, Chile', 'Osorno, Chile',
        'Quillota, Chile', 'Valdivia, Chile', 'Punta Arenas, Chile', 'San Antonio, Chile',
        'Coquimbo, Chile', 'Ovalle, Chile', 'Curicó, Chile', 'Linares, Chile',
        'Bogotá, Colombia', 'Medellín, Colombia', 'Cali, Colombia', 'Barranquilla, Colombia',
        'Cartagena, Colombia', 'Cúcuta, Colombia', 'Bucaramanga, Colombia', 'Pereira, Colombia',
        'Santa Marta, Colombia', 'Ibagué, Colombia', 'Pasto, Colombia', 'Manizales, Colombia',
        'Neiva, Colombia', 'Soledad, Colombia', 'Armenia, Colombia', 'Bello, Colombia',
        'Villavicencio, Colombia', 'Valledupar, Colombia', 'Montería, Colombia', 'Palmira, Colombia',
        'Buenaventura, Colombia', 'Floridablanca, Colombia', 'Sincelejo, Colombia', 'Popayán, Colombia',
        'Lima, Peru', 'Arequipa, Peru', 'Trujillo, Peru', 'Chiclayo, Peru',
        'Piura, Peru', 'Iquitos, Peru', 'Cusco, Peru', 'Chimbote, Peru',
        'Huancayo, Peru', 'Tacna, Peru', 'Juliaca, Peru', 'Ica, Peru',
        'Sullana, Peru', 'Ayacucho, Peru', 'Chincha Alta, Peru', 'Huánuco, Peru',
        'Pucallpa, Peru', 'Tarapoto, Peru', 'Puno, Peru', 'Tumbes, Peru',
        'Talara, Peru', 'Jaén, Peru', 'Huaraz, Peru', 'Abancay, Peru',
        'Quito, Ecuador', 'Guayaquil, Ecuador', 'Cuenca, Ecuador', 'Santo Domingo, Ecuador',
        'Machala, Ecuador', 'Durán, Ecuador', 'Manta, Ecuador', 'Portoviejo, Ecuador',
        'Ambato, Ecuador', 'Riobamba, Ecuador', 'Loja, Ecuador', 'Esmeraldas, Ecuador',
        'Ibarra, Ecuador', 'Milagro, Ecuador', 'Quevedo, Ecuador', 'Babahoyo, Ecuador',
        'La Paz, Bolivia', 'Santa Cruz de la Sierra, Bolivia', 'Cochabamba, Bolivia',
        'Oruro, Bolivia', 'Sucre, Bolivia', 'Tarija, Bolivia', 'Potosí, Bolivia',
        'Sacaba, Bolivia', 'Montero, Bolivia', 'Trinidad, Bolivia', 'Riberalta, Bolivia',
        'Asunción, Paraguay', 'Ciudad del Este, Paraguay', 'San Lorenzo, Paraguay',
        'Luque, Paraguay', 'Capiatá, Paraguay', 'Lambaré, Paraguay', 'Fernando de la Mora, Paraguay',
        'Limpio, Paraguay', 'Ñemby, Paraguay', 'Encarnación, Paraguay', 'Mariano Roque Alonso, Paraguay',
        'Montevideo, Uruguay', 'Salto, Uruguay', 'Paysandú, Uruguay', 'Las Piedras, Uruguay',
        'Rivera, Uruguay', 'Maldonado, Uruguay', 'Tacuarembó, Uruguay', 'Melo, Uruguay',
        'Mercedes, Uruguay', 'Artigas, Uruguay', 'Minas, Uruguay', 'San José de Mayo, Uruguay',
        'Durazno, Uruguay', 'Florida, Uruguay', 'Barros Blancos, Uruguay', 'Ciudad de la Costa, Uruguay',
        'Georgetown, Guyana', 'Linden, Guyana', 'New Amsterdam, Guyana', 'Anna Regina, Guyana',
        'Bartica, Guyana', 'Skeldon, Guyana', 'Rosignol, Guyana', 'Parika, Guyana',
        'Paramaribo, Suriname', 'Lelydorp, Suriname', 'Nieuw Nickerie, Suriname', 'Moengo, Suriname',
        'Nieuw Amsterdam, Suriname', 'Mariënburg, Suriname', 'Wageningen, Suriname', 'Albina, Suriname',
        'Cayenne, French Guiana', 'Saint-Laurent-du-Maroni, French Guiana', 'Kourou, French Guiana',
        'Matoury, French Guiana', 'Remire-Montjoly, French Guiana', 'Mana, French Guiana',
        'Caracas, Venezuela', 'Maracaibo, Venezuela', 'Valencia, Venezuela', 'Barquisimeto, Venezuela',
        'Maracay, Venezuela', 'Ciudad Guayana, Venezuela', 'Barcelona, Venezuela', 'Maturín, Venezuela',
        'San Cristóbal, Venezuela', 'Ciudad Bolívar, Venezuela', 'Cumana, Venezuela', 'Mérida, Venezuela',
        'Cabimas, Venezuela', 'Turmero, Venezuela', 'Barinas, Venezuela', 'Coro, Venezuela',
        'Punto Fijo, Venezuela', 'Los Teques, Venezuela', 'Guanare, Venezuela', 'Acarigua, Venezuela',
        
        # Caribbean Cities
        'Havana, Cuba', 'Santiago de Cuba, Cuba', 'Camagüey, Cuba', 'Holguín, Cuba',
        'Guantánamo, Cuba', 'Santa Clara, Cuba', 'Las Tunas, Cuba', 'Bayamo, Cuba',
        'Cienfuegos, Cuba', 'Pinar del Río, Cuba', 'Matanzas, Cuba', 'Ciego de Ávila, Cuba',
        'Sancti Spíritus, Cuba', 'Manzanillo, Cuba', 'Cardenas, Cuba', 'Palmira, Cuba',
        'Santo Domingo, Dominican Republic', 'Santiago, Dominican Republic', 'La Vega, Dominican Republic',
        'San Pedro de Macorís, Dominican Republic', 'La Romana, Dominican Republic', 'Bella Vista, Dominican Republic',
        'San Francisco de Macorís, Dominican Republic', 'Puerto Plata, Dominican Republic', 'San Juan, Dominican Republic',
        'Bajos de Haina, Dominican Republic', 'Bonao, Dominican Republic', 'Cotuí, Dominican Republic',
        'Port-au-Prince, Haiti', 'Cap-Haïtien, Haiti', 'Delmas, Haiti', 'Pétion-Ville, Haiti',
        'Carrefour, Haiti', 'Gonaïves, Haiti', 'Les Cayes, Haiti', 'Jacmel, Haiti',
        'Jérémie, Haiti', 'Fort-Liberté, Haiti', 'Hinche, Haiti', 'Limbé, Haiti',
        'Kingston, Jamaica', 'Spanish Town, Jamaica', 'Portmore, Jamaica', 'Montego Bay, Jamaica',
        'May Pen, Jamaica', 'Mandeville, Jamaica', 'Old Harbour, Jamaica', 'Savanna-la-Mar, Jamaica',
        'Ocho Rios, Jamaica', 'Linstead, Jamaica', 'Half Way Tree, Jamaica', 'Santa Cruz, Jamaica',
        'San Juan, Puerto Rico', 'Bayamón, Puerto Rico', 'Carolina, Puerto Rico', 'Ponce, Puerto Rico',
        'Caguas, Puerto Rico', 'Guaynabo, Puerto Rico', 'Arecibo, Puerto Rico', 'Toa Baja, Puerto Rico',
        'Mayagüez, Puerto Rico', 'Trujillo Alto, Puerto Rico', 'Toa Alta, Puerto Rico', 'Levittown, Puerto Rico',
        'Bridgetown, Barbados', 'Speightstown, Barbados', 'Oistins, Barbados', 'Bathsheba, Barbados',
        'Holetown, Barbados', 'Crane, Barbados', 'Six Cross Roads, Barbados', 'Blackmans, Barbados',
        'St. George\'s, Grenada', 'Gouyave, Grenada', 'Grenville, Grenada', 'Hillsborough, Grenada',
        'Sauteurs, Grenada', 'Victoria, Grenada', 'Charlestown, Grenada', 'Tivoli, Grenada',
        'Port of Spain, Trinidad and Tobago', 'San Fernando, Trinidad and Tobago', 'Arima, Trinidad and Tobago',
        'Point Fortin, Trinidad and Tobago', 'Chaguanas, Trinidad and Tobago', 'Laventille, Trinidad and Tobago',
        'Penal, Trinidad and Tobago', 'Marabella, Trinidad and Tobago', 'Tunapuna, Trinidad and Tobago',
        'Scarborough, Trinidad and Tobago', 'Diego Martin, Trinidad and Tobago', 'Siparia, Trinidad and Tobago',
        'Castries, Saint Lucia', 'Bisée, Saint Lucia', 'Vieux Fort, Saint Lucia', 'Micoud, Saint Lucia',
        'Soufrière, Saint Lucia', 'Dennery, Saint Lucia', 'Gros Islet, Saint Lucia', 'Choiseul, Saint Lucia',
        'Kingstown, Saint Vincent and the Grenadines', 'Georgetown, Saint Vincent and the Grenadines',
        'Chateaubelair, Saint Vincent and the Grenadines', 'Barrouallie, Saint Vincent and the Grenadines',
        'Port Elizabeth, Saint Vincent and the Grenadines', 'Biabou, Saint Vincent and the Grenadines',
        'Layou, Saint Vincent and the Grenadines', 'Calliaqua, Saint Vincent and the Grenadines',
        'St. John\'s, Antigua and Barbuda', 'All Saints, Antigua and Barbuda', 'Liberta, Antigua and Barbuda',
        'Potter\'s Village, Antigua and Barbuda', 'Bolans, Antigua and Barbuda', 'Swetes, Antigua and Barbuda',
        'Codrington, Antigua and Barbuda', 'Parham, Antigua and Barbuda', 'Bendals, Antigua and Barbuda',
        'English Harbour, Antigua and Barbuda', 'Falmouth, Antigua and Barbuda', 'Old Road, Antigua and Barbuda',
        'Basseterre, Saint Kitts and Nevis', 'Charlestown, Saint Kitts and Nevis', 'Dieppe Bay Town, Saint Kitts and Nevis',
        'Oranjestad, Saint Kitts and Nevis', 'Sandy Point Town, Saint Kitts and Nevis', 'Cayon, Saint Kitts and Nevis',
        'Gingerland, Saint Kitts and Nevis', 'St. Paul\'s, Saint Kitts and Nevis', 'Newton Ground, Saint Kitts and Nevis',
        'Tabernacle, Saint Kitts and Nevis', 'Mansion, Saint Kitts and Nevis', 'Fig Tree, Saint Kitts and Nevis',
        'Roseau, Dominica', 'Portsmouth, Dominica', 'Marigot, Dominica', 'Berekua, Dominica',
        'Mahaut, Dominica', 'St. Joseph, Dominica', 'Canefield, Dominica', 'Soufriere, Dominica',
        'Salisbury, Dominica', 'Calibishie, Dominica', 'Scotts Head, Dominica', 'Massacre, Dominica',
        'Nassau, Bahamas', 'Lucaya, Bahamas', 'Freeport, Bahamas', 'West End, Bahamas',
        'Cooper\'s Town, Bahamas', 'Marsh Harbour, Bahamas', 'High Rock, Bahamas', 'Alice Town, Bahamas',
        'Cockburn Town, Bahamas', 'Dunmore Town, Bahamas', 'Fresh Creek, Bahamas', 'Nicholls Town, Bahamas',
        'George Town, Cayman Islands', 'West Bay, Cayman Islands', 'Bodden Town, Cayman Islands',
        'North Side, Cayman Islands', 'East End, Cayman Islands', 'Savannah, Cayman Islands',
        'Newlands, Cayman Islands', 'Prospect, Cayman Islands', 'Gun Bay, Cayman Islands',
        'Spot Bay, Cayman Islands', 'West End, Cayman Islands', 'The Bluff, Cayman Islands',
        'Road Town, British Virgin Islands', 'Spanish Town, British Virgin Islands', 'The Valley, British Virgin Islands',
        'West End, British Virgin Islands', 'East End, British Virgin Islands', 'Cane Garden Bay, British Virgin Islands',
        'Long Bay, British Virgin Islands', 'Capoons Bay, British Virgin Islands', 'Carrot Bay, British Virgin Islands',
        'Josiahs Bay, British Virgin Islands', 'Maya Cove, British Virgin Islands', 'Brewers Bay, British Virgin Islands',
        'Charlotte Amalie, US Virgin Islands', 'Christiansted, US Virgin Islands', 'Frederiksted, US Virgin Islands',
        'Cruz Bay, US Virgin Islands', 'Anna\'s Retreat, US Virgin Islands', 'Altona, US Virgin Islands',
        'Kingshill, US Virgin Islands', 'Estate Thomas, US Virgin Islands', 'Frenchtown, US Virgin Islands',
        'Coral Bay, US Virgin Islands', 'Mandahl, US Virgin Islands', 'Bovoni, US Virgin Islands',
        'Willemstad, Curaçao', 'Barber, Curaçao', 'Sint Michiel, Curaçao', 'Westpunt, Curaçao',
        'Soto, Curaçao', 'Dorp Sint Willibrordus, Curaçao', 'Nieuw Nederland, Curaçao', 'Saliña, Curaçao',
        'Mahuma, Curaçao', 'Jandoret, Curaçao', 'Sabana Westpunt, Curaçao', 'Barber Kompas, Curaçao',
        'Oranjestad, Aruba', 'San Nicolaas, Aruba', 'Noord, Aruba', 'Santa Cruz, Aruba',
        'Paradera, Aruba', 'Tanki Leendert, Aruba', 'Savaneta, Aruba', 'Pos Chiquito, Aruba',
        'Bushiri, Aruba', 'Brasil, Aruba', 'Macuarima, Aruba', 'Piedra Plat, Aruba',
        'Philipsburg, Sint Maarten', 'Simpson Bay, Sint Maarten', 'Cole Bay, Sint Maarten',
        'Cay Bay, Sint Maarten', 'Lowlands, Sint Maarten', 'Maho, Sint Maarten',
        'Oyster Pond, Sint Maarten', 'Dawn Beach, Sint Maarten', 'Guana Bay, Sint Maarten',
        'Cupecoy, Sint Maarten', 'Pelican Key, Sint Maarten', 'Beacon Hill, Sint Maarten',
        'Marigot, Saint Martin', 'Grand Case, Saint Martin', 'Quartier d\'Orleans, Saint Martin',
        'Sandy Ground, Saint Martin', 'Colombier, Saint Martin', 'Friar\'s Bay, Saint Martin',
        'Cul de Sac, Saint Martin', 'Rambaud, Saint Martin', 'Hope Estate, Saint Martin',
        'Baie Nettle, Saint Martin', 'Concordia, Saint Martin', 'Orient Bay, Saint Martin',
        'Gustavia, Saint Barthélemy', 'Lorient, Saint Barthélemy', 'Saint-Jean, Saint Barthélemy',
        'Corossol, Saint Barthélemy', 'Flamands, Saint Barthélemy', 'Grand Cul-de-Sac, Saint Barthélemy',
        'Petit Cul-de-Sac, Saint Barthélemy', 'Public, Saint Barthélemy', 'Toiny, Saint Barthélemy',
        'Vitet, Saint Barthélemy', 'Gouverneur, Saint Barthélemy', 'Saline, Saint Barthélemy',
        'Fort-de-France, Martinique', 'Le Lamentin, Martinique', 'Le Robert, Martinique',
        'Sainte-Marie, Martinique', 'Le François, Martinique', 'Saint-Joseph, Martinique',
        'Ducos, Martinique', 'Rivière-Salée, Martinique', 'Le Marin, Martinique',
        'Sainte-Anne, Martinique', 'Les Trois-Îlets, Martinique', 'Le Diamant, Martinique',
        'Pointe-à-Pitre, Guadeloupe', 'Les Abymes, Guadeloupe', 'Baie-Mahault, Guadeloupe',
        'Le Gosier, Guadeloupe', 'Petit-Bourg, Guadeloupe', 'Sainte-Anne, Guadeloupe',
        'Le Moule, Guadeloupe', 'Basse-Terre, Guadeloupe', 'Capesterre-Belle-Eau, Guadeloupe',
        'Morne-à-l\'Eau, Guadeloupe', 'Lamentin, Guadeloupe', 'Saint-François, Guadeloupe'
    ]
    
    industries = [
        # Technology & Innovation
        "Technology", "Software Development", "Artificial Intelligence", "Machine Learning", "Data Science",
        "Cybersecurity", "Cloud Computing", "DevOps", "Web Development", "Mobile Development",
        "Game Development", "E-commerce", "Fintech", "Edtech", "Healthtech", "Proptech",
        "Blockchain", "Cryptocurrency", "Web3", "IoT", "AR/VR", "Quantum Computing",
        "Robotics", "Automation", "SaaS", "API Development", "Digital Marketing",
        
        # Finance & Banking
        "Finance", "Banking", "Investment Banking", "Investment Management", "Asset Management",
        "Private Equity", "Venture Capital", "Hedge Funds", "Insurance", "Wealth Management",
        "Commercial Banking", "Retail Banking", "Credit Cards", "Payment Processing",
        "Mortgage", "Real Estate Finance", "Trading", "Risk Management", "Compliance",
        "Accounting", "Tax Services", "Financial Planning", "Microfinance", "Capital Markets",
        
        # Healthcare & Life Sciences
        "Healthcare", "Pharmaceuticals", "Biotechnology", "Medical Devices", "Diagnostics",
        "Telemedicine", "Digital Health", "Mental Health", "Dental Care", "Veterinary",
        "Clinical Research", "Drug Discovery", "Genomics", "Personalized Medicine",
        "Medical Technology", "Health Insurance", "Elder Care", "Home Healthcare",
        "Rehabilitation", "Nutrition", "Wellness", "Fitness", "Sports Medicine",
        
        # Manufacturing & Industrial
        "Manufacturing", "Automotive", "Aerospace", "Defense", "Industrial Equipment",
        "Heavy Machinery", "Electronics Manufacturing", "Textiles", "Chemical",
        "Plastics", "Metals", "Mining", "Construction Materials", "Paper & Pulp",
        "Packaging", "3D Printing", "Additive Manufacturing", "Quality Control",
        "Supply Chain", "Logistics", "Warehousing", "Distribution", "Procurement",
        
        # Energy & Utilities
        "Energy", "Oil & Gas", "Renewable Energy", "Solar", "Wind Energy",
        "Nuclear Energy", "Hydroelectric", "Geothermal", "Utilities", "Electric Utilities",
        "Water Utilities", "Waste Management", "Environmental Services", "Sustainability",
        "Carbon Trading", "Energy Storage", "Smart Grid", "Power Generation",
        "Energy Efficiency", "Clean Technology", "Green Building", "HVAC",
        
        # Transportation & Logistics
        "Transportation", "Logistics", "Supply Chain", "Shipping", "Freight",
        "Aviation", "Airlines", "Automotive", "Rail", "Maritime", "Trucking",
        "Delivery", "Last Mile", "Warehousing", "Distribution", "Fleet Management",
        "Ride Sharing", "Car Sharing", "Public Transit", "Urban Mobility",
        "Electric Vehicles", "Autonomous Vehicles", "Drones", "Package Delivery",
        
        # Retail & Consumer
        "Retail", "E-commerce", "Fashion", "Apparel", "Luxury Goods", "Jewelry",
        "Cosmetics", "Personal Care", "Home Goods", "Furniture", "Electronics Retail",
        "Consumer Electronics", "Appliances", "Sporting Goods", "Toys", "Books",
        "Music", "Movies", "Gaming", "Grocery", "Food & Beverage", "Restaurants",
        "Fast Food", "Catering", "Hospitality", "Hotels", "Tourism", "Travel",
        
        # Media & Entertainment
        "Media", "Entertainment", "Broadcasting", "Film & Television", "Streaming",
        "Music", "Publishing", "Journalism", "Advertising", "Marketing", "Public Relations",
        "Creative Services", "Graphic Design", "Digital Media", "Social Media",
        "Content Creation", "Podcasting", "Radio", "Animation", "Visual Effects",
        "Event Management", "Sports", "Esports", "Talent Management", "Artist Management",
        
        # Real Estate & Construction
        "Real Estate", "Construction", "Property Management", "Real Estate Development",
        "Commercial Real Estate", "Residential Real Estate", "Architecture", "Engineering",
        "Civil Engineering", "Structural Engineering", "Mechanical Engineering",
        "Electrical Engineering", "Interior Design", "Landscape Architecture",
        "Building Materials", "Contracting", "Project Management", "Urban Planning",
        "Property Investment", "REITs", "Facility Management", "Maintenance",
        
        # Education & Training
        "Education", "Higher Education", "K-12 Education", "Early Childhood Education",
        "Vocational Training", "Corporate Training", "Online Education", "Edtech",
        "Language Learning", "Test Preparation", "Tutoring", "Educational Services",
        "Research", "Academic Research", "Scientific Research", "Think Tanks",
        "Libraries", "Museums", "Cultural Institutions", "Training & Development",
        "Learning Management", "Curriculum Development", "Educational Technology",
        
        # Government & Public Sector
        "Government", "Public Administration", "Defense", "Military", "Law Enforcement",
        "Public Safety", "Emergency Services", "Healthcare", "Social Services",
        "Municipal Government", "State Government", "Federal Government", "Regulatory",
        "Policy Making", "Public Works", "Transportation", "Parks & Recreation",
        "Environmental Protection", "Urban Planning", "Economic Development",
        "International Relations", "Diplomacy", "Intelligence", "Homeland Security",
        
        # Legal & Professional Services
        "Legal", "Law Firms", "Corporate Law", "Litigation", "Intellectual Property",
        "Patent Law", "Tax Law", "Employment Law", "Real Estate Law", "Criminal Law",
        "Family Law", "Immigration Law", "Environmental Law", "Securities Law",
        "Compliance", "Regulatory Affairs", "Consulting", "Management Consulting",
        "Strategy Consulting", "IT Consulting", "HR Consulting", "Financial Consulting",
        "Business Process", "Change Management", "Organizational Development",
        
        # Agriculture & Food
        "Agriculture", "Farming", "Crop Production", "Livestock", "Dairy",
        "Fisheries", "Aquaculture", "Food Processing", "Food Manufacturing",
        "Beverages", "Alcoholic Beverages", "Organic Food", "Sustainable Agriculture",
        "Agricultural Technology", "Farm Equipment", "Seeds", "Fertilizers",
        "Pesticides", "Food Safety", "Food Distribution", "Grocery", "Supermarkets",
        "Food Service", "Restaurants", "Catering", "Nutrition", "Dietetics",
        
        # Non-Profit & Social Impact
        "Non-Profit", "NGO", "Social Impact", "Charity", "Humanitarian",
        "Community Development", "Social Services", "Environmental Conservation",
        "Education", "Healthcare", "Poverty Alleviation", "Human Rights",
        "Advocacy", "Policy", "Fundraising", "Volunteer Management", "Grant Writing",
        "Program Management", "International Development", "Disaster Relief",
        "Youth Development", "Senior Services", "Disability Services", "Mental Health",
        
        # Telecommunications & Communications
        "Telecommunications", "Wireless", "Internet Service Provider", "Cable",
        "Satellite", "Network Infrastructure", "5G", "Fiber Optic", "VoIP",
        "Unified Communications", "Video Conferencing", "Collaboration Software",
        "Cloud Communications", "Cybersecurity", "Network Security", "Data Centers",
        "IT Services", "Managed Services", "Cloud Services", "Software as a Service",
        "Platform as a Service", "Infrastructure as a Service", "DevOps", "IT Support",
        
        # Arts & Culture
        "Arts", "Culture", "Museums", "Galleries", "Theater", "Dance", "Music",
        "Literature", "Visual Arts", "Performing Arts", "Creative Arts", "Design",
        "Fashion Design", "Interior Design", "Graphic Design", "Web Design",
        "Photography", "Videography", "Animation", "Digital Art", "Crafts",
        "Antiques", "Collectibles", "Cultural Heritage", "Art Education", "Art Therapy",
        
        # Sports & Recreation
        "Sports", "Professional Sports", "Amateur Sports", "Fitness", "Wellness",
        "Gyms", "Personal Training", "Yoga", "Pilates", "Martial Arts", "Dance",
        "Outdoor Recreation", "Adventure Sports", "Water Sports", "Winter Sports",
        "Team Sports", "Individual Sports", "Sports Medicine", "Sports Psychology",
        "Sports Marketing", "Sports Management", "Athletic Training", "Coaching",
        "Sports Equipment", "Sporting Goods", "Recreation Centers", "Parks",
        
        # Beauty & Personal Care
        "Beauty", "Personal Care", "Cosmetics", "Skincare", "Haircare", "Fragrances",
        "Nail Care", "Spa Services", "Salon Services", "Massage Therapy", "Wellness",
        "Alternative Medicine", "Holistic Health", "Aromatherapy", "Meditation",
        "Mindfulness", "Life Coaching", "Personal Development", "Self-Care",
        "Beauty Technology", "Beauty E-commerce", "Beauty Subscription", "Organic Beauty",
        
        # Pet & Animal Care
        "Pet Care", "Veterinary", "Animal Health", "Pet Food", "Pet Products",
        "Pet Services", "Pet Grooming", "Pet Training", "Pet Sitting", "Pet Insurance",
        "Animal Welfare", "Wildlife Conservation", "Zoos", "Aquariums", "Animal Research",
        "Livestock", "Equine", "Companion Animals", "Exotic Animals", "Animal Behavior",
        "Animal Nutrition", "Veterinary Technology", "Animal Pharmaceuticals",
        
        # Security & Safety
        "Security", "Physical Security", "Cybersecurity", "Information Security",
        "Network Security", "Private Security", "Corporate Security", "Event Security",
        "Residential Security", "Commercial Security", "Industrial Security",
        "Transportation Security", "Aviation Security", "Maritime Security",
        "Border Security", "National Security", "Intelligence", "Surveillance",
        "Risk Management", "Crisis Management", "Business Continuity", "Disaster Recovery",
        
        # Other Emerging Industries
        "Space Technology", "Satellite", "Aerospace", "Aviation", "Drones",
        "Autonomous Systems", "Smart Cities", "Internet of Things", "Wearable Technology",
        "Nanotechnology", "Materials Science", "Advanced Materials", "Semiconductors",
        "Optics", "Photonics", "Sensors", "Actuators", "Embedded Systems",
        "Industrial IoT", "Edge Computing", "Quantum Computing", "Bioinformatics",
        "Computational Biology", "Digital Twins", "Augmented Reality", "Virtual Reality",
        "Mixed Reality", "Haptics", "Voice Technology", "Natural Language Processing",
        "Computer Vision", "Speech Recognition", "Predictive Analytics", "Business Intelligence"
    ]
    
    company_sizes = ["Small (1-50)", "Medium (51-200)", "Large (201-1000)", "Enterprise (1000+)"]
    
    genders = ["Male", "Female", "Other"]
    
    remote_work_options = ["Yes", "No", "Hybrid"]
    
    # Generate data
    data = []
    
    for _ in range(num_records):
        # Basic demographics
        age = np.random.randint(22, 65)
        gender = np.random.choice(genders)
        education = np.random.choice(education_levels)
        
        # Professional info
        experience = min(age - 22, np.random.randint(0, 40))
        job_title = np.random.choice(job_titles)
        location = np.random.choice(locations)
        industry = np.random.choice(industries)
        company_size = np.random.choice(company_sizes)
        remote_work = np.random.choice(remote_work_options)
        
        # Calculate base salary with realistic factors
        base_salary = 45000
        
        # Experience multiplier
        experience_multiplier = 1 + (experience * 0.08)
        
        # Education multiplier
        education_multipliers = {
            "High School": 1.0,
            "Bachelor's": 1.3,
            "Master's": 1.6,
            "PhD": 1.9
        }
        education_multiplier = education_multipliers[education]
        
        # Job title multiplier
        job_title_multipliers = {
            "Software Engineer": 1.4, "Senior Software Engineer": 1.8, "Principal Software Engineer": 2.5,
            "Data Scientist": 1.6, "Senior Data Scientist": 2.0, "Principal Data Scientist": 2.8,
            "Product Manager": 1.5, "Senior Product Manager": 2.0, "Director of Product": 2.8,
            "Marketing Manager": 1.2, "Senior Marketing Manager": 1.6, "Marketing Director": 2.3,
            "Sales Representative": 1.0, "Senior Sales Representative": 1.3, "Sales Manager": 1.8,
            "HR Manager": 1.1, "Senior HR Manager": 1.5, "HR Director": 2.1,
            "Financial Analyst": 1.2, "Senior Financial Analyst": 1.6, "Finance Manager": 2.0,
            "Business Analyst": 1.1, "Senior Business Analyst": 1.4, "Business Intelligence Manager": 1.8,
            "Project Manager": 1.3, "Senior Project Manager": 1.7, "Program Manager": 2.2,
            "DevOps Engineer": 1.5, "Senior DevOps Engineer": 1.9, "DevOps Architect": 2.4,
            "UX Designer": 1.2, "Senior UX Designer": 1.6, "Design Director": 2.2,
            "Quality Assurance Engineer": 1.0, "Senior QA Engineer": 1.3, "QA Manager": 1.7,
            "Technical Writer": 1.0, "Senior Technical Writer": 1.3, "Documentation Manager": 1.6,
            "Customer Success Manager": 1.1, "Senior Customer Success Manager": 1.5,
            "Operations Manager": 1.2, "Senior Operations Manager": 1.6, "Operations Director": 2.2
        }
        job_multiplier = job_title_multipliers.get(job_title, 1.0)
        
        # Location multiplier (cost of living)
        location_multipliers = {
            "New York, NY": 1.4, "San Francisco, CA": 1.5, "Los Angeles, CA": 1.3,
            "Chicago, IL": 1.1, "Boston, MA": 1.3, "Seattle, WA": 1.3,
            "Austin, TX": 1.1, "Denver, CO": 1.0, "Atlanta, GA": 1.0,
            "Dallas, TX": 1.0, "Miami, FL": 1.1, "Phoenix, AZ": 0.9,
            "Philadelphia, PA": 1.0, "Detroit, MI": 0.9, "Portland, OR": 1.1,
            "Nashville, TN": 0.9, "Charlotte, NC": 0.9, "San Diego, CA": 1.2,
            "Minneapolis, MN": 1.0, "Cleveland, OH": 0.9
        }
        location_multiplier = location_multipliers.get(location, 1.0)
        
        # Industry multiplier
        industry_multipliers = {
            "Technology": 1.4, "Finance": 1.3, "Healthcare": 1.1, "Manufacturing": 1.0,
            "Retail": 0.9, "Education": 0.8, "Government": 0.9, "Consulting": 1.2,
            "Media": 1.0, "Real Estate": 1.1, "Transportation": 1.0, "Energy": 1.2,
            "Telecommunications": 1.1, "Pharmaceuticals": 1.3, "Aerospace": 1.2,
            "Automotive": 1.1, "Food & Beverage": 1.0, "Fashion": 1.0, "Gaming": 1.3,
            "E-commerce": 1.2, "Fintech": 1.4, "Biotech": 1.3, "Cybersecurity": 1.5,
            "AI/ML": 1.6
        }
        industry_multiplier = industry_multipliers.get(industry, 1.0)
        
        # Company size multiplier
        company_size_multipliers = {
            "Small (1-50)": 0.9,
            "Medium (51-200)": 1.0,
            "Large (201-1000)": 1.1,
            "Enterprise (1000+)": 1.2
        }
        company_multiplier = company_size_multipliers[company_size]
        
        # Remote work multiplier
        remote_multipliers = {
            "Yes": 1.05,
            "No": 1.0,
            "Hybrid": 1.02
        }
        remote_multiplier = remote_multipliers[remote_work]
        
        # Calculate final salary
        calculated_salary = (
            base_salary * 
            experience_multiplier * 
            education_multiplier * 
            job_multiplier * 
            location_multiplier * 
            industry_multiplier * 
            company_multiplier * 
            remote_multiplier
        )
        
        # Add some randomness
        noise = np.random.normal(1.0, 0.15)
        final_salary = max(30000, calculated_salary * noise)
        
        # Round to nearest 1000
        final_salary = round(final_salary / 1000) * 1000
        
        data.append({
            'age': age,
            'gender': gender,
            'education': education,
            'experience': experience,
            'job_title': job_title,
            'location': location,
            'industry': industry,
            'company_size': company_size,
            'remote_work': remote_work,
            'salary': final_salary
        })
    
    df = pd.DataFrame(data)
    return df

def get_feature_categories():
    """
    Return the categories for each feature for consistent encoding
    """
    return {
        'education': ["High School", "Bachelor's", "Master's", "PhD"],
        'gender': ["Male", "Female", "Other"],
        'remote_work': ["Yes", "No", "Hybrid"],
        'company_size': ["Small (1-50)", "Medium (51-200)", "Large (201-1000)", "Enterprise (1000+)"]
    }

if __name__ == "__main__":
    # Generate sample data
    sample_data = generate_synthetic_data(1000)
    print("Sample data generated:")
    print(sample_data.head())
    print(f"\nDataset shape: {sample_data.shape}")
    print(f"Average salary: ${sample_data['salary'].mean():,.0f}")
