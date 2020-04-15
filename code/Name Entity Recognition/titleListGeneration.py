
import random

# text needed for title generation
# 0. demographic indicator
subgroupPeople = ['All Race', 'White', 'Under age 18', 'above age 65', 'black or African American', 'American Indian and Alaska Native',
                  'Asian', 'Native Hawaiian and Other Pacific Islander', 'Hispanic or Latino Origin', 'who believe climate change',
                  'Christian', 'Catholic', 'Jewish', 'Muslim', ' people living in poverty areas', 'people below poverty level', 'frauds',
                  'death among children under 5 due to pediatric cancer', 'per Walmart store', 'dentist', 'retailers of personal computer',
                  'people living in slums', 'women that were screened for breast and cervical cancer by jurisdiction', 'people who are confirm to be infected by 2019-Nov Coronavirus',
                  'people with elementary occupation', 'people who are alumni of OSU', 'people working more than 49 hours per week',
                  'people whose permanent teeth have been removed because of tooth decay or gum disease', 'people with a bachelor\'s degree or higher',
                  'people who changed the job in the past one year', 'People who is infected by HIV', 'People whose native language is Russian',
                  'males 15 years and over', ' never married', 'now married, except separated', 'separated',
                  'widowed', 'divorced', 'people enrolled in Nursery school, people enrolled in preschool',
                  'people enrolled in Kindergarten', 'people enrolled in Elementary school (grades 1-8)', 'people enrolled in High school (grades 9-12)',
                  'people enrolled in College or graduate school']
demographic = ['Percent of population', 'Percent change', 'number of people',
               'difference in number of people', 'Population density', 'difference in population density']

# 1. economic indicator
economic = ["unemployment rate", "gross domestic income (nominal or ppp)", "gross domestic income (nominal or ppp) per capita",
            "GDP (nominal or ppp)", "GDP (nominal or ppp) per capita", "Median household income", "Household income", "price of land",
            "percent of houses with annual income of $300,000 and over", "percent of houses with annual income of $50,000 and less",
            "poverty rate", "economic growth rate", "percent of households above $200k", "average price for honey per pound",
            "federal government expenditure (per capita)", "median rent price", "sale amounts of beer", "NSF funding for \"Catalogue\"",
            "Agriculture exports", "number of McDonald's", "import and export statistics", "Gross profit of companies"]

# 2. physical indicator
physical = ["annual average temperature", "annual average precipitation",
            "number of fire points", "number of earthquake"]

# 3. social indicator
social = ["number of libraries", "average age", "adult obesity", "measles incidence", "flu incidence", "Human Development Index",
          "mortality associated with arterial hypertension", "number of patent", "number of patent per capita", "life expectancy",
          "crime rate", "homicide rate", "suicide rate", "firearm death rate", "gun violence rate", "social vulnerability index",
          "freedom index", "divorce rate", "people living in poverty areas", "rate of male", "energy consumption (per capita)",
          "CO2 emission (per capita)", " Percent of planted soybeans by acreage", "NBA player origins (per capita)", "number of species",
          "happiness score", "availability of safe drinking water", "number of cell phones per 100 person", "percent of farmland",
          "number of hospitals", "well-being index", "food insecurity rate", "diabetes rate", "helicobacter pylori rate", "number of schools",
          "fertility rate", "number of multi-racial households", "number of fixed residential broadband providers", "lung cancer mortality rate",
          "renter occupied", "burglary per 1000 household", "infant mortality rate", "number of pedestrian accidents", "number of academic articles published",
          "number of Olympic game awards", "percent of forest area", "percent of farms with female principal operator", "percentage of respondents who did not provide a workplace address at Area unit level",
          "License plate vanitization rate", 'Race diversity index', 'difference in race diversity', ]
# 4. housing indicator
housing = ['Average number of bedrooms of houses', 'Average square footage of houses', 'Age of householder', 'Average year built', 'Household income', 'Average monthly housing cost',
           'Average monthly housing cost as percentage of income', 'Average poverty level for household']

# 5. retail indicator
salesForRetail = 'Estimated annual sales for '
retailBusiness = ['total', 'total (excl. motor vehicle & parts)', 'total (excl. gasoline stations)', 'total (excl. motor vehicle & parts & gasoline stations)',
                  'Motor vehicle & parts Dealers', 'Auto & other motor veh. Dealers', 'New car dealers', 'Auto parts', 'acc. & tire store', 'Furniture & home furn. Stores', 'Furniture stores', 'Home furnishings stores',
                  'Electronics & appliance stores', 'Building material & garden eq. & supplies dealers', 'Building mat. & sup. dealers', 'Food & beverage stores', 'Grocery stores', 'Beer, wine & liquor stores',
                  'Health & personal care stores', 'Pharmacies & drug stores', 'Gasoline stations', 'Clothing & clothing accessories stores', 'Mens clothing stores', 'Womens clothing stores', 'Family clothing stores',
                  'Shoe stores', 'Sporting goods hobby, musical instrument, & book stores', 'General merchandise stores', 'Department stores', 'Other general merch. Stores', 'Warehouse clubs & supercenters',
                  'All oth. gen. merch. Stores', 'Miscellaneous store retailer', 'Nonstore retailers', 'Elect. shopping & m/o houses', 'Food services & drinking places']

# 6. manufacturing indicator
capitalExpenses = ['Number of employees', 'Annual payroll',	'Production workers average for year', 'Production workers annual hours', 'Production workers annual wages', 'Total cost of materials',
                   'Total value of shipments and receipts for services', 'Total capital expenditures']
connectWordManu = ' of '
manufacturingType = ['Manufacturing', 'Food manufacturing', 'Animal food manufacturing', 'Grain and oilseed milling', 'Sugar and confectionery product manufacturing',
                     'Fruit and vegetable preserving and specialty food manufacturing', 'Dairy product manufacturing', 'Animal slaughtering and processing', 'Seafood product preparation and packaging',
                     'Bakeries and tortilla manufacturing', 'Other food manufacturing', 'Beverage and tobacco product manufacturing', 'Beverage manufacturing', 'Tobacco manufacturing', 'Textile mills',
                     'Fiber, yarn, and thread mills', 'Fabric mills', 'Textile and fabric finishing and fabric coating mills', 'Textile product mills', 'Textile furnishings mills', 'Other textile product mills',
                     'Apparel manufacturing', 'Apparel knitting mills', 'Cut and sew apparel manufacturing', 'Apparel accessories and other apparel manufacturing', 'Leather and allied product manufacturing',
                     'Leather and hide tanning and finishing', 'Footwear manufacturing', 'Other leather and allied product manufacturing', 'Wood product manufacturing', 'Sawmills and wood preservation',
                     'Veneer, plywood, and engineered wood product manufacturing', 'Other wood product manufacturing', 'Paper manufacturing', 'Pulp, paper, and paperboard mills', 'Converted paper product manufacturing',
                     'Printing and related support activities', 'Printing and related support activities', 'Petroleum and coal products manufacturing', 'Petroleum and coal products manufacturing',
                     'Chemical manufacturing', 'Basic chemical manufacturing', 'Resin, synthetic rubber, and artificial synthetic fibers and filaments manufacturing',
                     'Pesticide, fertilizer, and other agricultural chemical manufacturing', 'Pharmaceutical and medicine manufacturing',
                     'Paint, coating, and adhesive manufacturing', 'Soap, cleaning compound, and toilet preparation manufacturing', 'Other chemical product and preparation manufacturing',
                     'Plastics and rubber products manufacturing', 'Plastics product manufacturing', 'Rubber product manufacturing', 'Nonmetallic mineral product manufacturing', 'Clay product and refractory manufacturing',
                     'Glass and glass product manufacturing', 'Cement and concrete product manufacturing', 'Lime and gypsum product manufacturing', 'Other nonmetallic mineral product manufacturing',
                     'Primary metal manufacturing', 'Iron and steel mills and ferroalloy manufacturing', 'Steel product manufacturing from purchased steel', 'Alumina and aluminum production and processing',
                     'Nonferrous metal (except aluminum) production and processing', 'Foundries', 'Fabricated metal product manufacturing', 'Forging and stamping,Cutlery and handtool manufacturing',
                     'Architectural and structural metals manufacturing', 'Boiler, tank, and shipping container manufacturing', 'Hardware manufacturing,Spring and wire product manufacturing',
                     'Machine shops; turned product; and screw, nut, and bolt manscrewufacturing', 'Coating, engraving, heat treating, and allied activities', 'Other fabricated metal product manufacturing',
                     'Machinery manufacturing', 'Agriculture, construction, and mining machinery manufacturing', 'Industrial machinery manufacturing', 'Commercial and service industry machinery manufacturing',
                     'Ventilation, heating, air-conditioning, and commercial refrigeration equipment manufacturing', 'Metalworking machinery manufacturing', 'Engine, turbine, and power transmission equipment manufacturing',
                     'Other general purpose machinery manufacturing', 'Computer and electronic product manufacturing', 'Computer and peripheral equipment manufacturing', 'Communications equipment manufacturing',
                     'Audio and video equipment manufacturing', 'Semiconductor and other electronic component manufacturing', 'Navigational, measuring, electromedical, and control instruments manufacturing',
                     'Manufacturing and reproducing magnetic and optical media', 'Electrical equipment, appliance, and component manufacturing', 'Electric lighting equipment manufacturing',
                     'Household appliance manufacturing', 'Electrical equipment manufacturing', 'Other electrical equipment and component manufacturing', 'Transportation equipment manufacturing', 'Motor vehicle manufacturing',
                     'Motor vehicle body and trailer manufacturing', 'Motor vehicle parts manufacturing', 'Aerospace product and parts manufacturing', 'Railroad rolling stock manufacturing', 'Ship and boat building',
                     'Other transportation equipment manufacturing,Furniture and related product manufacturing', 'Household and institutional furniture and kitchen cabinet manufacturing',
                     'Office furniture (including fixtures) manufacturing', 'Other furniture related product manufacturing', 'Miscellaneous manufacturing', 'Medical equipment and supplies manufacturing',
                     'Other miscellaneous manufacturing']

# 7. firm exporting indicator
exportingFirms = ['Number of firms', 'Sales, receipts, or value of shipments of firms',
                  'Exports value of firms', 'Number of paid employees', 'Annual payroll']

# 8. school finance indicator
schoolFinance = ['Elementary-secondary revenue', 'Elementary-secondary revenue from federal sources', 'Elementary-secondary revenue from state sources', 'Elementary-secondary revenue from local sources',
                 'Elementary-secondary expenditure',  'Current spending of elementary-secondary expenditure', 'Capital outlay of elementary-secondary expenditure',
                 'Elementary-secondary revenue from general formula assistance', 'Elementary-secondary revenue from compensatory programs', 'Elementary-secondary revenue from special education',
                 'Elementary-secondary revenue from vocational programs', 'Elementary-secondary revenue from transportation programs', 'Elementary-secondary revenue from other state aid',
                 'Elementary-secondary revenue from property taxes', 'Elementary-secondary revenue from parent government contributions', 'Elementary-secondary revenue from school lunch charges',
                 'Elementary-secondary revenue from local government']

# 9. government finance indicator
governFinance = ['Total revenue', 'General revenue', 'Intergovernmental revenue', 'Taxes', 'General sales', 'Selective sales', 'License taxes', 'Individual income tax',
                 'Corporate income tax', 'Other taxes', 'Current charge', 'Miscellaneous general revenue', 'Utility revenue', 'Liquor stores revenue',
                 'Insurance trust revenue', 'Total expenditure', 'Intergovernmental expenditure', 'Direct expenditure', 'Current operation', 'Capital outlay',
                 'Insurance benefits and repayments', 'Assistance and subsidies', 'Interest on debt', 'Salaries and wages', 'Total expenditure',
                 'General expenditure', 'Intergovernmental expenditure', 'Direct expenditure', 'General expenditure', 'Interest on general debt', 'Utility expenditure',
                 'Liquor stores expenditure', 'Insurance trust expenditure', 'Debt at end of fiscal year', 'Cash and security holdings', 'Total Taxes', 'Property Taxes',
                 'Sales and Gross Receipts Taxes', 'License Taxes', 'Income Taxes']
governFinancePost = ' of governments'

# 10. household indicator
household = ['Total households', 'Family households (families)', 'Family households with own children of the householder under 18 years',
             'Married-couple family', 'Households with male householder, no wife present, family', 'Households with female householder, no husband present, family',
             'Nonfamily households', 'Households with householder living alone', 'Households with one or more people under 18 years',
             'Households with one or more people 65 years and over', 'Average household size', 'Average family size']


# 11. time use indicator
timeUse = ['Average hours per day spent on ', 'Average percent of time engaged in ', 'Average hours per day by men spent on ', 'Average percent of time engaged in by men',
           'Average hours per day by women spent on ', 'Average percent of time engaged in by women']
timeUseType = ['Sleeping', 'Grooming', 'Health-related self care', 'Personal activities', 'Travel related to personal care]', 'Eating and drinking', 'Interior cleaning', 'Laundry',
               'Storing interior household items, including food', 'Food and drink preparation', 'Kitchen and food cleanup', 'Lawn and garden care', 'Household management',
               'Financial management', 'Interior maintenance, repair, and decoration', 'Exterior maintenance, repair, and decoration', 'Animals and pets', 'Care for animals and pets, not veterinary care',
               'Walking, exercising, and playing with animals', 'Vehicles', 'Appliances, tools, and toys', 'Travel related to household activities', 'Purchasing goods and services', 'Consumer goods purchases',
               'Grocery shopping', 'Financial services and banking', 'Medical and care services', 'Household services', 'Home maintenance, repair, decoration, and construction (not done by self)',
               'Vehicle maintenance and repair services', 'Government services', 'Travel related to purchasing goods and services', 'Caring for and helping household members',
               'Caring for and helping household children', 'Physical care for household children', 'Reading to and with household children', 'Talking with and listening to household children',
               'Playing with household children, not sports', 'Attending household children events', 'Activities related to household children education', 'Helping household children with Homework',
               'Activities related to household children health', 'Caring for and helping household adults', 'Caring for household adults', 'Physical care for household adults', 'Helping household adults',
               'Travel related to caring for and helping household members', 'Caring for and helping nonhousehold members', 'Caring for and helping nonhousehold children',
               'Caring for and helping nonhousehold adults', 'Caring for nonhousehold adults', 'Helping nonhousehold adults', 'Travel related to caring for and helping nonhousehold membership',
               'Working and work-related activities', 'Working', 'Work-related activities', 'Other income-generating activities', 'Job search and interviewing', 'Travel related to work',
               'Educational activities', 'Attending class', 'Taking class for degree, certificate, or licensure', 'Homework and research', 'Travel related to education',
               'Organizational, civic, and religious activities', 'Religious and spiritual activities', 'Attending religious services', 'Participating in religious practices',
               'Volunteering (organizational and civic activities)', 'Volunteer activities', 'Administrative and support activities', 'Social service and care activities ',
               'Indoor and outdoor maintenance, building, and cleanup activities', 'Participating in performance and cultural', 'activities', 'Attending meetings, conferences, and training',
               'Civic obligations and participation', 'Travel related to organizational, civic, and religious activities', 'Leisure and sports', 'Socializing, relaxing, and leisure',
               'Socializing and communicating', 'Attending or hosting social events', 'Relaxing and leisure', 'Watching TV', 'Relaxing and thinking', 'Playing games',
               'Computer use for leisure, excluding games', 'Reading for personal interest', 'Arts and entertainment (other than sports)', 'Sports, exercise, and recreation',
               'Participating in sports, exercise, and recreation', 'Walking', 'Attending sporting or recreational events', 'Travel related to leisure and sports',
               'Telephone calls, mail, and e-mail', 'Telephone calls (to or from)', 'Household and personal messages', 'Household and personal mail and messages',
               'Household and personal e-mail and messages', 'Travel related to telephone calls']


# generate title
def getTitle():
    titleTypeID = random.randint(0, 11)
    year = random.randint(1950, 2020)
    geoRegion = ['the United States', 'the UK',
                 'China', 'Canada', 'South Korea', 'France']
    lenGeoRegion = len(geoRegion)
    adminLevel = ['state', 'county', 'township', 'census tract']
    lenAdminLevel = len(adminLevel)

    if (titleTypeID == 0):
        lenDemo = len(demographic)
        lenSub = len(subgroupPeople)
        theme = demographic[random.randint(
            0, lenDemo-1)] + " of " + subgroupPeople[random.randint(0, lenSub-1)]
    elif (titleTypeID == 1):
        lenEco = len(economic)
        theme = economic[random.randint(0, lenEco-1)]
    elif (titleTypeID == 2):
        lenPhy = len(physical)
        theme = physical[random.randint(0, lenPhy-1)]
    elif (titleTypeID == 3):
        lenSoc = len(social)
        theme = social[random.randint(0, lenSoc-1)]
    elif (titleTypeID == 4):
        lenHou = len(housing)
        theme = housing[random.randint(0, lenHou-1)]
    elif (titleTypeID == 5):
        lenRet = len(retailBusiness)
        theme = salesForRetail + retailBusiness[random.randint(0, lenRet-1)]
    elif (titleTypeID == 6):
        lenCapExp = len(capitalExpenses)
        lenManType = len(manufacturingType)
        theme = capitalExpenses[random.randint(
            0, lenCapExp-1)] + connectWordManu + manufacturingType[random.randint(0, lenManType-1)]
    elif (titleTypeID == 7):
        lenExpFirm = len(exportingFirms)
        theme = exportingFirms[random.randint(0, lenExpFirm-1)]
    elif (titleTypeID == 8):
        lenSchFin = len(schoolFinance)
        theme = schoolFinance[random.randint(0, lenSchFin-1)]
    elif (titleTypeID == 9):
        lenGovFin = len(governFinance)
        theme = governFinance[random.randint(
            0, lenGovFin-1)] + governFinancePost
    elif (titleTypeID == 10):
        lenHouHold = len(household)
        theme = household[random.randint(0, lenHouHold-1)]
    else:
        lenTimeUse = len(timeUse)
        lenTimeUseType = len(timeUseType)
        theme = timeUse[random.randint(
            0, lenTimeUse-1)] + timeUseType[random.randint(0, lenTimeUseType-1)]
    theme = theme + " "
    title = ""
    region = "in " + geoRegion[random.randint(0, lenGeoRegion-1)] + " "
    admin = "by "+adminLevel[random.randint(0, lenAdminLevel-1)] + " "
    time = "in " + str(year) + " "

    # shuffle the three optional elements
    shuffleList = []
    shuffleList.append(region)
    shuffleList.append(admin)
    shuffleList.append(time)
    random.shuffle(shuffleList)

    elementNum = random.randint(1, 3)

    optElements = shuffleList[0:elementNum]
    optElements.append(theme)

    random.shuffle(optElements)

    for i in range(0, len(optElements)):
        if optElements[i] == theme:
            bIndex = len(title)
            eIndex = bIndex + len(theme) - 1
        title = title + optElements[i]

    return title, bIndex, eIndex


def getTitleList():
    titleList = []
    count = 0
    while count < 1000:
        title, bIndex, eIndex = getTitle()
        if title not in titleList:
            titleToList = list(title)
            if titleToList[0].islower():
                titleToList[0] = titleToList[0].upper()
                title = "".join(titleToList)
            titleList.append([title, bIndex, eIndex])
            count = count + 1
    # print(titleList)
    return titleList


def main():
    titleList = getTitleList()
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\code\\Name Entity Recognition\\testTitleList.txt', 'w') as f:
        for item, bIndex, eIndex in titleList:
            f.write("%s\n" % item)
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\code\\Name Entity Recognition\\labelTemplate.txt', 'w') as f:
        for item, bIndex, eIndex in titleList:
            f.write("(\n")
            f.write('"%s",\n' % item)
            indexLabel = '{"entities": [(' + str(bIndex) + ', ' + str(eIndex) + ', LABEL)]},\n'
            f.write(indexLabel)
            f.write("),\n")


if __name__ == "__main__":
    main()
