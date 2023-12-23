import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#PROBLEM 1: DATA EXPLORATION AND CLEANING

# Load the dataset
df = pd.read_excel('amazon_laptop_2023.xlsx')
df.head()

df.info()

# Summary of data types and missing values
data_info = pd.DataFrame(df.dtypes, columns=['Data Type'])
data_info['Missing Values'] = df.isnull().sum()
data_info['Unique Values'] = df.nunique()
data_info['First Value'] = df.iloc[0]
data_info


# All columns are of object type except rating, which is float64. This suggests that numerical columns like price might be stored as strings.

duplicates = df[df.duplicated()]
#  number of duplicate rows
print("Total number of duplicate rows: ", duplicates.shape[0])
df = df.drop_duplicates()


# **Solving Issues**
# Step 1: Removing currency symbols and converting 'price' to a numerical column
# Using regex to remove $ and commas, then converting to float, coercing errors to NaN
if 'price' in df.columns and df['price'].dtype != object:
    df['price'] = df['price'].astype(str)

# Now applying the string replacement and converting to float
df['price'] = df['price'].str.replace('[$,]', '', regex=True).astype(float, errors='ignore')
# Renaming the column to price($)
df = df.rename(columns={'price': 'price($)'})



# Step 2: Extracting numerical part of 'screen_size' and converting to a separate column
# converting text to float, e.g., "14 Inches" to 14.0
df['screen_size_inches'] = df['screen_size'].str.extract('(\d*\.?\d+)').astype(float, errors='ignore')

# Step 3: Handling 'harddisk' sizes with GB and TB units
# Extracting the numerical part and converting to float
df['harddisk_GB'] = df['harddisk'].str.extract('(\d+)').astype(float, errors='ignore')
# Multiplying by 1024 if the unit is TB to convert to GB
df['harddisk_GB'] = df.apply(lambda x: x['harddisk_GB'] * 1024 if 'TB' in str(x['harddisk']) else x['harddisk_GB'], axis=1)



# Step 4: Extracting numerical part of 'ram' 'cpu_speed' and converting to a separate numerical column
df['ram_GB'] = df['ram'].str.extract('(\d+)').astype(float, errors='ignore')
df['cpu_speed_GHz'] = df['cpu_speed'].str.extract('(\d+)').astype(float, errors='ignore')
df.loc[df['cpu_speed_GHz'] > 8.5, 'cpu_speed_GHz'] = None

# droping columns as created  new columns
df.drop(columns=['ram','screen_size', 'harddisk', 'cpu_speed'], inplace=True)
df.dropna(subset=['model'], inplace=True)

# step 5
# fill in missing data by using the similar models, cpu, special features
df['special_features'] = df.groupby('model')['special_features'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['cpu_speed_GHz'] = df.groupby('cpu')['cpu_speed_GHz'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['graphics_coprocessor'] = df.groupby('cpu')['graphics_coprocessor'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['color'] = df['color'].fillna('Unknown')



def standardize_text(text):
    if pd.isna(text):
        return text
    return ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), str(text).lower()))

# Apply this function only to categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(standardize_text)

#step 6 
# Coreecting the column values using regex

#step 6
#barnd correction 
#mapped some model names to their brand and removed unecessary values 
brand_corrections = {
    r'.*(latitude|dell)':'dell',
    r'.*(mac|apple)': 'apple',
    r'.*(lenovo)': 'lenovo',
    r'.*(mytrix|acer)':'acer',
    r'.*(toughbook|panasonic)':'panasonic',
    r'.*(luqeeg|unknowns|shoxlab|rokc|microtella|awow|best notebooks|carlisle foodservice product|elo|gizpro|jtd|lpt|quality refurbished computers)': 'unknown'
}

for regex, brands in brand_corrections.items():
    df['brand'] = df['brand'].str.replace(regex, brands, regex=True)

df['brand'] = df['brand'].str.capitalize()

#step6
#color correction

# Color mapping function using regex
color_corrections = {
    'Black': r'.*(black|balck|midnight|ash|dark|moon|paint|with illuminated razer logo|with high endurance clear coat and silky smooth finish)',
    'Silver': r'.*(silver|sliver|aluminum|platinum|matte|titan)',
    'Blue': r'.*(blue|sky|dark teal|lunar light|cobalt)',
    'Green': r'.*(green|sage|dark moss|teal|moss|soft mint)',
    'Gray': r'.*(grey|gray|gary|mercury|graphite|carbon fiber|speckles|dull)',
    'Gold': r'.*(gold)',
    'Red': r'.*(red)',
    'White': r'.*(white)',
    'Brown': r'.*(almond|beige mousse|dune)',
    'Pink': r'.*(pink)',
    'Unknown': r'.*(touchscreen|rgb backlit|acronym|electro punk|evo i71260p|apollo|information not available|light titan)'
}

#Funtion for handling rows where there are more than one color, adds a / between differnt colors by storing them in a set
def seprate_colors(color_name):
    colors = set()
    for coloring, pattern in color_corrections.items():
        if re.search(pattern, color_name, re.IGNORECASE):
            colors.add(coloring)
    return ' / '.join(sorted(colors)) if colors else 'Unknown'

df['color'] = df['color'].astype(str).apply(seprate_colors)


#step 6 
#cpu correction 
#Function to clean cpu 
def clean_cpu(cpu):

    cpu =str(cpu)
    #Handling uncecessary values
    if pd.isna(cpu) or cpu in ['unknown', 'others', 'nan']:
        return 'Unknown'
    cpu = re.sub(r'[^a-zA-Z0-9\s-]', '', cpu)

    # Standardising cpu names using regx 
    cpu = re.sub(r'(?i)core\s*i(\d+)\s*(\d+u?)', r'Intel core i\1-\2', cpu)  
    cpu = re.sub(r'(?i)ryzen\s*(\d+)\s*(\d+u|h)', r'AMD ryzen \1 \2', cpu)  
    cpu = re.sub(r'(?i)apple\s*m1', 'Apple m1', cpu)
    cpu = re.sub(r'(?i)core\s*m', 'Intel core m', cpu)
    cpu = re.sub(r'(?i)\bcore', 'Intel core', cpu)
    cpu = re.sub(r'(?i)celeron', 'Intel celeron', cpu)
    cpu = re.sub(r'(?i)pentium', 'Intel pentium', cpu)
    cpu = re.sub(r'(?i)athlon', 'AMD athlon', cpu)
    cpu = re.sub(r'(?i)core\s*i\d+\s*family', 'Intel core family', cpu)  
    cpu = re.sub(r'(?i)amd\s*r\s*series', 'AMD r series', cpu)
    cpu = re.sub(r'(?i)\bintel\b', 'Intel', cpu)
    cpu = re.sub(r'(?i)\bamd\b', 'AMD', cpu)
    cpu = re.sub(r'(?i)aseries\s+dualcore\s+a(\d+)', r'AMD A-series A\1', cpu)
    cpu = ' '.join(dict.fromkeys(cpu.split()))  

    #Replacing all the non cpu values with 'Unknown'
    if re.search(r'(?i)(intel|amd|apple|core|ryzen|athlon|celeron|pentium)', cpu) is None:
        return 'Unknown'

    return cpu.strip()

df['cpu'] = df['cpu'].apply((clean_cpu))

#step6
#OS correction
#Funtion to clean 'OS'
def clean_os(OS):
    #Handling uncecessary values
    if pd.isna(OS) or OS in ['unknown', 'others', 'nan']:
        return 'Unknown'

    # Standardizing OS names usning regex
    OS = re.sub(r'windows\s+10\s+(64|dg).*', 'Windows 10', OS)
    OS = re.sub(r'win\s+10\s+(multilanguage).*', 'Windows 10', OS)
    OS = re.sub(r'windows\s+10\s+(pro|professional).*', 'Windows 10 pro', OS)
    OS = re.sub(r'win\s+10\s+(pro).*', 'Windows 10 pro', OS)
    OS = re.sub(r'windows\s+11\s+(pro|professional).*', 'Windows 11 pro', OS)
    OS = re.sub(r'windows\s+10\s+(home|s).*', 'Windows 10 home', OS)
    OS = re.sub(r'windows\s+11\s+(home|s).*', 'Windows 11 home', OS)
    OS = re.sub(r'win\s+11\s+(multihome).*', 'Windows 11 home', OS)
    OS = re.sub(r'windows\s+7.*', 'Windows 7', OS)
    OS = re.sub(r'windows\s+8.*', 'Windows 8', OS)
    OS = re.sub(r'windows\s+81', 'Windows 8.1', OS)
    OS = re.sub(r'mac\s*os\s*x\s*10\.?(\d+).*', 'macOS 10.\1', OS)
    OS = re.sub(r'mac\s*os\s*x', 'macOS', OS)
    OS = re.sub(r'mac\s*os', 'macOS', OS)
    OS = re.sub(r'macos', 'macOS', OS)
    OS = re.sub(r'linux', 'Linux', OS)
    OS = re.sub(r'chrome\s*os', 'Chrome OS', OS)
    OS = re.sub(r'(no|unknown|nan|thinpro)', 'Unknown', OS)
    OS = re.sub(r'(microsoft|hp)', '', OS)

    return OS.strip()

df['OS'] = df['OS'].apply(clean_os)


#step6
#special features correction
#features mapping using regex
features_corrections = {
    'Anti-glare': r'anti[-\s]*glare|glarethin|glare\s*screen|glare\s*coating',
    'Backlit keyboard': r'backlit\s*keyboard|keyboard\s*backlit',
    'Fingerprint reader': r'fingerprint\s*reader',
    'HD audio': r'hd\s*audio|high\s*definition\s*audio',
    'Memory card slot': r'memory\s*card\s*slot',
    'Numeric keypad': r'numeric\s*keypad',
    'Stylus ': r'stylus|support\s*stylus',
    'WiFi': r'wi[-\s]*fi|wifi',
    'Bluetooth': r'bluetooth',
}
#Funtion for handling rows where there are more than one feature, adds a / between differnt features by storing them in a set
def extract_features(feature):
    features = set()
    for feature_name, pattern in features_corrections.items():
        if re.search(pattern, feature, re.IGNORECASE):
            features.add(feature_name)
    return ' / '.join(sorted(features)) if features else 'Unknown'

df['special_features'] = df['special_features'].astype(str).apply(extract_features)

#step6
#graphic correction
#Function to clean graphic
def clean_graphics(row):
    #checking the graphics_coprocessor rows if there is 'integrated' or 'dedicated' then copy and paste it in the respected graphic row
    gc = row['graphics_coprocessor']
    if pd.notna(gc) and isinstance(gc, str):
        if 'Integrated' in gc.lower():
            return 'integrated'
        elif 'dedicated' in gc.lower():
            return 'Dedicated'

    #graphic mappong function using regex
    graphics_correction = {
        r'.*(dedicated|rtx|560|mvidia)': 'Dedicated',
        r'.*(integrated|intel|adreno|r4|r5|mediatek|amd|mobility|vega|radeon|uhd)': 'Integrated'
    }

    for regex, graphic in graphics_correction.items():
        if re.search(regex, str(row).lower()):
            return graphic

    # Return original value if no match is found
    return row['graphics']

df['graphics'] = df.apply(clean_graphics, axis=1)


#step6
#graphic coprocessor correction
# cleaning graphics_coprocessor
#gpu mapping using regex
gpu_correcrion = {
    r't550':'nvidia quadro t500',
    r't600' : 'nvidia quadro t600',
    r'620u': 'intel uhd 620u',
    r'rtxâ' : 'rtx',
    r'ati': 'amd',
    r'arm malig52 2ee mc2 gpu':'arm mali-g52 mp2',
    r'geforce rtx 3070 ti iris xe graphics':'geforce rtx 3070 ti / iris xe graphics',
    r'rtx a2000 uhd graphics':'rtx a2000 / uhd graphics',
    r't500 iris xe graphics': 't500 / iris xe graphics',
    r'hd graphics optimus graphics':'hd graphics / optimus graphics',
    r'geforcer':'geforce',
}

for regex, gpu in gpu_correcrion.items():
    df['graphics_coprocessor'] = df['graphics_coprocessor'].astype(str).str.replace(regex, gpu, regex=True)

#Function to remove the brand names from the column to create a new column processor_brand
def extract_brand(graphics_coprocessor):
    if pd.isna(graphics_coprocessor):
        return None
    brand = re.search(r'intel|amd|nvidia|apple|mediatek|arm|ati', graphics_coprocessor, re.IGNORECASE)
    return brand.group(0).title() if brand else None

# Function to remove brand names and unecessary words
def remove(graphics_coprocessor):
    if pd.isna(graphics_coprocessor):
        return None
    return re.sub(r'intel|amd|nvidia|apple|mediatek|integrated|intergrated|integreted|dedicated|â|2gb|silver|embedded|2ee mc2 gpu|8gb gddr6|oc with 8gb gddr5|laptop|gpu|4gb gddr6|eligible|16|gb|gddr6|6gb|1 gddr6|with|design|w4gb|12|1|ddr6|premium|xps93007909slvpus|ada|4dp|processor|graphics|grpahics', '', graphics_coprocessor, flags=re.IGNORECASE).strip()

#created new column processor brand
df['processor_brand'] = df['graphics_coprocessor'].apply(extract_brand)
df['graphics_coprocessor'] = df['graphics_coprocessor'].apply(remove)
df['graphics_coprocessor'] = df['graphics_coprocessor'].replace('', np.nan)
df['graphics_coprocessor'].fillna('Unknown', inplace=True)
df['processor_brand'] = df['processor_brand'].fillna('Unknown')




#step6 
#model correction
#Function to clean model, mapping some common model using regex
def clean_model(model):
    model = str(model)
    if pd.isna(model):
        return model
    model = re.sub(r'(?i)elite\s*book\s*(\d+\s*\w*)', r'EliteBook \1', model)
    model = re.sub(r'(?i)omen\s*(\d+\s*\w*)', r'Omen \1', model)
    model = re.sub(r'(?i)pavilion\s*(\d+\s*\w*)', r'Pavilion \1', model)
    model = re.sub(r'(?i)pro\s*book\s*(\d+\s*\w*)', r'ProBook \1', model)
    model = re.sub(r'(?i)z\s*book\s*(\d+\s*\w*)', r'ZBook \1', model)
    model = re.sub(r'(?i)spectre\s*(\d+\s*\w*)', r'Spectre \1', model)
    model = re.sub(r'(?i)envy\s*(\d+\s*\w*)', r'Envy \1', model)
    model = re.sub(r'(?i)pro\s*desk\s*(\d+\s*\w*)', r'ProDesk \1', model)
    model = re.sub(r'(?i)victus\s*(\d+\s*\w*)', r'Victus \1', model)
    model = re.sub(r'(?i)elite\s*dragonfly\s*(\d+\s*\w*)', r'Elite Dragonfly \1', model)
    model = re.sub(r'(?i)chromebook\s*(\w+\s*\d+\s*\w*)', r'Chromebook \1', model)
    model = re.sub(r'(?i)dell?\s*latitude\s*(\d+\s*\w*)', r' Latitude \1', model)
    model = re.sub(r'(?i)lenovo?\s*thinkpad\s*(\w+\s*\d+\s*\w*)', r'ThinkPad \1', model)
    model = re.sub(r'(?i).*macbook\s*air.*', r'MacBook Air', model)
    model = re.sub(r'(?i).*macbook\s*pro.*', r'MacBook Pro', model)
    model = re.sub(r'(?i).*\bmacbook\b(?!.*air|.*pro).*', r'Apple MacBook', model)
    
    return model

#Function to remove the brand name from the model
def remove_brand1(df, brand, model):
    unique_brands = set(df[brand].dropna().unique())
    dictionary = {brand: re.compile(re.escape(brand), re.IGNORECASE) for brand in unique_brands}

    def clean_model1(model_value):
        if pd.isna(model_value):
            return model_value
        for brand, model_regex in dictionary.items():
            model_value = model_regex.sub('', model_value).strip()
        return model_value

    df[model] = df[model].apply(clean_model1)
    return unique_brands, df

#Function to remove unecessary words from rows
def remove1(model):
    if pd.isna(model):
        return None
    return re.sub(r'Laptop|laptop|2in1|detechable|detachable|multitouch|2023|newest|13|inch|16|gb|ram|storage |space|gray|z15s000ct', '', model, flags=re.IGNORECASE).strip()

df['model'] = df['model'].apply(clean_model)
df['model'] = df['model'].apply(remove1)

unique_brands, df = remove_brand1(df, 'brand', 'model')


#step7
#reording of colums
df = df[['brand','model','color','special_features','cpu','cpu_speed_GHz','graphics','processor_brand','graphics_coprocessor','OS','ram_GB','harddisk_GB','screen_size_inches','price($)','rating']]

df.info()
# Summary of data types and missing values
data_info = pd.DataFrame(df.dtypes, columns=['Data Type'])
data_info['Missing Values'] = df.isnull().sum()
data_info['Unique Values'] = df.nunique()
data_info['First Value'] = df.iloc[0]
data_info

print(df)

#step8
#writing it to new file
df.to_excel('amazon_laptop_2023_cleaned.xlsx', index=False)
df



#PROBLEM 2: DATA VISUALIZATION

df = pd.read_excel('amazon_laptop_2023_cleaned.xlsx')

# Filtering laptops for Customer A (Travel): 
# For this scenario, we will assume that more harddisk space, smaller screensize and higher price (to some extent) could indicate better build quality and durability

# Filter laptops within the $1500 budget
laptop_price = df[df['price($)'] <= 1500]


customer_a_laptops = df[
    (df['price($)'] <= 1500) & 
    (df['screen_size_inches'] < 15) & # Assuming smaller screen size for better portability
    (df['harddisk_GB'] > 512) # more space for storing the data
].copy()

# We will sort by price descending (assuming that higher-priced laptops within the budget might be of higher quality) and then by rating
customer_a_laptops.sort_values(by=['price($)', 'rating'], ascending=[False, False], inplace=True)

# Select top 5 laptops
top_5_customer_a = customer_a_laptops.head(5)


# Setting up the figure for four subplots
plt.figure(figsize=(18, 12))

# Visualization for Customer A: Price vs. Screen Size 
plt.subplot(2, 2, 1) 
scatter = sns.scatterplot(data=top_5_customer_a, x='screen_size_inches', y='price($)', hue='cpu', size='ram_GB', sizes=(100, 500), palette='Set2', legend='full')
plt.title('Top 5 Laptops for Customer A (Traveler): Price vs. Screen Size', fontsize=12)
plt.xlabel('Screen Size (inches)', fontsize=10)
plt.ylabel('Price ($)', fontsize=10)
plt.legend(title='CPU and RAM', title_fontsize='10', labelspacing=1.2)
plt.grid(True)

# labeling each point with the brand and model
for index, row in top_5_customer_a.iterrows():
    plt.text(row['screen_size_inches'], row['price($)'], f"{row['brand']} {row['model']}", horizontalalignment='center', size='small', color='black', weight='semibold')

# RAM Distribution
plt.subplot(2, 2, 2)
sns.histplot(laptop_price['ram_GB'], bins=10, kde=True)
plt.title('RAM Distribution')
plt.xlabel('RAM (GB)')
plt.ylabel('Frequency')

# Price Distribution
plt.subplot(2, 2, 3)
sns.histplot(laptop_price['price($)'], bins=10, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')


# Screen Size Distribution
plt.subplot(2, 2, 4)
sns.histplot(laptop_price['screen_size_inches'], bins=10, kde=True)
plt.title('Screen Size Distribution')
plt.xlabel('Screen Size (inches)')
plt.ylabel('Frequency')



# Filtering laptops for Customer A (Graphic Designer):
# For this scenario, we will assume that more harddisk space, bigger screensize, higher CPU speed, higher RAM and higher price (to some extent) could indicate better build quality and durability

customer_b_laptops = df[
    (df['graphics'] == 'Dedicated') & # Filtering for dedicated graphics
    (df['cpu_speed_GHz'] >=2.5)&# Assuming that 2.5GHz is the minimum for the graphic designer
    (df['ram_GB'] >= 16) & # Assuming that 24GB RAM is the minimum for graphic design work
    (df['harddisk_GB'] > 512)&# more space for storing the data
    (df['screen_size_inches'] >= 15)
].copy()

# We will sort by RAM and then by rating
df['rating'].fillna(0, inplace=True)
customer_b_laptops.sort_values(by=['cpu_speed_GHz', 'ram_GB', 'rating'], ascending=False, inplace=True)

# Select top 5 laptops
top_5_customer_b = customer_b_laptops.head(5)


plt.figure(figsize=(15, 12))

# Scatter plot for Price vs RAM for top 5 laptops
plt.subplot(2, 2, 1)
scatter = sns.scatterplot(data=top_5_customer_b, x='ram_GB', y='price($)', hue='cpu', size='screen_size_inches', sizes=(100, 500), palette='Set2', legend='full')
plt.title('Top 5 Laptops for Graphic Designer: Price vs. RAM', fontsize=12)
plt.xlabel('RAM (GB)', fontsize=10)
plt.ylabel('Price ($)', fontsize=10)
plt.legend(title='CPU and Screen Size', title_fontsize='10', labelspacing=1.2)
plt.grid(True)

# labeling each point with the brand and model
for index, row in top_5_customer_b.iterrows():
    plt.text(row['ram_GB'], row['price($)'], f"{row['brand']} {row['model']}", horizontalalignment='left', size='small', color='black', weight='semibold')


# Box plot for Price Distribution
plt.subplot(2, 2, 2)
sns.boxplot(x=df['price($)'])
plt.title('Price Distribution in All Laptops')
plt.xlabel('Price ($)')

# Box plot for Hard Disk Distribution
plt.subplot(2, 2, 3)
sns.boxplot(x=df['harddisk_GB'])
plt.title('Hard Disk Distribution in All Laptops')
plt.xlabel('Hard Disk (GB)')

# Box plot for RAM Distribution
plt.subplot(2, 2, 4)
sns.boxplot(x=df['ram_GB'])
plt.title('RAM Distribution in All Laptops')
plt.xlabel('RAM (GB)')

top_5_customer_a
top_5_customer_b
plt.tight_layout()
plt.show()