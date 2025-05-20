import requests
import pandas as pd
import time
import json

areas = [
    'quadronno - crocetta', 'farini', 'palestro', 'repubblica',
    'morgagni', 'centrale', 'frua', 'piazzale siena', 'lodi - brenta',
    'lanza', 'bovisa', 'ripamonti', 'cascina merlata - musocco',
    'bignami - ponale', 'buenos aires', 'dergano', 'plebisciti - susa',
    'paolo sarpi', 'brera', 'montenero', 'guastalla', 'pasteur',
    'piave - tricolore', 'turro', 'gambara', 'maggiolina',
    "porta romana - medaglie d'oro", 'giambellino', 'moscova',
    'città studi', 'cermenate - abbiategrasso', 'porta nuova',
    'istria', 'rubattino', 'barona', 'isola', 'missori',
    'monte rosa - lotto', 'solari', 'scala - manzoni', 'gallaratese',
    'turati', 'corso san gottardo', 'carrobbio', 'washington',
    'sempione', 'quarto cagnino', 'parco trotter', 'pezzotti - meda',
    'cadorna - castello', 'quartiere adriano', 'udine',
    'garibaldi - corso como', 'cenisio', 'bisceglie', 'corvetto',
    'tripoli - soderini', 'vincenzo monti', 'martini - insubria',
    'porta vittoria', 'argonne - corsica', 'niguarda',
    'molise - cuoco', 'arena', 'zara', 'trenno', 'duomo',
    'navigli - darsena', 'porta venezia', 'de angeli', 'precotto',
    'piazza napoli', 'indipendenza', 'san siro', 'rovereto',
    'vigentino - fatima', 'corso genova', 'arco della pace',
    'san vittore', 'bocconi', 'vercelli - wagner', 'cadore',
    'bande nere', 'cascina dei pomi', 'dezza', 'villa san giovanni',
    'ticinese', 'portello - parco vittoria', 'chiesa rossa', 'qt8',
    'amendola - buonarroti', 'certosa', 'bicocca', 'pagano', 'inganni',
    'crescenzago', 'viale ungheria - mecenate', 'santa giulia',
    "sant'ambrogio", 'gorla', 'ascanio sforza', 'famagosta',
    'quartiere forlanini', 'greco - segnano', 'cimiano', 'casoretto',
    'affori', 'quadrilatero della moda', 'tre castelli - faenza',
    'quarto oggiaro', 'bologna - sulmona', 'ghisolfa - mac mahon',
    'ponte nuovo', 'quartiere feltre', 'melchiorre gioia', 'city life',
    'lambrate', 'ponte lambro', 'san babila', 'baggio', 'lorenteggio',
    'cantalupa - san paolo', 'gratosoglio', 'rogoredo', 'comasina',
    "ca' granda", 'prato centenaro', 'san carlo',
    'borgogna - largo augusto', 'primaticcio', 'ortica', 'bruzzano'
]

def get_coordinates(api_key, area):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    search_query = f"{area}, Milan, Italy"
    
    params = {
        'address': search_query,
        'key': api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            print(f"Error for {area}: {data['status']}")
            return None, None
    except Exception as e:
        print(f"Error for {area}: {str(e)}")
        return None, None

def main():
    api_key = input("Please enter your Google Maps API key: ")
    
    coordinates_data = []
    for area in areas:
        print(f"Getting coordinates for {area}...")
        lat, lng = get_coordinates(api_key, area)
        coordinates_data.append({
            'area': area,
            'latitude': lat,
            'longitude': lng
        })
        time.sleep(0.5)  # Be nice to the API
    
    df = pd.DataFrame(coordinates_data)
    df.to_csv('milan_areas_coordinates.csv', index=False)
    print("CSV file has been created successfully!")

if __name__ == "__main__":
    main() 