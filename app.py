from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rdflib import Graph
import nltk
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'img')

stopwords_indonesia = stopwords.words('indonesian')
additional_stopwords = ['baiknya', 'berkali', 'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya']
stopwords_indonesia.extend(additional_stopwords)

def extract_label(uri):
    """Mengambil label dari URI RDF."""
    return uri.split('#')[-1]

def custom_tokenizer(text):
    tokens = re.findall(r'\b\w\w+\b', text.lower())
    tokens = [word for word in tokens if word not in stopwords_indonesia]
    return tokens


g = Graph()
g.parse('parfumrevisi.rdf', format='xml')
stemmer = PorterStemmer()

def extract_label(uri):
    """Mengambil label dari URI RDF."""
    return uri.split('#')[-1]

query = """
PREFIX parf: <http://www.semanticweb.org/asus/ontologies/2024/8/untitled-ontology-47#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?Parfum ?Varian ?Rating ?Harga ?Untuk ?KarakterWangi ?Gambar ?Komentar1 ?Komentar2 ?Komentar3 ?Volume ?Nama
WHERE {
    ?Parfum parf:targetGender ?Untuk .
    ?Parfum parf:memilikiVarian ?Varian .
    ?Parfum parf:memilikiRating ?Rating .
    ?Parfum parf:memilikiHarga ?Harga .
    ?Parfum parf:memilikiJenisAroma ?KarakterWangi .
    ?Parfum parf:memilikiGambar ?Gambar .
    ?Parfum parf:memilikiKomentar1 ?Komentar1 . 
    ?Parfum parf:memilikiKomentar2 ?Komentar2 . 
    ?Parfum parf:memilikiKomentar3 ?Komentar3 . 
    ?Parfum parf:memilikiVolume ?Volume .
    ?Parfum parf:memilikiNama ?Nama .
    
}
"""

results = g.query(query)

data = []
for row in results:
    data.append({
        'Jenis_Kelamin':  extract_label(str(row.Untuk)),
        'Varian': str(row.Varian),
        'Rating': str(row.Rating),
        'Ukuran_Botol': str(row.Volume),
        'Jenis_Parfum': str(row.Nama),
        'Harga': str(row.Harga),
        'Karakter_Wangi':  extract_label(str(row.KarakterWangi)),
        'Komentar_1': str(row.Komentar1),
        'Komentar_2': str(row.Komentar2),
        'Komentar_3': str(row.Komentar3),
        'Gambar': str(row.Gambar)
    })

df = pd.DataFrame(data)

df['Textual_Data'] = (
    (df['Jenis_Kelamin'] + " ") * 2 +  
    (df['Varian'] + " ") * 3 +         
    (df['Jenis_Parfum'] + " ") * 3 +   
    (df['Rating'].astype(str) + " ") * 2 + 
    (df['Karakter_Wangi'] + " ") * 2 +
    df['Harga'].astype(str) + " " +
    df['Komentar_1'] + " " +
    df['Komentar_2'] + " " +
    df['Komentar_3'] + " " +
    df['Gambar']
)


vectorizer = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    ngram_range=(1, 2),
    max_df=0.8,
    min_df=2
)
tfidf_matrix = vectorizer.fit_transform(df['Textual_Data'])

# def search_perfume(query, min_price=0, max_price=float('inf')):
    
#     # query = stemmer.stem(query)  
#     query_tfidf = vectorizer.transform([query])
#     cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
#     df_filtered = df[(df['Harga'] >= min_price) & (df['Harga'] <= max_price)]
#     df_filtered = df_filtered.reset_index(drop=True)
#     cosine_similarities_filtered = cosine_similarity(query_tfidf, vectorizer.transform(df_filtered['Textual_Data'])).flatten()
#     top_indices = cosine_similarities_filtered.argsort()[-15:][::-1]  # Ambil 3 hasil teratas
#     results = df_filtered.iloc[top_indices].copy()
#     results['Similarity'] = cosine_similarities_filtered[top_indices]
#     return results

def search_perfume(query, min_price=0, max_price=float('inf')):
    # Convert price strings to float after removing dots for comparison
    df['Harga_Float'] = df['Harga'].str.replace('.', '').astype(float)

    # Filter based on the numeric prices
    df_filtered = df[(df['Harga_Float'] >= min_price) & (df['Harga_Float'] <= max_price)]
    
    # Perform the same cosine similarity search on the filtered dataframe
    query_tfidf = vectorizer.transform([query])
    cosine_similarities_filtered = cosine_similarity(query_tfidf, vectorizer.transform(df_filtered['Textual_Data'])).flatten()
    top_indices = cosine_similarities_filtered.argsort()[-15:][::-1]
    
    results = df_filtered.iloc[top_indices].copy()
    results['Similarity'] = cosine_similarities_filtered[top_indices]
    
    return results


# def search_exact_match(query, min_price=0, max_price=float('inf')):
#     query_words = query.lower().strip().split()
    
#     df_filtered = df[(df['Harga'] >= min_price) & (df['Harga'] <= max_price)]
    
#     def row_matches(row):
#         combined_text = (
#             row['Jenis_Kelamin'].lower() + " " +
#             row['Varian'].lower() + " " +
#             row['Jenis_Parfum'].lower() + " " +
#             row['Karakter_Wangi'].lower()
#         )
#         return all(word in combined_text for word in query_words)
    
#     exact_match_results = df_filtered[df_filtered.apply(row_matches, axis=1)]
    
#     return exact_match_results


@app.route('/')
def index():
    top_rated_perfumes = df.sort_values(by='Rating', ascending=False).head(10).to_dict(orient='records')
    return render_template('index.html', top_rated_perfumes=top_rated_perfumes)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['search_query']
    min_price = float(request.form.get('min_price', 0))
    max_price = float(request.form.get('max_price', float('inf')))
    # search_results = search_perfume(query, min_price, max_price)

    # return render_template('results.html', perfumes=search_results.to_dict(orient='records'))
    search_results_cosine = search_perfume(query, min_price, max_price)
    
    # Hasil pencarian berdasarkan Exact Match
    # search_results_exact = search_exact_match(query, min_price, max_price)
    
    return render_template('results.html', 
                           perfumes_cosine=search_results_cosine.to_dict(orient='records'))
                        #    perfumes_exact=search_results_exact.to_dict(orient='records'))


# @app.route('/filter_by_price', methods=['POST'])
# def filter_by_price():
#     min_price = request.form.get('min_price', type=float)
#     max_price = request.form.get('max_price', type=float)

#     if min_price is not None and max_price is not None:
#         filtered_perfumes = df[(df['Harga'].astype(float) >= min_price) & (df['Harga'].astype(float) <= max_price)]
#     elif min_price is not None and max_price is None:
#         filtered_perfumes = df[df['Harga'].astype(float) >= min_price]
#     elif max_price is not None and min_price is None:
#         filtered_perfumes = df[df['Harga'].astype(float) <= max_price]
#     else:
#         filtered_perfumes = df  # Tampilkan semua jika tidak ada input

#     return render_template('filtered.html', perfumes=filtered_perfumes.to_dict(orient='records'))

@app.route('/filter_by_price', methods=['POST'])
def filter_by_price():
    min_price = request.form.get('min_price', type=float)
    max_price = request.form.get('max_price', type=float)
    
    # Convert price strings to float after removing dots for comparison
    df['Harga_Float'] = df['Harga'].str.replace('.', '').astype(float)

    if min_price is not None and max_price is not None:
        filtered_perfumes = df[(df['Harga_Float'] >= min_price) & (df['Harga_Float'] <= max_price)]
    elif min_price is not None and max_price is None:
        filtered_perfumes = df[df['Harga_Float'] >= min_price]
    elif max_price is not None and min_price is None:
        filtered_perfumes = df[df['Harga_Float'] <= max_price]
    else:
        filtered_perfumes = df  # Tampilkan semua jika tidak ada input

    return render_template('filtered.html', perfumes=filtered_perfumes.to_dict(orient='records'))


@app.route('/detail/<jenis_parfum>')
def detail(jenis_parfum):
    parfum_detail = df[df['Jenis_Parfum'] == jenis_parfum].to_dict(orient='records')[0]
    return render_template('detail.html', parfum=parfum_detail)

if __name__ == '__main__':
    app.run(debug=True)
