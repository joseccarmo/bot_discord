from youtube_transcript_api import YouTubeTranscriptApi
import re
import unicodedata
from urllib.parse import urlparse, parse_qs
import yt_dlp

def obter_ids_corrigidos_yt_dlp(url_canal: str, limite: int) -> list[str]:
    """
    Recebe a URL de um canal e retorna uma lista com os IDs dos primeiros N vídeos.
    Garante que a URL aponte para a aba de vídeos (/videos).
    """
    
    # Garante que a URL termina com /videos
    if not url_canal.endswith('/videos'):
        url_canal = url_canal.strip('/') + '/videos'
    
    # 2. Opções do yt-dlp
    ydl_opts = {
        'extract_flat': True,          # Extração plana, mais rápido
        'playlistend': limite,         # Limita a extração aos primeiros N vídeos
        'format': 'best',              # Configuração padrão de formato
        'skip_download': True,
        'quiet': True,                 # Suprime as mensagens de progresso desnecessárias
    }

    print(f"Buscando IDs na URL: {url_canal}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extrai as informações. O 'download=False' apenas evita o download real.
            info = ydl.extract_info(url_canal, download=False)
            
            # Percorre a lista de entradas e coleta o 'id'
            ids = [entry.get('id') 
                   for entry in info.get('entries', []) 
                   if entry and entry.get('id')]
            
            return ids
    except Exception as e:
        # A exceção agora será mais clara se houver um erro real
        print(f"Ocorreu um erro com yt-dlp: {e}")
        return []

# Exemplo de uso:
link_do_canal = "https://www.youtube.com/@manualdomundo"
primeiros_ids = obter_ids_corrigidos_yt_dlp(link_do_canal, 5)

print("\nIDs dos vídeos (via yt-dlp):")
for id_video in primeiros_ids:
    print(f"-> {id_video}")

i = 0

while i <= len(primeiros_ids):

    codigo =  primeiros_ids[i]
    ytt_api = YouTubeTranscriptApi().fetch(video_id=codigo, languages=['pt']) # chamada da trancriçao

    list_snippets = []

    # is iterable
    for snippet in ytt_api:
        if snippet.text[0] == '[':
            continue
        else:
            list_snippets.append(unicodedata.normalize('NFKD', snippet.text))

    text_snippets = ' '.join(list_snippets)
    frases_snippets = re.split(r'(?<=[.!?])\s+', text_snippets)

    # Imprime a lista de frases, cada uma em uma nova linha para melhor visualização.
    #print(frases_snippets)

    # --- 2. PREPARAÇÃO DO DATASET ---
    # O Trainer do Hugging Face funciona melhor com arquivos.
    # Vamos salvar nossa lista de frases em um arquivo .txt.
    nome_do_arquivo_dataset = f'transcricao/transcricao_{codigo}.txt'
    with open(nome_do_arquivo_dataset, "w", encoding="utf-8") as f:
        # Juntamos todas as frases com quebras de linha duplas para separar os pensamentos
        f.write("\n\n".join(frases_snippets))

    print(f"Dataset salvo em '{nome_do_arquivo_dataset}'")
    i += 1
