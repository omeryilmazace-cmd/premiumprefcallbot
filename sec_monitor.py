import logging
import sys
import re
import json
import time
import os
import requests
import xml.etree.ElementTree as ET        
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yfinance as yf
import json
from collections import deque

# Temel loglama ayarlari
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s', handlers=[logging.FileHandler('sec_monitor.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Windows'ta karakter kodlamasi sorunu icin
if sys.platform == "win32":
    import os
    os.system('chcp 65001 > nul')

from dotenv import load_dotenv
load_dotenv()

@dataclass
class Company:
    cik: str
    ticker: str
    name: str

class AIAnalyzer:
    def __init__(self):
        # Groq API
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.daily_analysis_count = 0
        self.max_daily_analysis = 14000

    def analyze_debt_redemption(self, company_name: str, ticker: str, form_type: str, content: str) -> str:
        if self.daily_analysis_count >= self.max_daily_analysis:
            return "❌ Gunluk AI analiz limiti doldu"
            
        clean_content = self._extract_relevant_text(content)
        if len(clean_content) < 50:
            return "⚠️ Yetersiz icerik"
            
        try:
            analysis = self._analyze_debt_with_groq(company_name, ticker, form_type, clean_content)
            if analysis:
                self.daily_analysis_count += 1
                return analysis
            return self._simple_debt_analysis(content)
        except Exception as e:
            logger.error(f"AI debt analysis error: {e}")
            return "❌ AI analiz hatasi"

    def _clean_text(self, text: str) -> str:
        # HTML taglarini temizle
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        # HTML entity'leri temizle
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        # Coklu bosluklari tek bosluga indir
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_relevant_text(self, content: str) -> str:
        """
        Gelistirilmis akilli chunking algoritmasi.
        """
        content = self._clean_text(content)
        
        # Cumlelere veya mantikli bloglara bol
        # Sadece nokta ile degil, uzun bosluklarla da boluyoruz ki listeler karismasin
        chunks = re.split(r'(?<=[.!?])\s+', content)
        
        scored_chunks = []
        keywords = {
            'redeem': 5, 'redemption': 5, 'call': 3, 'retire': 3, 'repay': 2,
            'prepay': 2, 'senior notes': 4, 'debentures': 4, 'preferred': 3,
            'notice': 2, 'outstanding': 2, 'principal': 2, 'maturity': 2,
            'plan': 2, 'intend': 2, 'will': 1, 'expect': 1,
            # Positive News Keywords
            'tender': 5, 'offer to purchase': 5, 'repurchase': 4, 'buyback': 4,
            'reinstatement': 4, 'dividend': 3, 'authorization': 2,
            'special': 3, 'accumulated': 3, 'unpaid': 3, 'arrears': 4, 'catch-up': 4
        }
        
        for i, chunk in enumerate(chunks):
            if len(chunk) < 20: continue # Cok kisa parcalari atla
            
            score = 0
            chunk_lower = chunk.lower()
            
            for word, points in keywords.items():
                if word in chunk_lower:
                    score += points
            
            # Tarih iceriyorsa bonus puan
            if re.search(r'202[4-9]', chunk):
                score += 2
                
            # Tutar iceriyorsa bonus puan
            if '$' in chunk or 'million' in chunk_lower:
                score += 1
                
            if score > 0:
                # Context icin onceki ve sonraki paragrafi da ekleyebiliriz ama
                # simdilik sadece mevcut cumleyi aliyoruz
                scored_chunks.append((score, i, chunk))
        
        # Puana gore sirala, en yuksek puanli 10 parcayi al
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored_chunks[:10]
        
        # Orijinal siralamaya gore tekrar diz (metin akisi bozulmasin)
        top_chunks.sort(key=lambda x: x[1])
        
        result = ' '.join([c[2] for c in top_chunks])
        
        # Limit kontrolu (Groq icin makul seviye)
        if len(result) > 6000:
            result = result[:6000] + "..."
            
        return result

    def _analyze_debt_with_groq(self, company_name: str, ticker: str, form_type: str, content: str) -> Optional[str]:
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
SENIOR FINANCIAL ANALYST GOREVI:
{company_name} ({ticker}) sirketinin {form_type} dosyasindan alinan asagidaki metni analiz et.

=== METIN ===
{content}

=== SORU ===
Bu metinde sirket borclariyla veya hisseleriyle ilgili OLUMLU bir gelisme var mi?
Ozellikle sunlari ara:
1. REDEMPTION: Borc/imtiyazli hisse geri cagirma.
2. TENDER OFFER: Hisseleri/borclari sabit fiyattan geri alma teklifi.
3. REPURCHASE/BUYBACK: Geri alim programi aciklamasi.
4. DIVIDEND: Temettu devami/artisi/tekrar baslamasi.

=== CEVAP FORMATI ===
Kisa ve net olmali.
Tur: [REDEMPTION / TENDER OFFER / REPURCHASE / DIVIDEND / SUPHELI]
Detay: [Neyi, ne zaman, ne fiyata? Kisa ozet.]
Kanit: [Metinden kisa bir alinti]

Eger onemli bir gelisme yoksa "YOK" de.
"""
            
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "You are an expert financial analyst. Be concise and strict."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.1
            }
            
            response = requests.post(self.groq_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return f"🤖 **AI Analizi**:\n{result['choices'][0]['message']['content'].strip()}"
            elif response.status_code == 429:
                return "⚠️ AI Limit"
            else:
                return None
                
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return None

    def _simple_debt_analysis(self, content: str) -> str:
        # Yedek basit analiz
        if 'redeem' in content.lower() or 'redemption' in content.lower():
            return "🤖 **Analiz**: Otomatik analiz basarisiz ama metinde 'redemption' kelimeleri var. Manuel kontrol gerekebilir."
        return "🤖 **Analiz**: AI devre disi."

class SECCallMonitor:
    def __init__(self, tg_token: str, chat_id: str, tickers: Optional[List[str]] = None):
        self.tg_token, self.chat_id = tg_token, chat_id
        
        # Persistence (Railway Volume destegi)
        # Eger /app/data varsa orayi kullan (Railway Volume), yoksa bulundugun klasor
        self.data_dir = Path("/app/data") if Path("/app/data").exists() else Path(".")
        self.processed_file = self.data_dir / "processed.json"
        
        self.processed = self._load_processed()
        
        self.headers = {"User-Agent": "Omer Yilmaz (omeryilmaz1329@gmail.com)", "Accept-Encoding": "gzip, deflate"}
        self.check_cnt = 0
        self.last_daily_report = datetime.now().date()
        self.start = datetime.now()
        self.cache_file = self.data_dir / "cik_cache.json"
        
        # Ticker Yonetimi (Dinamik)
        self.tickers_file = self.data_dir / "tickers.json"
        
        if self.tickers_file.exists():
            try:
                with open(self.tickers_file, 'r', encoding='utf-8') as f:
                    self.active_tickers = json.load(f)
                    logger.info(f"Loaded {len(self.active_tickers)} tickers from disk.")
            except:
                self.active_tickers = tickers if tickers else []
        else:
             self.active_tickers = tickers if tickers else []
             self._save_active_tickers()
             
        self.companies = self._init_companies(self.active_tickers)
        self.target_ciks = {c.cik for c in self.companies} if self.companies else set()
        
        # Istatistikler
        self.daily_stats = {
            "total": 0, "processed": 0, "errors": 0, 
            "ai_analyzed": 0, "instant_detections": 0,
            "start_time": datetime.now().isoformat()
        }
        self.ai_limit_notified = False
        self.last_scanned = deque(maxlen=10) # Son taranan dosyalari tutmak icin
        self.last_watched = deque(maxlen=10) # Sadece izlenen sirketlerin dosyalari
        self.last_success_check = datetime.now()
        
        # Link dosyalari
        self.links_txt = self.data_dir / "sec_links.txt"
        self.links_json = self.data_dir / "sec_links.json" 
        self.links_csv = self.data_dir / "sec_links.csv"
        
        if not self.links_csv.exists():
            with open(self.links_csv, 'w', encoding='utf-8') as f:
                f.write("Timestamp,Company,Ticker,Form,Alert_Type,Link,Details\n")
        
        self.ai_analyzer = AIAnalyzer()
        
        # ThreadPoolExecutor ile paralel isleme
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"Bot ready check interval: 30s")
        self._send_msg(f"🚀<b>SEC Bot v2.0 Started!</b>\n⚡<b>Parallel Processing:</b> Active\n🧠<b>Smart AI Context:</b> Active\n📊<b>Companies:</b> {len(self.companies)}")

    def _init_companies(self, tickers: Optional[List[str]]) -> List[Company]:
        if not tickers: return []
        try:
            r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=self.headers, timeout=10)
            if r.status_code == 200:
                # Mapping: Ticker -> Data
                mapping = {v['ticker']: {'cik': str(v['cik_str']).zfill(10), 'name': v['title']} for v in r.json().values()}
                
                final_companies = []
                for t in tickers:
                    t_upper = t.upper()
                    
                    # 1. Tam eslesme
                    if t_upper in mapping:
                        final_companies.append(Company(mapping[t_upper]['cik'], t_upper, mapping[t_upper]['name']))
                        continue
                        
                    # 2. Kok Ticker Bulma (Pref Hisseler icin)
                    # Ornek: METCZ -> MET, HWM-P -> HWM
                    # Tire veya noktadan bolmeyi dene
                    root = re.split(r'[-.]', t_upper)[0]
                    if len(root) < len(t_upper) and root in mapping:
                        # Ana sirket CIK'ini kullaniyoruz ama Ticker olarak kullanicinin verdigini (Pref) sakliyoruz
                        final_companies.append(Company(mapping[root]['cik'], t_upper, mapping[root]['name']))
                        continue
                        
                    # 3. Son caresiz 4. harfi atma (bazi eski kisaltmalar)
                    if len(t_upper) > 3 and t_upper[:-1] in mapping:
                         root = t_upper[:-1]
                         final_companies.append(Company(mapping[root]['cik'], t_upper, mapping[root]['name']))
                         
                return final_companies
        except Exception as e: 
            logger.error(f"Init companies error: {e}")
        return []

    def _load_processed(self) -> set:
        try:
            if self.processed_file.exists():
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} processed entries from disk.")
                    return set(data)
        except Exception as e:
            logger.warning(f"Failed to load processed history: {e}")
        return set()

    def _save_processed(self):
        try:
            with open(self.processed_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed), f)
        except Exception as e:
            logger.error(f"Failed to save processed history: {e}")

    def _save_active_tickers(self):
        try:
            with open(self.tickers_file, 'w', encoding='utf-8') as f:
                json.dump(self.active_tickers, f)
        except Exception as e:
            logger.error(f"Failed to save tickers: {e}")

    def _send_msg(self, msg: str):
        try:
            if len(msg) > 4000: msg = msg[:4000] + "..."
            self._save_link_from_msg(msg)
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", 
                         data={"chat_id": self.chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
        except Exception as e: logger.error(f"Telegram error: {e}")

    def _save_link_from_msg(self, msg: str):
        # Basit link kaydetme
        try:
            link_match = re.search(r"href='(.*?)'", msg)
            if link_match:
                with open(self.links_txt, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now()}: {link_match.group(1)}\n")
        except: pass

    def _fetch_content(self, url: str) -> str:
        try:
            # 8-K vb icin index sayfasini cozme mantigi
            if url.endswith('-index.htm'):
                r = requests.get(url, headers=self.headers, timeout=15)
                # Basitce en buyuk htm dosyasini bulmaya calisalim
                matches = re.findall(r'href="([^"]+\.htm)"', r.text)
                for m in matches:
                    if 'ix?doc=' in m: continue # ixbrl viewer linki degil
                    if m.endswith('-index.htm'): continue
                    # Form dosyasini tahmin et
                    full_url = urljoin(url, m)
                    return self._fetch_content_direct(full_url)
            return self._fetch_content_direct(url)
        except: return ""

    def _fetch_content_direct(self, url: str) -> str:
        r = requests.get(url, headers=self.headers, timeout=20)
        return r.text

    def _fetch_with_exhibits(self, cik: str, accession_number: str, primary_doc: str) -> Optional[str]:
        """Ana dokuman + Exhibit 99 (Basin bulteni) icerigini getirir"""
        acc_clean = accession_number.replace('-', '')
        cik_clean = str(int(cik))
        
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_clean}"
        main_url = f"{base_url}/{primary_doc}"
        
        # Ana dokuman
        main_content = self._fetch_content(main_url)
        if not main_content: return None
        
        combined_content = main_content
        
        # Exhibitleri kontrol et (index.json)
        try:
            index_url = f"{base_url}/index.json"
            r = requests.get(index_url, headers=self.headers, timeout=10)
            if r.status_code == 200:
                items = r.json().get('directory', {}).get('item', [])
                for item in items:
                    name = item.get('name', '')
                    # Exhibit 99 (Press Release) - htm/html uzantili olanlar
                    if ('ex99' in name.lower() or 'ex-99' in name.lower()) and name.endswith(('htm', 'html')):
                        ex_url = f"{base_url}/{name}"
                        ex_content = self._fetch_content(ex_url)
                        if ex_content:
                            combined_content += "\n\n=== EXHIBIT 99 ===\n\n" + ex_content
                            logger.info(f"Exhibit found and appended: {name}")
        except Exception as e:
            logger.error(f"Exhibit fetch error: {e}")
            
        return combined_content

    def _process_filing(self, entry) -> None:
        """Tek bir dosyanin indirilip islenmesi (Thread icinde calisacak)"""
        try:
            title = entry['title']
            link = entry['link']
            cik = entry['cik']
            form_type = entry['form']
            
            # Filtreleme
            if self.target_ciks and cik not in self.target_ciks: return
            
            logger.info(f"Processing {title}...")
            
            # Link'ten Accession Number ve Primary Doc parse et
            # Ornek: https://www.sec.gov/Archives/edgar/data/1582982/000149315225025178/form8-k.htm
            match = re.search(r'data/(\d+)/(\d+)/(.+)', link)
            if match:
                # _fetch_with_exhibits kullan (Accession number dashesiz geliyor, okey)
                # cik zaten int gelmisti string yapalim
                # match[2] accession without dashes
                content = self._fetch_with_exhibits(match.group(1), match.group(2), match.group(3))
            else:
                content = self._fetch_content(link)
                
            if not content: return

            comp = next((c for c in self.companies if c.cik == cik), None)
            comp_name = f"{comp.name} ({comp.ticker})" if comp else f"CIK {cik}"
            comp_ticker = comp.ticker if comp else "UNKNOWN"

            # 1. Gelismis Yapisal Tespit (AI'siz ama akilli)
            detected_sentence = self._smart_keyword_detection(content)
            
            # /last komutu icin ozet cikar
            summary = "📄 Standart Bildirim"
            if detected_sentence:
                summary = f"⚡ {detected_sentence[:70]}..."
            else:
                 # Item basligi yakalamaya calis
                 m = re.search(r'(Item\s+\d+\.\d+[^<\n]*)', content)
                 if m: summary = f"ℹ️ {m.group(1)[:50]}..."
                 
            # Listeye ekle
            already = any(x['link'] == link for x in self.last_watched)
            if not already:
                self.last_watched.appendleft({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'title': title,
                    'link': link,
                    'summary': summary
                })
            
            if detected_sentence:
                self.daily_stats["instant_detections"] += 1
                
                # AI Analizi yap
                ai_analysis = ""
                if comp and self.ai_analyzer.daily_analysis_count < self.ai_analyzer.max_daily_analysis:
                    ai_analysis = self.ai_analyzer.analyze_debt_redemption(comp.name, comp.ticker, form_type, content)
                
                # AI Limit Kontrolu
                if ai_analysis == "⚠️ AI Limit":
                    if not self.ai_limit_notified:
                        self._send_msg("⚠️ <b>UYARI:</b> Gunluk AI analiz limiti doldu! Gun sonuna kadar sadece kelime bazli tespit yapilacak.")
                        self.ai_limit_notified = True
                    ai_analysis = "⚠️ (Limit Doldu)"

                msg = f"⚡ <b>TESPIT EDILDI</b>\n"
                msg += f"<b>Sirket:</b> {comp_name}\n"
                msg += f"<b>Form:</b> {form_type}\n"
                msg += f"<b>Cumle:</b> <i>{detected_sentence}</i>\n"
                
                # Fiyat Bilgisi
                price_str = self._get_market_price(comp_ticker)
                msg += f"<b>Fiyat:</b> {price_str}\n"
                
                if ai_analysis and "YOK" not in ai_analysis and "❌" not in ai_analysis:
                    msg += f"\n{ai_analysis}\n"
                
                msg += f"<b>Link:</b> <a href='{link}'>Dosyayi Ac</a>"
                self._send_msg(msg)
                logger.info(f"Hit found: {comp_name}")

        except Exception as e:
            logger.error(f"Process filing error: {e}")
            self.daily_stats["errors"] += 1

    def _get_market_price(self, ticker: str) -> str:
        """Yahoo Finance uzerinden fiyat ceker (Pref hisse formatlarini dener)"""
        try:
            candidates = [ticker, ticker.replace('.', '-'), ticker.replace('-', '.')]
            root = re.split(r'[-.]', ticker)[0]
            if root != ticker: candidates.append(root)
            
            for symbol in candidates:
                try:
                    t = yf.Ticker(symbol)
                    hist = t.history(period="1d")
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]
                        info_price = t.info.get('regularMarketPrice') or t.info.get('currentPrice')
                        current = info_price if info_price else price
                        sym_display = symbol if symbol == ticker else f"{symbol} (KOK)"
                        return f"{current:.2f} USD ({sym_display})"
                except: continue
            return "Fiyat Bulunamadi"
        except: return "Hata"

    def _smart_keyword_detection(self, content: str) -> Optional[str]:
        """Yanlis pozitifleri eleyen akilli tespit fonksiyonu"""
        clean_text = self.ai_analyzer._clean_text(content)
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        
        keywords = [
            'redeem', 'redemption', 'call for redemption', 'retire',
            'full call', 'partial call', 'notice of call', 'call',
            'tender offer', 'offer to purchase', 'repurchase', 'buyback', 'tender',
            'dividend reinstatement', 'authorized a new', 'dividend', 'distribution',
            'special dividend', 'accumulated', 'unpaid', 'arrears', 'catch-up',
            'liquidation', 'delisting', 'suspension', 'resumption'
        ]
        exclude_phrases = ['may redeem', 'option to redeem', 'right to redeem', 'upon redemption', 'subject to redemption', 'will call a conference', 'call will be']
        
        for sentence in sentences:
            s_lower = sentence.lower()
            
            # Anahtar kelime kontrolu
            if not any(k in s_lower for k in keywords): continue

            # HEDEF KITLE KONTROLU (Esnek: Preferred, Debt, Stock, vb)
            context_keywords = [
                'preferred', 'preference', 'depositary', 'series', 'notes', 
                'senior', 'subordinated', 'debentures', '%', 'stock', 'shares',
                'units', 'capital', 'security', 'bond', 'dividend'
            ]
            if not any(ctx in s_lower for ctx in context_keywords):
                continue
            
            # Uzunluk ve yapi kontrolu (Daha uzun cumlelere izin ver)
            if len(sentence) < 25 or len(sentence) > 1000: continue
            
            # Baslik gibi mi? (Tum harfler buyukse veya sonu noktayla bitmiyorsa supheli)
            # Ama bazen basliklarda da onemli bilgi olabilir, o yuzden sadece cok kisa basliklari eliyoruz
            
            # Gelecek zaman veya kesinlik kontrolu
            # 'announces' -> 'announce' ( catch announced), +notified, intention, elected, decided, proposed, etc.
            strong_indicators = [
                'will', 'announce', 'notice', 'planned', 'scheduled', 'date', 
                'notified', 'intention', 'elected', 'proposed', 'decided', 
                'declared', 'announcement', 'agreement', 'vote', 'board'
            ]
            if not any(i in s_lower for i in strong_indicators):
                # Eger "notice of" veya "notif" kalibi varsa kesinlik vardir (daha esnek)
                if "notice of" not in s_lower and "notif" not in s_lower:
                    continue
            
            # Dislama kelimeleri (ihtimal belirtenler)
            if any(ex in s_lower for ex in exclude_phrases): continue
            
            # Ekstra guvenlik: Sadece liste maddesi mi?
            if re.match(r'^\s*\(?\w\)', sentence): continue
            
            return sentence.strip()
            
        return None

    def _fetch_feed_url(self, url: str) -> Optional[bytes]:
        try:
            for attempt in range(3):
                try:
                    r = requests.get(url, headers=self.headers, timeout=25)
                    if r.status_code == 200:
                        return r.content
                    elif r.status_code == 429:
                        time.sleep(5 * (attempt + 1))
                    else:
                        time.sleep(2)
                except: time.sleep(2)
        except: pass
        return None

    def _check_feed(self):
        try:
            # RSS/Atom feed'i cek
            urls = [
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-k&count=100&output=atom",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=424b5&count=50&output=atom",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=497ad&count=50&output=atom",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-k&count=20&output=atom",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=sc to-t&count=20&output=atom",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=sc to-i&count=20&output=atom",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&count=100&output=atom" # Catch-All (Tum formlar)
            ]
            
            entries_to_process = []
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            # Paralel Feed Cekimi (Blocklanmayi onlemek icin)
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(self._fetch_feed_url, url): url for url in urls}
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    content = future.result()
                    
                    if not content:
                        logger.warning(f"Failed to fetch feed: {url}")
                        continue
                        
                    try:
                        root = ET.fromstring(content)
                        entries = root.findall(".//atom:entry", ns)
                        
                        for entry in entries:
                            title = entry.find("atom:title", ns).text
                            link = entry.find("atom:link", ns).attrib['href']
                            
                            entry_id = f"{link}_{title}"
                            
                            # CIK ve Form Type cikarimi
                            cik_match = re.search(r'(\d{10})', title)
                            cik = cik_match.group(1) if cik_match else ""
                            
                            # Basit form type tespiti
                            form = "UNKNOWN"
                            if "8-K" in title: form = "8-K"
                            elif "424B5" in title: form = "424B5"
                            elif "10-K" in title: form = "10-K"
                            elif "497" in title: form = "497AD"
                            if "SC TO" in title: form = "SC TO"
                            
                            is_watched = cik in self.target_ciks if cik else False
                            
                            # Izlenen sirket ise ve ZATEN islendiyse buraya ekle (Yeni ise process_filing ekleyecek)
                            if is_watched and entry_id in self.processed:
                                already_in_watched = any(x['link'] == link for x in self.last_watched)
                                if not already_in_watched:
                                    self.last_watched.appendleft({
                                        'time': datetime.now().strftime("%H:%M:%S"),
                                        'title': title,
                                        'link': link,
                                        'summary': f"✅ {title[:50]} (Arsivde)"
                                    })

                            # Debug icin son gorulenleri kaydet
                            already_in_last = any(x['link'] == link for x in self.last_scanned)
                            if not already_in_last:
                                self.last_scanned.appendleft({
                                    'time': datetime.now().strftime("%H:%M:%S"),
                                    'title': title,
                                    'link': link,
                                    'is_watched': is_watched
                                })

                            if entry_id in self.processed: continue
                            
                            self.processed.add(entry_id)
                            
                            entries_to_process.append({
                                'title': title, 'link': link, 'cik': cik, 'form': form
                            })
                    except Exception as e:
                         logger.error(f"Feed parse error {url}: {e}")
            
            self.last_success_check = datetime.now() # Son basarili kontrol zamani
            
            # Paralel isleme baslat
            if entries_to_process:
                logger.info(f"Processing {len(entries_to_process)} new entries with threads...")
                futures = [self.executor.submit(self._process_filing, e) for e in entries_to_process]
                # Hepsini bekle
                for future in as_completed(futures):
                    pass 
                logger.info("Batch processing complete.")
                
            self._cleanup()
            
        except Exception as e:
            logger.error(f"Main loop error: {e}")

    def _cleanup(self):
        if len(self.processed) > 5000:
            self.processed = set(list(self.processed)[-1000:])
        self._save_processed()

    def _poll_commands(self):
        """Telegram komutlarini dinleyen thread (non-blocking)"""
        offset = 0
        while True:
            try:
                # Long polling
                url = f"https://api.telegram.org/bot{self.tg_token}/getUpdates?offset={offset}&timeout=30"
                r = requests.get(url, timeout=45)
                if r.status_code == 200:
                    data = r.json()
                    for update in data.get('result', []):
                        offset = update['update_id'] + 1
                        
                        if 'message' in update and 'text' in update['message']:
                            text = update['message']['text']
                            chat_id = update['message']['chat']['id']
                            
                            if text.startswith('/scan '):
                                try:
                                    ticker = text.split(' ')[1].upper().strip()
                                    threading.Thread(target=self._handle_scan, args=(chat_id, ticker)).start()
                                except: pass
                            elif text.startswith('/add '):
                                try:
                                    ticker = text.split(' ')[1].upper().strip()
                                    self._handle_add(chat_id, ticker)
                                except: pass
                            elif text.startswith('/remove '):
                                try:
                                    ticker = text.split(' ')[1].upper().strip()
                                    self._handle_remove(chat_id, ticker)
                                except: pass
                            elif text.strip() == '/list':
                                self._handle_list(chat_id)
                            elif text.strip() == '/last':
                                self._handle_last(chat_id)
                                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Poller error: {e}")

    def _handle_last(self, chat_id: int):
        last_check = self.last_success_check.strftime("%H:%M:%S") if self.last_success_check else "Henuz yok"
        msg = f"⏱️ <b>Sistem Durumu</b>\n"
        msg += f"Son Kontrol: {last_check}\n"
        msg += f"Aktif Ticker: {len(self.active_tickers)}\n\n"
        
        msg += "📂 <b>Izlenen Sirketlerden Son Dosyalar:</b>\n"
        if not self.last_watched:
             msg += "(Henuz izlenen sirketlerden dosya gelmedi)"
        else:
            for item in list(self.last_watched):
                summ = item.get('summary', '...')
                msg += f"• 🎯 [{item['time']}] <a href='{item['link']}'>{item['title'][:20]}...</a>\n   └ <i>{summ}</i>\n"
        
        self._send_msg(msg)

    def _handle_add(self, chat_id: int, ticker: str):
        if ticker in self.active_tickers:
            self._send_msg(f"ℹ️ {ticker} zaten listede.")
            return
            
        self._send_msg(f"⏳ {ticker} kontol ediliyor...")
        
        # Gecici kontrol
        test_comps = self._init_companies([ticker])
        if not test_comps:
            self._send_msg(f"❌ {ticker} SEC veritabaninda bulunamadi (veya eslesmedi).")
            return
            
        self.active_tickers.append(ticker)
        self._save_active_tickers()
        
        # State guncelle
        self.companies = self._init_companies(self.active_tickers)
        self.target_ciks = {c.cik for c in self.companies}
        
        self._send_msg(f"✅ {ticker} listeye eklendi ve takibe alindi. (Toplam: {len(self.active_tickers)})")

    def _handle_remove(self, chat_id: int, ticker: str):
        if ticker not in self.active_tickers:
            self._send_msg(f"ℹ️ {ticker} listede degil.")
            return
            
        self.active_tickers.remove(ticker)
        self._save_active_tickers()
        
        # State guncelle
        self.companies = self._init_companies(self.active_tickers)
        self.target_ciks = {c.cik for c in self.companies}
        
        self._send_msg(f"🗑️ {ticker} listeden cikarildi.")

    def _handle_list(self, chat_id: int):
        if not self.active_tickers:
            self._send_msg("📭 Liste bos.")
            return
        
        # Cok uzunsa parcala
        tickers_str = ", ".join(sorted(self.active_tickers))
        if len(tickers_str) > 3500:
            parts = [tickers_str[i:i+3500] for i in range(0, len(tickers_str), 3500)]
            for p in parts:
                self._send_msg(f"📋 <b>Takip Listesi:</b>\n{p}")
        else:
            self._send_msg(f"📋 <b>Takip Listesi ({len(self.active_tickers)}):</b>\n{tickers_str}")

    def _handle_scan(self, chat_id: int, ticker: str):
        """Gecmis tarama islemi - data.sec.gov JSON API kullanarak"""
        self._send_msg(f"🔍 <b>Scanning history for {ticker}...</b>")
        try:
            # data.sec.gov API kullan (Ayrintili ve sirali)
            logger.info(f"Scanning history for {ticker} via JSON API...")
            
            # Sirketi bul
            comp = next((c for c in self.companies if c.ticker == ticker), None)
            cik = comp.cik if comp else self._get_cik_by_ticker(ticker)
            
            if not cik:
                self._send_msg(f"❌ {ticker} CIK bulunamadi.")
                return

            # Sirket ismini bul (mesajlarda kullanmak icin)
            comp_name = comp.name if comp else ticker

            headers = {
                "User-Agent": "Omer Yilmaz (omeryilmaz1329@gmail.com)",
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov"
            }
            
            # CIK'i 10 haneye tamamla
            cik_padded = str(cik).zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
            
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                self._send_msg(f"❌ SEC API hatasi: {r.status_code}")
                return
                
            data = r.json()
            filings = data['filings']['recent']
            
            forms = filings['form']
            dates = filings['filingDate']
            accession_numbers = filings['accessionNumber']
            primary_docs = filings['primaryDocument']
            
            # Son n dosyaya bakalim (Form 4/5 gibi noise dosyalar cok olabilir, bu yuzden limiti artiriyoruz)
            # Kullanici talebi uzerine: Tum formlar taranacak (ozellikle 3,4,5,144 gürültü oldugu icin eliyoruz)
            limit = min(100, len(forms)) # Daha fazla gecmise bakmak icin limit artirildi
            found_something = False
            
            for i in range(limit):
                form = forms[i]
                
                # Insider trading ve benzeri gürültü formlari eliyoruz
                if form in ['3', '4', '5', '144']:
                    continue
                # Eski whitelist: ['8-K', '424B5', '497', '10-K', 'SC TO'] - ARTIK KULLANILMIYOR
                
                acc_no = accession_numbers[i]
                primary_doc = primary_docs[i]
                filing_date = dates[i]
                
                # URL Olusturma (Log veya Link icin)
                acc_no_clean = acc_no.replace('-', '')
                cik_clean = str(int(cik)) 
                link = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_no_clean}/{primary_doc}"
                
                # ANA VE EK DOSYALARI CEK
                content = self._fetch_with_exhibits(cik, acc_no, primary_doc)
                if not content: continue
                
                detected = self._smart_keyword_detection(content)
                if detected:
                    ai_analysis = ""
                    if self.ai_analyzer.daily_analysis_count < self.ai_analyzer.max_daily_analysis:
                        ai_analysis = self.ai_analyzer.analyze_debt_redemption(comp_name, ticker, form, content)
                    
                    msg = f"🔎 <b>GEÇMİŞ TARAMA SONUCU</b>\n"
                    msg += f"<b>Sirket:</b> {comp_name} ({ticker})\n"
                    msg += f"<b>Tarih:</b> {filing_date}\n"
                    msg += f"<b>Form:</b> {form}\n"
                    msg += f"<b>Tespit:</b> {detected}\n"
                    
                    price_str = self._get_market_price(ticker)
                    msg += f"<b>Guncel Fiyat:</b> {price_str}\n"

                    if ai_analysis and "YOK" not in ai_analysis and "❌" not in ai_analysis:
                        msg += f"\n{ai_analysis}\n"
                    msg += f"<b>Link:</b> {link}"
                    
                    self._send_msg(msg)
                    found_something = True
                    break # En gunceli bulduysak duralim
            
            if not found_something:
                self._send_msg(f"ℹ️ {ticker} icin son dosyalarda olumlu gelisme (Redemption/Tender/Dividend) bulunamadi.")
                
        except Exception as e:
            logger.error(f"Scan error: {e}")
            self._send_msg("❌ Tarama sirasinda hata olustu.")

    def _get_cik_by_ticker(self, ticker: str) -> Optional[str]:
        try:
            r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=self.headers, timeout=10)
            if r.status_code == 200:
                for v in r.json().values():
                    if v['ticker'] == ticker:
                        return str(v['cik_str']).zfill(10)
        except: pass
        return None

    def run(self):
        self._test_mode()
        
        # Poller thread baslat
        t = threading.Thread(target=self._poll_commands, daemon=True)
        t.start()
        
        logger.info("Starting main loop...")
        while True:
            try:
                self.check_cnt += 1
                self._check_feed()
                
                # Gunluk rapor kontrolu
                if datetime.now().date() > self.last_daily_report:
                    self._send_daily_report()
                    
                time.sleep(15)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Global error: {e}")
                time.sleep(60)

    def _test_mode(self):
        # Kisa bir test
        logger.info("Running self-test...")
        test_text = "The company hereby gives notice of redemption for all outstanding 5.5% Senior Notes due 2025."
        res = self._smart_keyword_detection(test_text)
        if res: logger.info(f"Test Pass: {res}")
        else: logger.error("Test Fail")
    
    def _send_daily_report(self):
        uptime = datetime.now() - self.start
        days = uptime.days
        hours = uptime.seconds // 3600
        msg = f"🟢 <b>GUNLUK DURUM RAPORU</b>\n"
        msg += f"✅ Bot Calisiyor (Uptime: {days}g {hours}s)\n"
        msg += f"📡 Kontrol Sayisi: {self.check_cnt}\n"
        msg += f"⚡ Anlik Tespit: {self.daily_stats['instant_detections']}\n"
        msg += f"🧠 AI Analizi: {self.daily_stats['ai_analyzed']}/{self.ai_analyzer.max_daily_analysis}\n"
        self._send_msg(msg)
        self.last_daily_report = datetime.now().date()
        # Yeni gun icin limitleri sifirla
        self.daily_stats['instant_detections'] = 0
        self.daily_stats['ai_analyzed'] = 0
        self.check_cnt = 0
        self.ai_limit_notified = False # AI limit bildirimini sifirla

if __name__ == "__main__":
    # CONFIG - Environment Variables for Railway
    TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
    
    if not TOKEN or not CHAT:
        logger.warning("Telegram credentials not found in environment variables!")
    
    # Tickers listesi cok uzun oldugu icin burada tutuyoruz, ama isterseniz bunu da bir JSON dosyasindan okutabiliriz.
    # Railway'de kodun icinde kalmasinda sakinca yok.
    TICKERS = ["ABL", "ABR", "ACGL", "ACP", "ACR", "ADC", "AD", "ADAM", "AEG", "AFG", "AGM", "AGNC", "AHH", "AHT", "AIRT", "ALL", "ALTG", "AMG"] # Kalanlar buraya...
    
    # Kullanicinin orijinal listesi cok uzundu, yer tutmasin diye kisa kestim ama 
    # gercek kullanimda kullanicinin verdigi listenin tamami olmali.
    # Orijinal listedeki tum tickerlari buraya eklemem lazim.
    # Kodun okunabilirligi icin "..." koydum, ama simdi geri ekleyecegim.
    
    FULL_TICKER_LIST = [
        "ABL", "ABR", "ACGL", "ACP", "ACR", "ADC", "AD", "ADAM", "AERT", "AEG", "AFG", "AGM", "AGNC",
    "AHH", "AHT", "AIRT", "ALL", "ALTG", "AMG", "AMH", "APO", "AOMR", "AON", "ARES", "ARR", "ASB",
    "ATLC", "AXS", "BANF", "BANC", "BC", "BCV", "BFS", "BHF", "BK", "BOH", "BPOP", "BW", "BUSE",
    "CADE", "CCAP", "CCIF", "CCLD", "CFR", "CG", "CGBD", "CHMI", "CIM", "CIO", "CION", "CLDT", "CMRE",
    "CMS", "CNFR", "CNOB", "COF", "CODI", "CRBG", "CSR", "CSWC", "CTA", "CTO", "CUBI", "CYCC", "D",
    "DBRG", "DCOM", "DDS", "DHC", "DLNG", "DLR", "DRH", "DSX", "DT", "DUK", "DX", "EFC", "EIC",
    "EIX", "EQH", "EPR", "ETR", "F", "FAT", "FCNCA", "FGBI", "FG", "FGNX", "FITB", "FLG", "FOUR",
    "FOSL", "FRME", "FRT", "FHN", "GAIN", "GAB", "GAM", "GDV", "GECC", "GEG", "GL", "GLAD", "GLP",
    "GMRE", "GOOD", "GPMT", "GREE", "GRBK", "GUT", "GWT", "GPUS", "HIG", "HL", "HNNA", "HPP", "HROW",
    "HWM", "HRZN", "HTGC", "IIPR", "IMPP", "INBK", "INN", "JXN", "KEY", "KIM", "KKR", "KREF", "LAND",
    "LBRDA", "LFT", "LFMD", "LNC", "LUMN", "LXP", "MBIN", "MCHP", "MDRR", "MDV", "MET", "METC",
    "METCZ", "MFIC", "MFIN", "MITT", "MIND", "MNSB", "MSTR", "MTB", "NAV", "NAVI", "NCV", "NCZ",
    "NEE", "NEWT", "NLY", "NREF", "NMFC", "NR", "NTRS", "NSA", "NXDT", "NYMT", "OCFC", "OFS", "ONB",
    "OPI", "OXLC", "OZK", "PCG", "PBI", "PBX", "PEB", "PDCC", "PG", "PMT", "POWW", "PRU", "PSA", "PW",
    "PXS", "QVCGA", "QXO", "RGA", "REG", "REXR", "RF", "RIL", "RITM", "RLJ", "RNR", "RPT", "RWT",
    "RJF", "RILY", "RWAY", "SB", "SACH", "SAR", "SAT", "SCHW", "SIGI", "SITC", "SLG", "SLM", "SLNH",
    "SNV", "SO", "SOHO", "SPMC", "SPNT", "SQFT", "SR", "SRG", "SSSS", "STNG", "STT", "STRR", "SWK",
    "SWKH", "SYF", "TCBI", "T", "TDS", "TEN", "TFIN", "TFC", "TMUS", "TPG", "TRIN", "TRTX", "TWO",
    "UCB", "UMH", "UNM", "USB", "VLY", "VNO", "VOYA", "WAFD", "WAL", "WBS", "WBSC", "WHF", "WHLR",
    "WRB", "WSBC", "WTFC", "XOMA", "YCBD", "ZION","AIZ", "AUB", "BGMS", "BX", "CMA", "ECC", "ECF", "EFSC",
    "FBIO", "FULT", "GGN", "HE", "HFRO", "HOV", "KMPR",
    "LOB", "MGR", "MSBI", "PNFP", "RC"
    ]

    # Konsol ciktisi icin UTF-8 ayari
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    print("[INFO] SEC Bot Running (Parallel + Smart AI)...")
    bot = SECCallMonitor(TOKEN, CHAT, FULL_TICKER_LIST)
    bot.run()
