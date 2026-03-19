import os
import requests
import time

# 预置的顶级名画直链
ARTWORKS = {
    "vangogh_starry_night.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "monet_water_lilies.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Claude_Monet_-_Water_Lilies_-_Google_Art_Project.jpg/1024px-Claude_Monet_-_Water_Lilies_-_Google_Art_Project.jpg",
    "rembrandt_night_watch.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/The_Night_Watch_-_HD.jpg/1024px-The_Night_Watch_-_HD.jpg",
    "davinci_mona_lisa.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/1024px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
    "vermeer_pearl_earring.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/1024px-1665_Girl_with_a_Pearl_Earring.jpg",
    "klimt_the_kiss.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/The_Kiss_-_Gustav_Klimt_-_Google_Cultural_Institute.jpg/1024px-The_Kiss_-_Gustav_Klimt_-_Google_Cultural_Institute.jpg",
    "hokusai_great_wave.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century_classic_ukiyo-e_japanese_woodblock_print_%20reproduction.jpg/1024px-Tsunami_by_hokusai_19th_century_classic_ukiyo-e_japanese_woodblock_print_%20reproduction.jpg",
    "renoir_luncheon.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Pierre-Auguste_Renoir_-_Le_Moulin_de_la_Galette.jpg/1024px-Pierre-Auguste_Renoir_-_Le_Moulin_de_la_Galette.jpg",
    "botticelli_venus.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg/1024px-Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg",
    "munch_scream.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/1024px-The_Scream.jpg"
}

# 终极伪装：不仅带上 Chrome 的 User-Agent，还加上 Accept 格式和来源页 (Referer)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://commons.wikimedia.org/"
}

def download_image(name, url):
    save_path = os.path.join("art_database", name)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        print(f"   [SKIP] {name} 已存在且完整。")
        return
    
    print(f"   [DOWNLOADING] 正在潜行获取: {name}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"   ✅ 成功保存: {name}")
    except Exception as e:
        print(f"   [ERROR] 获取失败: {e}")

if __name__ == "__main__":
    os.makedirs("art_database", exist_ok=True)
    print("🚀 开始构建物料池（终极防拦截模式）...")
    
    for name, url in ARTWORKS.items():
        download_image(name, url)
        time.sleep(2) # 强行等待 2 秒，安全第一
        
    print("🎉 下载进程结束！")